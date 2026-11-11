#!/usr/bin/env python3
"""Eight-bit SOURCE fidelity on two consumed inputs, not a new quality cohort.

The tiny CUDA operator must pass before checkpoint loading. The original
three-bit parent/protection construction remains intact; only its scoped
transport-fit callback creates the isolated eight-bit control. FullKV-generated
tokens enter decoding only. Actual storage is reported against raw K/V bytes,
not 40 MiB or a claim of matched storage with another method.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110/source-prefix")
CAUSAL = Path("/scratch/djy8hg/workdir/fox-review-v11/run-20261110-1/causal")
LABEL = "evicastkv_eight_bit_shared_source_fidelity_control"
POLICY = dict(
    label=LABEL,
    source_bits=8,
    parent_source_bits=3,
    optional_visual_fraction=0.05,
    protection="unchanged_attention_optional_all_text_first4_last8",
    predictions=128,
    decode_updates=127,
    seed=13,
    native_repeat_atol=0.001,
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    operator=dict(rows=131, heads=2, dim=128, extra_lease_rows=4, seed=321, rtol=0.01, atol=2**-6),
    soft_worker_seconds=1100,
    maximum_worker_seconds=1200,
    initial_ceiling="measured_original_raw_FullKV_KV_backing_per_input",
    require_initial_strictly_smaller_than_raw=True,
    history="unchanged_direct_lossless_source_history",
    replay=False,
    no_reference_tokens_or_logits_in_constructor=True,
    no_benchmark_labels=True,
    no_shared_budget_claim=True,
    no_latency_claim=True,
    independent_quality_N=0,
)


def read(path):
    return json.loads(Path(path).read_text())


def operator_before_model(operator, loader):
    result = operator()
    if result.get("passed") is not True:
        raise ValueError("actual tiny CUDA source operator must pass before checkpoint loading")
    return result, loader()


def cuda_operator_control():
    """Real CUDA kernel, native eight-bit oracle, protected bits and tail sentinel."""
    import torch

    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.exact_storage import decoded
    from mosaickv.evaluation.precision_source_control import construct, decode_source

    specification = POLICY["operator"]
    n, h, d = specification["rows"], specification["heads"], specification["dim"]
    generator = torch.Generator().manual_seed(specification["seed"])
    x = (torch.randn(1, n, h * d, generator=generator) * 0.2).bfloat16().cuda()
    projections = [torch.nn.Linear(h * d, h * d, bias=False).bfloat16().cuda() for _ in range(2)]
    for projection in projections:
        projection.weight.copy_(
            (torch.randn(projection.weight.shape, generator=generator) * 0.2).bfloat16().cuda()
        )
    attention = SimpleNamespace(k_proj=projections[0], v_proj=projections[1])
    theta = torch.arange(n, device="cuda").float()[None, :, None] * 0.01
    cos, sin = (
        theta.cos().expand(1, n, d).bfloat16().contiguous(),
        theta.sin().expand(1, n, d).bfloat16().contiguous(),
    )
    key, value = (projection(x).reshape(1, n, h, d).transpose(1, 2) for projection in projections)
    rotated = torch.cat((-key[..., d // 2 :], key[..., : d // 2]), -1)
    pair = ((key * cos[:, None] + rotated * sin[:, None]).contiguous(), value.contiguous())
    mandatory = torch.zeros(h, n, dtype=torch.bool)
    mandatory[:, [0, 1, n - 2, n - 1]] = True
    identity = (
        torch.arange(n, dtype=torch.int32, device="cuda"),
        torch.arange(n, dtype=torch.int16, device="cuda")[None].expand(h, n),
    )
    with patch.object(shared_source_format, "protected_positions", lambda *a, **k: mandatory):
        parent = shared_source_format.construct(
            x,
            pair,
            (tuple(range(n)),) * h,
            (1,),
            (cos, sin),
            identity,
            exclude_global_exact=False,
        )
    immutable = (
        x,
        attention.k_proj.weight,
        attention.v_proj.weight,
        parent.source.codes,
        parent.cos,
        parent.sin,
    )
    before = tuple(item.clone().contiguous() for item in immutable)
    candidate, detail = construct(parent, x, attention)
    native_x = decode_source(candidate.source)
    native_k, native_v = (
        projection(native_x).reshape(1, n, h, d).transpose(1, 2) for projection in projections
    )
    native_rotated = torch.cat((-native_k[..., d // 2 :], native_k[..., : d // 2]), -1)
    expected = (
        (native_k * cos[:, None] + native_rotated * sin[:, None]).contiguous(),
        native_v.contiguous(),
    )
    rows = parent.row_positions.long()
    for target, exact in zip(expected, parent.exact, strict=True):
        target[0, rows // n, rows % n] = decoded(exact)[0, 0]
    sentinel = -123.0
    destinations = tuple(
        torch.full(
            (1, h, n + specification["extra_lease_rows"], d),
            sentinel,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(2)
    )
    candidate.fill(attention, destinations, direct=True)
    maximum = 0.0
    for target, reference, original in zip(destinations, expected, pair, strict=True):
        prompt = target[..., :n, :]
        torch.testing.assert_close(
            prompt, reference, rtol=specification["rtol"], atol=specification["atol"]
        )
        maximum = max(maximum, float((prompt.float() - reference.float()).abs().max()))
        if not torch.equal(
            prompt[0, rows // n, rows % n].contiguous().view(torch.int16),
            original[0, rows // n, rows % n].contiguous().view(torch.int16),
        ) or not bool((target[..., n:, :] == sentinel).all()):
            raise RuntimeError("eight-bit operator altered protected rows or lease-tail sentinel")
    if not all(
        torch.equal(old.view(torch.uint8), value.contiguous().view(torch.uint8))
        for old, value in zip(before, immutable, strict=True)
    ):
        raise RuntimeError("eight-bit operator mutated projection weights or original source")
    torch.cuda.synchronize()
    return dict(
        passed=True,
        actual_CUDA_operator_executed=True,
        geometry=specification,
        source_bits=8,
        parent_still_three_bits=parent.source.bits == 3,
        native_eight_bit_oracle_checked=True,
        maximum_absolute_projection_error=maximum,
        protected_rows_bitwise_original=True,
        strided_lease_tail_sentinel_unchanged=True,
        numerical_patches_verified=True,
        projection_weights_and_original_source_unchanged=True,
        constructor=detail,
        pretrained_checkpoint_loaded=False,
    )


@contextmanager
def bounded_adapter(adapter, deadline):
    def wrap(function):
        def call(*args, **kwargs):
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded eight-bit source diagnostic soft deadline reached")
            return function(*args, **kwargs)

        return call

    with (
        patch.object(adapter, "prefill", wrap(adapter.prefill)),
        patch.object(adapter, "decode_one_token", wrap(adapter.decode_one_token)),
    ):
        yield


def eight_bit_formats(
    adapter,
    prepared,
    prefill,
    captured,
    normalized,
    phases,
    *,
    builder=None,
    constructor=None,
    deadline=None,
):
    """No labels, generated prefix, or reference outputs are accepted here."""
    import evicastkv_review_source_controls as source
    import torch

    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.precision_source_control import construct

    builder = source._formats if builder is None else builder
    constructor = construct if constructor is None else constructor
    layers = tuple(adapter._language_layers())
    indices = {id(layer.self_attn): index for index, layer in enumerate(layers)}
    if len(indices) != len(layers) or set(normalized.records) != set(range(len(layers))):
        raise ValueError("one unique original attention and normalized source per layer required")
    records = []

    def fit(parent, attention):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("eight-bit construction soft deadline reached")
        if id(attention) not in indices:
            raise ValueError("transport callback attention is not a registered model layer")
        index = indices[id(attention)]
        before = int(torch.cuda.memory_allocated()) if normalized.records[index].is_cuda else None
        packet, detail = constructor(parent, normalized.records[index], attention)
        records.append(
            dict(
                layer=index,
                detail=detail,
                persistent_bytes=backing_bytes(packet.tensors()),
                global_allocator_allocated_before=before,
                global_allocator_allocated_after=int(torch.cuda.memory_allocated())
                if before is not None
                else None,
                global_allocator_peak_at_exit=int(torch.cuda.max_memory_allocated())
                if before is not None
                else None,
                allocator_envelope_is_not_exclusive_constructor_KV_peak=True,
            )
        )
        return packet, detail

    with patch.object(shared_source_format, "fit_transport", fit):
        formats, proof = builder(
            adapter, prepared, prefill, captured, normalized, phases, "source3_attention_optional"
        )
    if sorted(item["layer"] for item in records) != list(range(len(layers))):
        raise ValueError(
            "every original model layer must receive exactly one eight-bit constructor"
        )
    if any(packet.source.bits != 8 or packet.kind != LABEL for packet in formats):
        raise ValueError("silent fallback or wrong precision source format")
    return formats, proof, records


def request(adapter, payload, config, *, prefix, expected_protection, deadline=None):
    import evicastkv_review_source_controls as source

    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    if len(prefix) != 128 or any(type(value) is not int or value < 0 for value in prefix):
        raise ValueError("complete fixed FullKV prefix required for decode only")
    old_builder, old_run = source._formats, storage_runs.run
    evidence = {}

    def build(*args):
        pairs, _ = adapter._legacy_layers(args[2].state.past_key_values)
        evidence["raw_FullKV_ceiling_bytes"] = backing_bytes(
            tuple(t for pair in pairs for t in pair)
        )
        formats, proof, records = eight_bit_formats(
            *args[:6], builder=old_builder, deadline=deadline
        )
        actual = [(item["layer"], item["mask_sha256"]) for item in proof]
        if actual != [(item["layer"], item["mask_sha256"]) for item in expected_protection]:
            raise ValueError("eight-bit control changed the frozen original protection masks")
        evidence.update(construction_records=records, protection_mask_digests=actual)
        return formats, proof

    def decode(*args, **kwargs):
        if "forced" in kwargs or "initial_physical_byte_cap" in kwargs:
            raise ValueError("nested conditioning or byte-cap policy is forbidden")
        kwargs.update(forced=prefix, initial_physical_byte_cap=evidence["raw_FullKV_ceiling_bytes"])
        return old_run(*args, **kwargs)

    with patch.object(source, "_formats", build), patch.object(storage_runs, "run", decode):
        logits, record = source._request(adapter, payload, config, "source3_attention_optional")
    initial = record["storage"]["initial_active_kv_storage_bytes"]
    if (
        record["used_fullkv_fallback"]
        or not record["inactive_original_source_retired"]
        or not initial < evidence["raw_FullKV_ceiling_bytes"]
        or record["raw_source_kv_bytes"] != evidence["raw_FullKV_ceiling_bytes"]
    ):
        raise ValueError(
            "eight-bit source did not retire originals with meaningful persistent savings"
        )
    record.update(
        method=LABEL,
        source_bits=8,
        evidence=evidence,
        initial_smaller_than_raw_FullKV=True,
        source_fallback=False,
        fixed_prefix_decode_only=True,
        constructor_reference_access=False,
        actual_realized_bytes_not_shared_budget=True,
        whole_request_KV_peak_certified=False,
        whole_device_memory_certified=False,
        construction_workspace_in_global_allocator_envelope_not_exclusive_KV_certificate=True,
    )
    return logits, record


def verify_row_accounting(row, job):
    """Check observed bytes and complete decode history, not a producer flag alone."""
    storage, evidence = row["storage"], row["evidence"]
    raw, initial = row["raw_source_kv_bytes"], storage["initial_active_kv_storage_bytes"]
    final, peak = storage["final_active_kv_storage_bytes"], storage["peak_active_kv_storage_bytes"]
    if (
        row["index"] != job["index"]
        or row["sample_id"] != job["sample_id"]
        or row["task"] != job["task"]
        or row["method"] != LABEL
        or row["source_bits"] != 8
        or row["used_fullkv_fallback"] is not False
        or row["source_fallback"] is not False
        or row["inactive_original_source_retired"] is not True
        or type(raw) is not int
        or type(initial) is not int
        or not 0 < initial < raw
        or type(final) is not int
        or type(peak) is not int
        or not 0 < final <= peak
        or initial > peak
        or evidence["raw_FullKV_ceiling_bytes"] != raw
        or storage["history_updates_per_layer"] != [127] * 32
    ):
        raise ValueError("actual source bytes, original retirement or complete history are invalid")
    records = evidence["construction_records"]
    if sorted(record["layer"] for record in records) != list(range(32)) or any(
        record["detail"]["source_bits"] != 8
        or record["detail"]["protected_original_KV_bitwise"] is not True
        or record["detail"]["original_logical_positions_preserved"] is not True
        or record["detail"]["no_original_source_alias"] is not True
        for record in records
    ):
        raise ValueError("complete bit-preserving eight-bit construction evidence required")
    if list(map(tuple, evidence["protection_mask_digests"])) != [
        (record["layer"], record["mask_sha256"]) for record in job["expected_protection"]
    ]:
        raise ValueError("frozen protection evidence differs")
    timings = row["phase_wall_seconds"]
    names = {
        "media_and_processor",
        "prefill_with_diagnostic_captures",
        "construction",
        "retire_original_sources",
        "decode_including_capture_and_retirement",
    }
    if not names <= set(timings) or any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        for value in timings.values()
    ):
        raise ValueError("finite complete instrumentation timing fields required")
    return True


def prepare(root, parent=PARENT, causal=CAUSAL):
    import evicastkv_review_source_prefix as fixed

    root, parent, causal = Path(root), Path(parent), Path(causal)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch precision-control root required")
    gate = read(causal / "GATE_source_causal_control.json")
    if gate["status"] != "passed_causal_capture" or gate["failures"] or gate["rows"] != 8:
        raise ValueError(
            "complete causal K/V controls required; their negative joint result is retained"
        )
    if sha(causal / "causal_summary.json") != gate["summary_sha256"]:
        raise ValueError("causal summary binding changed")
    causal_summary = read(causal / "causal_summary.json")
    raw_controls = [
        item for item in causal_summary["rows"] if item["arm"] == "raw_prompt_K_raw_prompt_V"
    ]
    if (len(raw_controls) != 2 or {item["input_index"] for item in raw_controls} != {0, 1}) or any(
        item["diagnostics"]["prefix_128"]["token_agreement"] != 1.0
        or item["diagnostics"]["prefix_128"]["maximum_logit_difference"] > 0.001
        for item in raw_controls
    ):
        raise ValueError("both full-128 raw/raw causal controls must validate")
    old = read(parent / "plan.json")
    jobs, bindings = [], {str(parent / "plan.json"): sha(parent / "plan.json")}
    for index, old_job in enumerate(old["jobs"]):
        payload = fixed.label_blind_payload(old_job["inputs"][0])
        if sha(Path(old_job["inputs"][0])) != old_job["input_sha256"] or any(
            sha(Path(item["path"])) != item["sha256"] for item in payload["media"]
        ):
            raise ValueError("immutable consumed input/media bindings changed")
        result = read(parent / f"workers/{index}/result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("completed two-input consumed native references required")
        config = json.loads(json.dumps(old_job.get("config", old["config"])))
        config["execution"]["seed"] = 13
        rows = [
            json.loads(line)
            for line in (parent / f"workers/{index}/raw.jsonl").read_text().splitlines()
        ]
        records = [
            row
            for row in rows
            if row["arm"] == "source3_attention_optional" and row["status"] == "completed"
        ]
        if len(records) != 1:
            raise ValueError("one frozen three-bit attention protection record required")
        jobs.append(
            dict(
                index=index,
                config=config,
                input=old_job["inputs"][0],
                sample_id=payload["sample_id"],
                task=payload["task"],
                expected_protection=[
                    dict(layer=item["layer"], mask_sha256=item["mask_sha256"])
                    for item in records[0]["protection_records"]
                ],
            )
        )
        for name, expected in result["outputs"].items():
            path = parent / f"workers/{index}" / name
            if sha(path) != expected:
                raise ValueError("parent native/control artifacts changed")
            bindings[str(path)] = expected
        for path in [
            parent / f"workers/{index}/result.json",
            Path(old_job["inputs"][0]),
            *(Path(item["path"]) for item in payload["media"]),
        ]:
            bindings[str(path)] = sha(path)
    if len(jobs) != 2 or {job["task"] for job in jobs} != {"llava_description", "mmvet_prefixkv"}:
        raise ValueError("only the two original consumed Description/MM-Vet inputs are registered")
    for path in [
        causal / "GATE_source_causal_control.json",
        causal / "causal_summary.json",
        Path(__file__),
        SOURCE / "mosaickv/src/mosaickv/evaluation/precision_source_control.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/shared_source_kernel.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_controls.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_prefix.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
    ]:
        bindings[str(path)] = sha(path)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=clean_source(SOURCE),
            source_root=str(SOURCE),
            parent=str(parent),
            causal_parent=str(causal),
            bindings=bindings,
            jobs=jobs,
            policy=POLICY,
            worker_GPUs=1,
            worker_CPUs=12,
            worker_minutes=20,
            expected_rows=2,
            method=LABEL,
            no_paper_or_fresh_campaign=True,
        ),
    )
    return dict(workers=1, expected_rows=2, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY):
        raise ValueError("eight-bit fidelity policy changed")
    if any(sha(Path(path)) != expected for path, expected in plan["bindings"].items()):
        raise ValueError("immutable precision-control evidence/source binding changed")
    return plan


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        MODEL,
        comparisons,
        forced_prefix,
        label_blind_payload,
        native_reference,
    )
    from evicastkv_source_protection_diagnostic import require_isolated_slurm_gpu

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs

    root = Path(root)
    plan = validate(root)
    if index != 0:
        raise ValueError("one bounded GPU worker for two consumed inputs only")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], attempted_samples=2)
    rows, controls, saved = [], [], {}
    started = time.monotonic()
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        configs = [RunConfig.from_mapping(job["config"]) for job in plan["jobs"]]
        if any(
            (cfg.model.id, cfg.model.revision) != MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
            for cfg in configs
        ):
            raise ValueError("same pinned BF16 eager full-MHA LLaVA checkpoint required")

        def load_model():
            torch.manual_seed(13)
            gc.collect()
            torch.cuda.empty_cache()
            return load_hf_adapter(
                MODEL[0],
                revision=MODEL[1],
                model_kwargs=_model_kwargs(configs[0], local_files_only=True),
                processor_kwargs={"local_files_only": True},
            )

        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                protocol_sha256=sha(root / "protocol.json"),
                method=LABEL,
                model=dataclasses.asdict(configs[0].model),
                processor_revision=MODEL[1],
                per_input_configs=[
                    dict(config_sha256=config_sha256(cfg), dataset=dataclasses.asdict(cfg.dataset))
                    for cfg in configs
                ],
                environment=environment(),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                GPU=gpu,
                cuda_runtime=torch.version.cuda,
                vllm="not_used",
                sglang="not_used",
                backend="huggingface",
                attention="eager",
                seed=13,
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                measurement_type="consumed_source_precision_fidelity_control",
                result_category="unified_comparison",
                result_origin="locally_measured",
                benchmark_labels_read=False,
                latency_comparison=False,
            ),
        )

        def operator():
            try:
                with torch.inference_mode():
                    outcome = cuda_operator_control()
            except Exception as error:
                write_once(
                    out / "GATE_cuda_operator.json",
                    dict(
                        passed=False,
                        error=f"{type(error).__name__}: {error}",
                        pretrained_checkpoint_loaded=False,
                        no_pretrained_request_executed=True,
                    ),
                )
                raise
            write_once(out / "GATE_cuda_operator.json", outcome)
            return outcome

        _, adapter = operator_before_model(operator, load_model)
        with (
            torch.inference_mode(),
            bounded_adapter(adapter, started + POLICY["soft_worker_seconds"]),
        ):
            for job, config in zip(plan["jobs"], configs, strict=True):
                row = dict(
                    index=job["index"],
                    sample_id=job["sample_id"],
                    task=job["task"],
                    method=LABEL,
                    status="failed",
                    config_sha256=config_sha256(config),
                    source_sha=plan["source_sha"],
                )
                try:
                    payload = label_blind_payload(job["input"])
                    parent_logits = torch.load(
                        Path(plan["parent"]) / f"workers/{job['index']}/logits.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    reference = parent_logits["fullkv_native_reference"]
                    native, native_record = native_reference(adapter, payload)
                    native_check = comparisons(reference, native)
                    if (
                        native_check["prefix_128"]["token_agreement"] != 1.0
                        or native_check["prefix_128"]["maximum_logit_difference"]
                        > POLICY["native_repeat_atol"]
                        or float((reference[:16].double() - native[:16].double()).abs().max())
                        > POLICY["native_repeat_atol"]
                    ):
                        raise ValueError(
                            "native FullKV repeat failed unchanged token/logit tolerance"
                        )
                    controls.append(
                        dict(index=job["index"], diagnostics=native_check, record=native_record)
                    )
                    saved[f"native_{job['index']}"] = native
                    prefix = forced_prefix(reference)
                    logits, record = request(
                        adapter,
                        payload,
                        config,
                        prefix=prefix,
                        expected_protection=job["expected_protection"],
                        deadline=started + POLICY["soft_worker_seconds"],
                    )
                    saved[f"eight_bit_{job['index']}"] = logits
                    row.update(
                        record,
                        diagnostics=comparisons(reference, logits),
                        parent_three_bit_diagnostics=comparisons(
                            reference, parent_logits["source3_attention_optional"]
                        ),
                        native_repeat_control=native_check,
                        reference_prefix_sha256=digest(prefix),
                        status="completed",
                    )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    result["failures"].append(dict(index=job["index"], error=row["error"]))
                rows.append(row)
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        for job in plan["jobs"][len(rows) :]:
            rows.append(
                dict(
                    index=job["index"],
                    sample_id=job["sample_id"],
                    task=job["task"],
                    method=LABEL,
                    status="blocked_before_inference",
                    error="earlier operator/setup failure",
                )
            )
        with (out / "raw.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_once(out / "native_controls.json", controls)
        torch.save(saved, out / "logits.pt")
        result.update(
            status="completed"
            if len(rows) == 2 and all(row["status"] == "completed" for row in rows)
            else "failed",
            rows=len(rows),
            completed_samples=sum(row["status"] == "completed" for row in rows),
            wall_seconds=time.monotonic() - started,
        )
        result["outputs"] = {path.name: sha(path) for path in out.iterdir() if path.is_file()}
        write_once(out / "result.json", result)
    return result


def analyze(root):
    import torch
    from evicastkv_review_source_prefix import (
        analysis_cpu_controls,
        comparisons,
        verify_recomputed_diagnostics,
    )

    root = Path(root)
    plan = validate(root)
    analysis_cpu_controls()
    result = read(root / "worker/result.json")
    if result["source_sha"] != plan["source_sha"]:
        raise ValueError("precision-control worker source differs")
    for name, expected in result["outputs"].items():
        if sha(root / "worker" / name) != expected:
            raise ValueError("precision-control raw artifacts changed")
    rows = [json.loads(line) for line in (root / "worker/raw.jsonl").read_text().splitlines()]
    saved = torch.load(root / "worker/logits.pt", weights_only=True, map_location="cpu")
    failures, verification = list(result["failures"]), []
    operator_path = root / "worker/GATE_cuda_operator.json"
    if operator_path.exists():
        operator = read(operator_path)
        required = [
            "passed",
            "actual_CUDA_operator_executed",
            "parent_still_three_bits",
            "native_eight_bit_oracle_checked",
            "protected_rows_bitwise_original",
            "strided_lease_tail_sentinel_unchanged",
            "numerical_patches_verified",
            "projection_weights_and_original_source_unchanged",
        ]
        if (
            any(operator.get(key) is not True for key in required)
            or operator.get("source_bits") != 8
            or operator.get("geometry") != POLICY["operator"]
        ):
            failures.append(
                dict(error="actual CUDA operator gate did not pass all registered controls")
            )
    else:
        failures.append(dict(error="actual CUDA operator gate unavailable"))
    if len(rows) != 2 or sorted(row["index"] for row in rows) != [0, 1]:
        raise ValueError("exactly the two registered candidate rows must be retained")
    for row in rows:
        if row["status"] != "completed":
            continue
        verify_row_accounting(row, plan["jobs"][row["index"]])
        parent_logits = torch.load(
            Path(plan["parent"]) / f"workers/{row['index']}/logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        reference = parent_logits["fullkv_native_reference"]
        verification.append(
            verify_recomputed_diagnostics(
                comparisons(reference, saved[f"eight_bit_{row['index']}"]), row["diagnostics"]
            )
        )
        verification.append(
            verify_recomputed_diagnostics(
                comparisons(reference, parent_logits["source3_attention_optional"]),
                row["parent_three_bit_diagnostics"],
            )
        )
        verification.append(
            verify_recomputed_diagnostics(
                comparisons(reference, saved[f"native_{row['index']}"]),
                row["native_repeat_control"],
            )
        )
    complete = len(rows) == 2 and all(row["status"] == "completed" for row in rows) and not failures
    gate = dict(
        status="completed_precision_fidelity_diagnostic"
        if complete
        else "blocked_precision_control",
        source_sha=plan["source_sha"],
        rows=len(rows),
        completed=sum(row["status"] == "completed" for row in rows),
        failures=failures,
        all_unchanged_32_token_guards_pass=complete
        and all(row["diagnostics"]["registered_guard_32"]["registered_guard_pass"] for row in rows),
        both_initial_caches_smaller_than_raw=complete
        and all(row["initial_smaller_than_raw_FullKV"] for row in rows),
        all_128_source_tokens_agree=complete
        and all(row["diagnostics"]["prefix_128"]["token_agreement"] == 1.0 for row in rows),
        numerical_recomputation_verifications=verification,
        all_points_reported=True,
        independent_quality_N=0,
        fresh_confirmation_authorized=False,
        no_shared_byte_comparison=True,
        no_latency_claim=True,
        whole_request_KV_peak_certified=False,
        superiority_established=False,
        production_three_bit_path_unchanged=True,
    )
    write_once(root / "GATE_source_precision_control.json", gate)
    write_once(root / "precision_summary.json", dict(rows=rows, policy=POLICY))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--causal", type=Path, default=CAUSAL)
    args = parser.parse_args()
    value = (
        prepare(args.root, args.parent, args.causal)
        if args.mode == "prepare"
        else globals()[args.mode](args.root)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
