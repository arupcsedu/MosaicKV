#!/usr/bin/env python3
"""Original-BF16-source fidelity on two consumed inputs; no superiority claim.

This independent lane never constructs a three-bit parent. Every prompt K/V
word is reconstructed from its original normalized source and exact bitpatches.
FullKV prefix tokens condition decoding only, never the constructor. Timings
include extra audits and must not be interpreted as a latency comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

from evicastkv_source_precision_control import bounded_adapter, operator_before_model, read

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110/source-prefix")
LABEL = "evicastkv_original_BF16_source_lossless_fidelity_control"
POLICY = dict(
    label=LABEL,
    source_precision="original_BF16_no_quantization",
    target="every_original_prefill_KV_word_not_three_bit_decoder",
    positions="all_original_prompt_positions",
    predictions=128,
    continuation_updates=127,
    seed=13,
    native_repeat_atol=0.001,
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    operator=dict(rows=131, heads=8, dim=128, extra_lease_rows=4, seed=321),
    initial_ceiling="actual_raw_FullKV_KV_backing",
    reject_persistent_cost_at_or_above_raw=True,
    source_history="unchanged_direct_lossless_source_history",
    reader="generic_two_buffer_shared_workspace_no_sequential_dispatch",
    constructor_audit="explicit_temporaries_plus_per_ATen_ScratchAudit",
    decode_workspace_audit=True,
    soft_worker_seconds=1100,
    worker_minutes=20,
    no_labels_or_reference_prefix_in_constructor=True,
    no_shared_byte_or_latency_claim=True,
    independent_quality_N=0,
)


def bitwise(left, right):
    import torch

    return torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


def lossless_formats(
    adapter, prepared, prefill, captured, normalized, phases, *, deadline=None, constructor=None
):
    """Original K/V are a lossless target, not benchmark/reference answers."""
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.lossless_source_control import KIND, construct, owner
    from mosaickv.evaluation.projection_cache import ScratchAudit, identity

    del prepared  # neither attention scores nor labels select any rows
    constructor = construct if constructor is None else constructor
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    layers = tuple(adapter._language_layers())
    if len(pairs) != len(layers) or set(normalized.records) != set(range(len(layers))):
        raise ValueError("complete original KV and normalized source for every layer required")
    if len({id(layer.self_attn) for layer in layers}) != len(layers):
        raise ValueError("unique original attention layer identities required")
    raw = backing_bytes(tuple(t for pair in pairs for t in pair))
    capture_tensors = []
    pending = [getattr(captured, field, {}) for field in ("records", "windows", "priors")]
    while pending:
        value = pending.pop()
        if isinstance(value, torch.Tensor):
            capture_tensors.append(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    originals = (
        *tuple(t for pair in pairs for t in pair),
        *normalized.records.values(),
        *phases.phases,
        *capture_tensors,
    )
    formats, records = [], []
    whole_observed = 0
    for index, (pair, layer) in enumerate(zip(pairs, layers, strict=True)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("bounded lossless construction soft deadline reached")
        positions = torch.arange(pair[0].shape[-2], device=pair[0].device, dtype=torch.int64)
        layer_peak = 0

        def observe(extras, positions=positions):
            nonlocal layer_peak, whole_observed
            live = (*originals, positions, *(t for p in formats for t in p.tensors()), *extras)
            value = backing_bytes(live)
            layer_peak = max(layer_peak, value)
            whole_observed = max(whole_observed, value)

        excluded = {identity(p.weight) for p in (layer.self_attn.k_proj, layer.self_attn.v_proj)}
        audit = ScratchAudit(excluded=excluded, observe=observe)
        with audit:
            packet = constructor(
                normalized.records[index],
                pair,
                layer.self_attn,
                phases.phases,
                positions,
                maximum_persistent_bytes=backing_bytes(pair),
                workspace_observer=observe,
            )
            # Independently verify constructor output before original owners retire.
            reconstructed = packet.materialize(layer.self_attn, workspace_observer=observe)
            if not all(bitwise(a, b) for a, b in zip(pair, reconstructed, strict=True)):
                raise ValueError("all-row original K/V reconstruction changed a bit")
        forbidden = {owner(t) for t in (*originals, positions)}
        forbidden.update(owner(p.weight) for p in (layer.self_attn.k_proj, layer.self_attn.v_proj))
        if packet.kind != KIND or any(owner(t) in forbidden for t in packet.tensors() if t.numel()):
            raise ValueError("lossless packet fallback or alias of originals/model weights")
        if not 0 < packet.component_bytes < backing_bytes(pair):
            raise ValueError("lossless packet is not strictly smaller than original layer K/V")
        records.append(
            dict(
                layer=index,
                detail=packet.diagnostics,
                all_original_KV_bitwise=True,
                no_original_or_weight_alias=True,
                positions_sha256=digest(positions.cpu().tolist()),
                persistent_bytes=packet.component_bytes,
                construction_observed_tensor_backing_peak_bytes=layer_peak,
                observed_ATen_operations=audit.operations,
                exact_whole_request_KV_peak_certified=False,
            )
        )
        formats.append(packet)
        del reconstructed, packet, audit
    actual = backing_bytes(tuple(t for p in formats for t in p.tensors()))
    if not 0 < actual < raw:
        raise ValueError("complete lossless persistent representation is not smaller than raw K/V")
    return (
        tuple(formats),
        records,
        dict(
            raw_FullKV_ceiling_bytes=raw,
            persistent_format_bytes=actual,
            unused_compatibility_attention_capture_bytes=backing_bytes(tuple(capture_tensors)),
            construction_observed_tensor_backing_peak_bytes=whole_observed,
            boundary="original_KV_and_captures_plus_formats_and_observed_constructor_temporaries",
            per_ATen_audit_does_not_certify_opaque_vendor_workspace=True,
            global_allocator_envelope_reported_separately=True,
            exact_whole_request_KV_peak_certified=False,
        ),
    )


def request(adapter, payload, config, *, prefix, deadline=None):
    import evicastkv_review_source_controls as source

    from mosaickv.evaluation import storage_runs

    if len(prefix) != 128 or any(type(token) is not int or token < 0 for token in prefix):
        raise ValueError("128 FullKV tokens for decode-only conditioning required")
    old_run, evidence = storage_runs.run, {}

    def build(*args):
        packets, records, accounting = lossless_formats(*args[:6], deadline=deadline)
        evidence.update(accounting, construction_records=records)
        return packets, records

    def decode(*args, **kwargs):
        if "forced" in kwargs or "initial_physical_byte_cap" in kwargs:
            raise ValueError("nested conditioning or byte-cap override forbidden")
        kwargs.update(
            forced=prefix,
            initial_physical_byte_cap=evidence["raw_FullKV_ceiling_bytes"],
            audit=True,
        )
        return old_run(*args, **kwargs)

    with patch.object(source, "_formats", build), patch.object(storage_runs, "run", decode):
        logits, record = source._request(adapter, payload, config, "source3_attention_optional")
    if (
        record["used_fullkv_fallback"]
        or not record["inactive_original_source_retired"]
        or record["raw_source_kv_bytes"] != evidence["raw_FullKV_ceiling_bytes"]
        or not 0
        < record["storage"]["initial_active_kv_storage_bytes"]
        < evidence["raw_FullKV_ceiling_bytes"]
    ):
        raise ValueError("lossless source original retirement/storage/fallback controls failed")
    record.update(
        method=LABEL,
        evidence=evidence,
        all_original_prompt_KV_bitwise=True,
        selection_or_reference_access=False,
        fixed_prefix_decode_only=True,
        source_fallback=False,
        actual_bytes_not_shared_budget=True,
        whole_request_KV_peak_certified=False,
        no_latency_claim=True,
    )
    return logits, record


def cuda_operator_control():
    """Execute original-native BF16 target and sparse repairs on actual CUDA."""
    from types import SimpleNamespace

    import torch

    from mosaickv.evaluation.lossless_source_control import construct, native_component, owner

    spec = POLICY["operator"]
    n, h, d = spec["rows"], spec["heads"], spec["dim"]
    g = torch.Generator().manual_seed(spec["seed"])
    x = (torch.randn(1, n, h * d, generator=g) * 0.1).bfloat16().cuda()
    projections = [
        torch.nn.Linear(h * d, h * d, bias=False, dtype=torch.bfloat16).eval().cuda()
        for _ in range(2)
    ]
    attention = SimpleNamespace(
        k_proj=projections[0], v_proj=projections[1], num_key_value_groups=1, training=False
    )
    theta = torch.arange(n, device=x.device).float()[None, :, None] * 0.01
    phases = tuple(t.expand(1, n, d).bfloat16().contiguous() for t in (theta.cos(), theta.sin()))
    positions = torch.arange(n, device=x.device, dtype=torch.int64)
    pair = tuple(native_component(x, attention, c, *phases).contiguous() for c in (0, 1))
    # Ensure real bitpatch application is checked rather than only an empty patch.
    pair[0][0, 1, 5, 7] += 1
    frozen = (x, *pair, *phases, positions, *(p.weight for p in projections))
    before = tuple(t.clone() for t in frozen)
    packet = construct(x, pair, attention, phases, positions)
    targets = tuple(
        torch.full(
            (1, h, n + spec["extra_lease_rows"], d), -123, dtype=torch.bfloat16, device=x.device
        )
        for _ in (0, 1)
    )
    packet.fill(attention, targets, direct=True)
    if (
        not all(bitwise(a, b[..., :n, :]) for a, b in zip(pair, targets, strict=True))
        or not all(bool((t[..., n:, :] == -123).all()) for t in targets)
        or not all(bitwise(a, b) for a, b in zip(before, frozen, strict=True))
        or set(owner(t) for t in packet.tensors() if t.numel()) & set(owner(t) for t in frozen)
    ):
        raise ValueError("actual CUDA lossless source bitwise/sentinel/noalias controls failed")
    torch.cuda.synchronize()
    return dict(
        passed=True,
        actual_CUDA_operator_executed=True,
        geometry=spec,
        every_original_KV_word_bitwise=True,
        real_numerical_patch_exercised=True,
        strided_lease_tail_sentinel_unchanged=True,
        no_original_or_weight_alias=True,
        original_inputs_and_weights_unchanged=True,
        pretrained_checkpoint_loaded=False,
        packet=packet.diagnostics,
    )


def prepare(root, *, cpu_gate_root=None, parent=PARENT):
    from evicastkv_review_source_prefix import label_blind_payload

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch lossless-control root required")
    revision = clean_source(SOURCE)
    bindings, jobs = {}, []
    old = read(parent / "plan.json")
    bindings[str(parent / "plan.json")] = sha(parent / "plan.json")
    for index, old_job in enumerate(old["jobs"]):
        payload = label_blind_payload(old_job["inputs"][0])
        result = read(parent / f"workers/{index}/result.json")
        if (
            result["status"] != "completed"
            or result["failures"]
            or sha(Path(old_job["inputs"][0])) != old_job["input_sha256"]
        ):
            raise ValueError("complete immutable consumed V7 controls required")
        for name, expected in result["outputs"].items():
            path = parent / f"workers/{index}" / name
            if Path(name).name != name or sha(path) != expected:
                raise ValueError("parent control artifact changed")
            bindings[str(path)] = expected
        for media in payload["media"]:
            if sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("consumed media changed")
            bindings[media["path"]] = media["sha256"]
        bindings[str(parent / f"workers/{index}/result.json")] = sha(
            parent / f"workers/{index}/result.json"
        )
        bindings[old_job["inputs"][0]] = old_job["input_sha256"]
        config = json.loads(json.dumps(old_job.get("config", old["config"])))
        config["execution"]["seed"] = 13
        jobs.append(
            dict(
                index=index,
                input=old_job["inputs"][0],
                config=config,
                sample_id=payload["sample_id"],
                task=payload["task"],
            )
        )
    if len(jobs) != 2 or {j["task"] for j in jobs} != {"llava_description", "mmvet_prefixkv"}:
        raise ValueError("only the two consumed V7 inputs are permitted")
    gate = None
    if cpu_gate_root is not None:
        from evicastkv_review_followups import cpu_gate

        gate = cpu_gate(Path(cpu_gate_root), revision)
        bindings[str(Path(cpu_gate_root) / "full-cpu/cpu-tests.json")] = sha(
            Path(cpu_gate_root) / "full-cpu/cpu-tests.json"
        )
    for path in (
        Path(__file__),
        SOURCE / "mosaickv/src/mosaickv/evaluation/lossless_source_control.py",
        SOURCE / "mosaickv/scripts/evicastkv_source_precision_control.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_controls.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_prefix.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
    ):
        bindings[str(path)] = sha(path)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent=str(parent),
            bindings=bindings,
            jobs=jobs,
            policy=POLICY,
            cpu_gate=gate,
            worker_GPUs=1,
            worker_CPUs=12,
            worker_minutes=20,
            expected_rows=2,
            no_paper_or_fresh_campaign=True,
        ),
    )
    return dict(workers=1, expected_rows=2, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or any(
        sha(Path(p)) != h for p, h in plan["bindings"].items()
    ):
        raise ValueError("immutable lossless source/config/evidence policy changed")
    return plan


def verify_row(row, job):
    storage, evidence = row["storage"], row["evidence"]
    raw, initial = row["raw_source_kv_bytes"], storage["initial_active_kv_storage_bytes"]
    final, peak = storage["final_active_kv_storage_bytes"], storage["peak_active_kv_storage_bytes"]
    if (
        row["index"] != job["index"]
        or row["sample_id"] != job["sample_id"]
        or row["task"] != job["task"]
        or row["method"] != LABEL
        or row["used_fullkv_fallback"] is not False
        or row["source_fallback"] is not False
        or row["inactive_original_source_retired"] is not True
        or type(raw) is not int
        or type(initial) is not int
        or not 0 < initial < raw
        or evidence["raw_FullKV_ceiling_bytes"] != raw
        or row["output_tokens"] != 128
        or storage["history_updates_per_layer"] != [127] * 32
        or type(final) is not int
        or type(peak) is not int
        or not 0 < final <= peak
        or not initial <= peak
    ):
        raise ValueError("actual lossless source retirement/history/storage controls failed")
    records = evidence["construction_records"]
    if sorted(r["layer"] for r in records) != list(range(32)) or any(
        r["all_original_KV_bitwise"] is not True
        or r["no_original_or_weight_alias"] is not True
        or r["detail"]["reference"] != "original_prefill_KV_not_lossy_parent_decoder"
        or not 0 < r["persistent_bytes"] < r["detail"]["raw_kv_bytes"]
        for r in records
    ):
        raise ValueError("every layer must verify all original prompt K/V words without alias")
    timings = row["phase_wall_seconds"]
    if not {
        "media_and_processor",
        "prefill_with_diagnostic_captures",
        "construction",
        "retire_original_sources",
        "decode_including_capture_and_retirement",
    } <= set(timings) or any(
        not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in timings.values()
    ):
        raise ValueError("complete finite diagnostic timing fields required")
    return True


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
        raise ValueError("one bounded lossless worker only")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    rows, controls, saved = [], [], {}
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"])
    start = time.monotonic()
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        configs = [RunConfig.from_mapping(j["config"]) for j in plan["jobs"]]
        if any(
            (c.model.id, c.model.revision) != MODEL
            or c.model.precision.value != "bf16"
            or c.execution.attention_implementation != "eager"
            for c in configs
        ):
            raise ValueError("same pinned BF16 eager full-MHA checkpoint required")
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
                    dict(config_sha256=config_sha256(c), dataset=dataclasses.asdict(c.dataset))
                    for c in configs
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
                measurement_type="consumed_original_BF16_source_fidelity_control",
                result_category="unified_comparison",
                result_origin="locally_measured",
                benchmark_labels_read=False,
                no_latency_comparison=True,
            ),
        )

        def operator():
            try:
                with torch.inference_mode():
                    value = cuda_operator_control()
            except Exception as error:
                write_once(
                    out / "GATE_cuda_operator.json",
                    dict(
                        passed=False,
                        error=f"{type(error).__name__}: {error}",
                        pretrained_checkpoint_loaded=False,
                    ),
                )
                raise
            write_once(out / "GATE_cuda_operator.json", value)
            return value

        def load():
            torch.manual_seed(13)
            gc.collect()
            torch.cuda.empty_cache()
            return load_hf_adapter(
                MODEL[0],
                revision=MODEL[1],
                model_kwargs=_model_kwargs(configs[0], local_files_only=True),
                processor_kwargs={"local_files_only": True},
            )

        _, adapter = operator_before_model(operator, load)
        with (
            torch.inference_mode(),
            bounded_adapter(adapter, start + POLICY["soft_worker_seconds"]),
        ):
            for job, cfg in zip(plan["jobs"], configs, strict=True):
                row = dict(
                    index=job["index"],
                    sample_id=job["sample_id"],
                    task=job["task"],
                    method=LABEL,
                    status="failed",
                    config_sha256=config_sha256(cfg),
                    source_sha=plan["source_sha"],
                )
                try:
                    payload = label_blind_payload(job["input"])
                    old = torch.load(
                        Path(plan["parent"]) / f"workers/{job['index']}/logits.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    ref = old["fullkv_native_reference"]
                    native, native_record = native_reference(adapter, payload)
                    check = comparisons(ref, native)
                    if (
                        check["prefix_128"]["token_agreement"] != 1.0
                        or check["prefix_128"]["maximum_logit_difference"]
                        > POLICY["native_repeat_atol"]
                        or float((ref[:16].double() - native[:16].double()).abs().max())
                        > POLICY["native_repeat_atol"]
                    ):
                        raise ValueError("native FullKV repeat token/logit control failed")
                    saved[f"native_{job['index']}"] = native
                    controls.append(
                        dict(index=job["index"], diagnostics=check, record=native_record)
                    )
                    prefix = forced_prefix(ref)
                    logits, record = request(
                        adapter,
                        payload,
                        cfg,
                        prefix=prefix,
                        deadline=start + POLICY["soft_worker_seconds"],
                    )
                    saved[f"lossless_{job['index']}"] = logits
                    row.update(
                        record,
                        diagnostics=comparisons(ref, logits),
                        native_repeat_control=check,
                        reference_prefix_sha256=digest(prefix),
                        status="completed",
                    )
                    verify_row(row, job)
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    if hasattr(error, "report"):
                        row["feasibility_decline"] = error.report
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
        torch.save(saved, out / "logits.pt")
        write_once(out / "native_controls.json", controls)
        result.update(
            status="completed" if all(r["status"] == "completed" for r in rows) else "failed",
            rows=len(rows),
            completed_samples=sum(r["status"] == "completed" for r in rows),
            wall_seconds=time.monotonic() - start,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        )
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
    if result["source_sha"] != plan["source_sha"] or any(
        Path(n).name != n or sha(root / "worker" / n) != h for n, h in result["outputs"].items()
    ):
        raise ValueError("lossless worker artifact/source binding changed")
    rows = [json.loads(line) for line in (root / "worker/raw.jsonl").read_text().splitlines()]
    if len(rows) != 2 or sorted(r["index"] for r in rows) != [0, 1]:
        raise ValueError("exactly the two registered candidate rows must be preserved")
    failures, verification = list(result["failures"]), []
    opath = root / "worker/GATE_cuda_operator.json"
    required = (
        "passed",
        "actual_CUDA_operator_executed",
        "every_original_KV_word_bitwise",
        "real_numerical_patch_exercised",
        "strided_lease_tail_sentinel_unchanged",
        "no_original_or_weight_alias",
        "original_inputs_and_weights_unchanged",
    )
    operator = read(opath) if opath.exists() else {}
    if (
        any(operator.get(k) is not True for k in required)
        or operator.get("geometry") != POLICY["operator"]
    ):
        failures.append(dict(error="actual CUDA lossless operator gate incomplete/failed"))
    saved = torch.load(root / "worker/logits.pt", weights_only=True, map_location="cpu")
    for row in rows:
        if row["status"] != "completed":
            continue
        verify_row(row, plan["jobs"][row["index"]])
        parent = torch.load(
            Path(plan["parent"]) / f"workers/{row['index']}/logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        for key, record in (
            (f"lossless_{row['index']}", row["diagnostics"]),
            (f"native_{row['index']}", row["native_repeat_control"]),
        ):
            verification.append(
                verify_recomputed_diagnostics(
                    comparisons(parent["fullkv_native_reference"], saved[key]), record
                )
            )
    complete = not failures and all(r["status"] == "completed" for r in rows)
    gate = dict(
        status="completed_lossless_source_diagnostic"
        if complete
        else "blocked_lossless_source_control",
        source_sha=plan["source_sha"],
        rows=len(rows),
        completed=sum(r["status"] == "completed" for r in rows),
        failures=failures,
        all_original_prompt_KV_bitwise=complete,
        all_unchanged_32_token_guards_pass=complete
        and all(r["diagnostics"]["registered_guard_32"]["registered_guard_pass"] for r in rows),
        all_128_source_tokens_agree=complete
        and all(r["diagnostics"]["prefix_128"]["token_agreement"] == 1.0 for r in rows),
        both_initial_caches_smaller_than_raw=complete,
        numerical_recomputation_verifications=verification,
        independent_quality_N=0,
        fresh_confirmation_authorized=False,
        whole_request_KV_peak_certified=False,
        no_shared_byte_comparison=True,
        no_latency_claim=True,
        superiority_established=False,
        production_three_bit_path_unchanged=True,
    )
    write_once(root / "GATE_lossless_source_control.json", gate)
    write_once(root / "lossless_summary.json", dict(rows=rows, policy=POLICY))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    args = parser.parse_args()
    value = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root)
        if args.mode == "prepare"
        else globals()[args.mode](args.root)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
