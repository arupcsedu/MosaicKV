#!/usr/bin/env python3
"""Consumed-input source/protection/replay controls, not a common-byte claim.

Only the existing BF16/full-MHA LLaVA source runtime is used. This is a new,
explicitly labeled diagnostic; it does not change the frozen exact candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import weakref
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
RUNS = Path("/scratch/djy8hg/runs/evicastkv")
PARENT = RUNS / "source-transport-corrected-consumed-20261101-v1"
MODEL = ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
ARMS = (
    "fullkv_native_reference",
    "prompt_topk_half_source_slots",
    "source3_mandatory_only",
    "source3_random_optional",
    "source3_attention_optional",
    "direct_projected_kv3_attention_optional_local",
    "source3_attention_optional_replay_on",
)
POLICY = dict(
    id="fox_source_controls_consumed_v1_not_frozen_source_or_exact_candidate",
    output_predictions=128,
    continuation_updates=127,
    prompt_window=16,
    source_bits=3,
    projected_key_bits=3,
    projected_value_bits=3,
    quant_group=128,
    source_axis=2,
    projected_key_axis=2,
    projected_value_axis=3,
    mandatory="all_original_text_plus_first4_last8_original_positions",
    optional_visual_fraction=0.05,
    random_seed=13,
    source_transport_corrections="existing_fit_transport_unchanged",
    source_transport_rtol=0.01,
    source_transport_atol=2**-6,
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    source_generation_history="existing_lossless_direct_source_history",
    replay="existing_SharedSourceCache_immutable_reconstruction_only",
    source_global_exact_rows_removed=False,
    timing_repetitions=1,
    warmups=0,
)


def read(path):
    return json.loads(Path(path).read_text())


def consumed_inputs(parent):
    """Exactly the historical two inputs; reject annotations before returning."""
    jobs = parent["jobs"]
    if len(jobs) != 2:
        raise ValueError("only the same two consumed source inputs are authorized")
    result = []
    for job in jobs:
        if len(job["inputs"]) != 1:
            raise ValueError("one consumed input per worker required")
        payload = read(job["inputs"][0])
        if set(payload) != {"task", "sample_id", "prompt", "media"}:
            raise ValueError("source controls prohibit answers, references and draft tokens")
        result.append(dict(job, input_sha256=sha(Path(job["inputs"][0]))))
    if result[0]["inputs"][0] == result[1]["inputs"][0]:
        raise ValueError("distinct historical inputs required")
    return result


def protection_masks(scores, text_positions, *, fraction=0.05, seed=13):
    """Label-blind per-head masks; random and attention have equal cardinality."""
    import torch

    if (
        scores.ndim != 2
        or not all(scores.shape)
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0).any())
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
        or type(seed) is not int
    ):
        raise ValueError(
            "finite nonnegative head/position scores and explicit fraction/seed required"
        )
    h, n = scores.shape
    if any(type(p) is not int or not 0 <= p < n for p in text_positions):
        raise ValueError("original text positions out of range")
    mandatory_ids = sorted(
        set(text_positions) | set(range(min(4, n))) | set(range(max(0, n - 8), n))
    )
    mandatory = torch.zeros((h, n), dtype=torch.bool)
    mandatory[:, mandatory_ids] = True
    available = [p for p in range(n) if p not in set(mandatory_ids)]
    count = min(len(available), math.floor(n * fraction))
    attention, randomized = mandatory.clone(), mandatory.clone()
    host = scores.detach().float().cpu()
    for head in range(h):
        order = sorted(available, key=lambda p: (-float(host[head, p]), p))
        attention[head, order[:count]] = True
        shuffled = list(available)
        random.Random(seed + head).shuffle(shuffled)
        randomized[head, shuffled[:count]] = True
    return dict(mandatory=mandatory, random=randomized, attention=attention), dict(
        mandatory_positions=mandatory_ids,
        optional_count_per_head=count,
        random_attention_cardinality_equal=True,
        no_labels_or_reference_outputs_used=True,
    )


def direct_projected(pair, mask):
    """Existing local asymmetric 3-bit recipe. It is NOT official KIVI."""
    import torch

    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.precision_tier import construct

    h, n, d = pair[0].shape[1:]
    if mask.shape != (h, n) or mask.dtype != torch.bool:
        raise ValueError("complete per-head protection mask required")
    common = tuple(torch.nonzero(mask.all(0)).flatten().tolist())
    base = construct(pair, bits=3, protected=common, encoded_exact=True)
    extra = mask.clone()
    extra[:, common] = False
    flat = extra.flatten().nonzero().flatten().to(device=pair[0].device, dtype=torch.int32)
    bodies = (
        tuple(encode(t.reshape(1, 1, h * n, d).index_select(2, flat.long())) for t in pair)
        if flat.numel()
        else ()
    )
    return HeadLocalFormat(base, flat, bodies)


def _historical_inventory():
    """Hash finite known evidence; do not invent a cross-version parity gate."""
    inventory = []
    for path, kind in (
        (PARENT / "analysis/decision.json", "corrected_source_transport"),
        (
            RUNS / "source-native-metrics-consumed-20261101-v1/analysis/decision.json",
            "native_metrics",
        ),
        (
            RUNS / "prepared-joint-replay-consumed-20261105-v1/analysis/decision.json",
            "prepared_vs_legacy_replay_not_OFF_vs_ON",
        ),
        (
            RUNS / "native-history-consumed-20261031-v1/analysis/decision.json",
            "source_vs_raw_history_not_replay_OFF_vs_ON",
        ),
    ):
        if not path.is_file():
            inventory.append(dict(kind=kind, path=str(path), status="missing"))
            continue
        record = read(path)
        inventory.append(
            dict(
                kind=kind,
                path=str(path),
                sha256=sha(path),
                producer_sha=record.get("source_sha"),
                complete=record.get("complete"),
                failures=record.get("failures"),
                qualifies_new_controls=False,
                why="different configuration or different intervention; preserve original scope",
            )
        )
    return inventory


def baseline_inventory():
    """Documentation/source-bound fidelity inventory, not fresh GPU parity."""
    specs = (
        (
            "prefixkv_reimpl",
            "PrefixKV",
            "597f1ab032704951550f93bcc8a23f1454b80aa4",
            "prefixkv_parity_report.md",
        ),
        (
            "lookm_reimpl",
            "LOOK-M",
            "ecf0f51a9c416c2d85e47faf2638502f01a6d748",
            "lookm_official_execution_blocker.md",
        ),
        ("vl_cache_reimpl", None, None, "vl_cache_spec.md"),
    )
    rows = []
    for method, folder, revision, filename in specs:
        path = SOURCE / "mosaickv/docs/baselines" / filename
        rows.append(
            dict(
                method=method,
                evidence_document=str(path),
                evidence_sha256=sha(path),
                upstream_path=str(SOURCE / "third_party" / folder) if folder else "not_available",
                pinned_revision=revision or "no_official_code_assumed",
                official_fidelity_for_this_new_diagnostic=False,
                historical_scope=(
                    "legacy same-model diagnostic exists; not parity for this BF16 adaptation"
                    if folder == "PrefixKV"
                    else "upstream imports missing MixMerLlamaAttention_drop; local port only"
                    if folder == "LOOK-M"
                    else "paper-derived local port; no official executable comparator"
                ),
            )
        )
    return rows


def prepare(root):
    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch output root required")
    source_sha = clean_source(SOURCE)
    parent = read(PARENT / "plan.json")
    gate = read(PARENT / "analysis/decision.json")
    if not gate.get("complete") or gate.get("failures"):
        raise ValueError("historical common source transport must be complete")
    jobs = consumed_inputs(parent)
    bound = {
        str(PARENT / name): sha(PARENT / name) for name in ("plan.json", "analysis/decision.json")
    }
    for index, job in enumerate(jobs):
        original = PARENT / f"workers/{index}"
        result = read(original / "result.json")
        if result.get("status") != "completed" or result.get("failures"):
            raise ValueError("historical worker incomplete")
        bound[str(original / "result.json")] = sha(original / "result.json")
        for filename, expected in result["outputs"].items():
            path = original / filename
            if sha(path) != expected:
                raise ValueError("historical evidence changed")
            bound[str(path)] = expected
        path = Path(job["inputs"][0])
        bound[str(path)] = sha(path)
        for item in read(path)["media"]:
            if sha(Path(item["path"])) != item["sha256"]:
                raise ValueError("historical media changed")
            bound[item["path"]] = item["sha256"]
    files = (
        "scripts/evicastkv_review_source_controls.py",
        "src/mosaickv/evaluation/shared_source_format.py",
        "src/mosaickv/evaluation/graphed_storage.py",
        "src/mosaickv/evaluation/storage_runs.py",
        "src/mosaickv/evaluation/precision_tier.py",
        "src/mosaickv/evaluation/source_history.py",
        "src/mosaickv/evaluation/head_local_precision.py",
        "src/mosaickv/evaluation/request_precision.py",
        "src/mosaickv/evaluation/phase_capacity.py",
        "src/mosaickv/evaluation/precision_tier_kernel.py",
        "src/mosaickv/evaluation/shared_source_kernel.py",
        "docs/fox_source_controls_20261110.md",
    )
    bound.update(
        {str(SOURCE / "mosaickv" / file): sha(SOURCE / "mosaickv" / file) for file in files}
    )
    plan = dict(
        source_sha=source_sha,
        jobs=jobs,
        arms=ARMS,
        policy=POLICY,
        bound_files=bound,
        expected_rows=2 * len(ARMS),
        maximum_workers=2,
        maximum_a100s_total=6,
        cpus_per_worker=12,
        time_limit_seconds=7200,
        config=parent["config"],
        parent=str(PARENT),
        historical_inventory=_historical_inventory(),
        baseline_inventory=baseline_inventory(),
        independent_quality_N=0,
        paper_superiority_claim=False,
        shared_realized_byte_comparison=False,
        timing_is_single_trial_diagnostic_not_headline=True,
        exact_retention_study_remains_separate=True,
        fresh_evaluation_authorized=False,
        source_to_current_Mistral_FP16_bridge="unsupported_full_MHA_BF16_constructor",
        KIVI="unsupported_no_verified_pinned_official_implementation",
        KVQuant="unsupported_no_verified_pinned_official_implementation",
        ZipCache="unsupported_no_verified_pinned_official_implementation",
        common_byte_campaign="NOT_RUN_needs_label_blind_attainable_target_controller",
    )
    write_once(root / "plan.json", plan)
    return dict(workers=2, plan_sha256=sha(root / "plan.json"), expected_rows=plan["expected_rows"])


def validate(root):
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != POLICY or tuple(plan["arms"]) != ARMS:
        raise ValueError("source control policy drift")
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bound file changed: {path}")
    return plan


def _formats(adapter, prepared, prefill, captured, normalized, phases, arm):
    import torch

    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.request_precision import ranked_positions

    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    n, h, d = source[0][0].shape[2], source[0][0].shape[1], source[0][0].shape[3]
    text = tuple(
        position
        for span in prepared.modality_map
        if getattr(span.modality, "value", span.modality) == "text"
        for position in range(span.start, span.end)
    )
    identity = (
        torch.arange(n, device=adapter.device, dtype=torch.int32),
        torch.arange(n, device=adapter.device, dtype=torch.int16)[None].expand(h, n),
    )
    formats, proof = [], []
    for layer, pair in enumerate(source):
        scores = captured.windows[layer][0].float().mean(-2)
        masks, metadata = protection_masks(scores, text, seed=POLICY["random_seed"] + layer * h)
        chosen = (
            "mandatory"
            if arm == "source3_mandatory_only"
            else "random"
            if arm == "source3_random_optional"
            else "attention"
        )
        mask = masks[chosen]
        if arm == "prompt_topk_half_source_slots":
            mandatory = tuple(metadata["mandatory_positions"])
            count = max(len(mandatory), n // 2)
            positions = ranked_positions(scores, count, mandatory)
            index = torch.tensor(positions, device=pair[0].device)[None, :, :, None].expand(
                1, h, count, d
            )
            selected = tuple(t.gather(2, index) for t in pair)
            packet = ExactFormat(
                "kv", tuple(selected[0].shape), pair=tuple(encode(t) for t in selected)
            )
            detail = dict(selected_positions_sha256=digest(positions), count=count)
        elif arm == "direct_projected_kv3_attention_optional_local":
            packet = direct_projected(pair, mask)
            detail = dict(local_recipe_not_KIVI=True, quantized_cached_keys="post_RoPE")
        else:
            # Scoped, single-threaded diagnostic override of only the protection
            # mask. Quantization, checks, patches and model arithmetic are intact.
            with patch.object(
                shared_source_format, "protected_positions", lambda *a, _mask=mask, **k: _mask
            ):
                packet = shared_source_format.construct(
                    normalized.records[layer],
                    pair,
                    (tuple(range(n)),) * h,
                    text,
                    phases.phases,
                    identity,
                    exclude_global_exact=False,
                )
            packet, detail = shared_source_format.fit_transport(
                packet, adapter._language_layers()[layer].self_attn
            )
        proof.append(
            dict(
                layer=layer,
                protection=chosen,
                **metadata,
                mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                protected_rows_per_head=mask.sum(-1).tolist(),
                component_bytes=packet.component_bytes,
                details=detail,
            )
        )
        formats.append(packet)
    return tuple(formats), proof


def _request(adapter, payload, cfg, arm):
    import evicastkv_request_validation as parent_runtime
    import torch

    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.storage_runs import run

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    phases = {}

    def timed(name, action):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = action()
        torch.cuda.synchronize()
        phases[name] = time.monotonic() - start
        return result

    prepared = timed(
        "media_and_processor", lambda: adapter.prepare_inputs(parent_runtime.load_messages(payload))
    )
    with (
        PromptQueryCapture(adapter, summaries=True, normalized=False, prior=False) as attention,
        SourceCapture(adapter) as normalized,
        PhaseCapture(adapter) as rotary,
    ):
        prefill = timed(
            "prefill_with_diagnostic_captures",
            lambda: adapter.prefill(prepared, capture_queries=False, capture_attentions=False),
        )
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    raw_bytes = backing_bytes(tuple(t for pair in pairs for t in pair))
    if arm == "fullkv_native_reference":
        # Keep the full original K/V as the reference representation; no lossy
        # codec or selection. Extra capture references are retired identically.
        formats = tuple(
            ExactFormat(
                "kv",
                tuple(k.shape),
                pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in (k, v)),
            )
            for k, v in pairs
        )
        proof = []
        phases["construction"] = 0.0
    else:
        formats, proof = timed(
            "construction",
            lambda: _formats(adapter, prepared, prefill, attention, normalized, rotary, arm),
        )
    representation_bytes = backing_bytes(tuple(t for packet in formats for t in packet.tensors()))
    packet_hash = _packet_hash(formats)
    source_refs = [weakref.ref(t) for pair in pairs for t in pair]
    del pairs

    def retire():
        prefill.state.past_key_values = None
        prepared.model_inputs.clear()
        attention.records.clear()
        attention.windows.clear()
        attention.priors.clear()
        normalized.records.clear()
        rotary.phases = None
        gc.collect()

    timed("retire_original_sources", retire)
    original_source_retired = not any(ref() is not None for ref in source_refs)
    if arm != "fullkv_native_reference" and not original_source_retired:
        raise RuntimeError("inactive original K/V survives source retirement")
    replay = arm == "source3_attention_optional_replay_on"
    logits, measured = timed(
        "decode_including_capture_and_retirement",
        lambda: run(
            adapter,
            prefill,
            formats,
            tokens=128,
            audit=False,
            source_history=True,
            source_history_constructor="direct",
            shared_workspace=True,
            workspace_graph=replay,
        ),
    )
    if _packet_hash(formats) != packet_hash:
        raise RuntimeError("decode/replay mutated immutable source representation")
    measured.update(
        phase_wall_seconds=phases,
        end_to_end_diagnostic_seconds=time.monotonic() - begin,
        whole_request_allocator_peak_bytes=int(torch.cuda.max_memory_allocated()),
        whole_request_allocator_reserved_peak_bytes=int(torch.cuda.max_memory_reserved()),
        whole_request_KV_only_peak_certified=False,
        inactive_original_source_retired=original_source_retired,
        fullkv_original_owner_transferred_to_reference=arm == "fullkv_native_reference",
        raw_source_kv_bytes=raw_bytes,
        representation_storage_bytes=representation_bytes,
        representation_sha256=packet_hash,
        protection_records=proof,
        replay_requested=replay,
        replay_eligible=True,
        replay_executed=measured["storage"].get("graph_replays", 0) > 0,
        replay_capture_count=measured["storage"].get("graph_captures", 0),
        replay_capture_cost_in_decode_timer=True,
        replay_retirement_in_decode_timer=True,
        replay_scope="immutable_prompt_and_sealed_history_reconstruction_not_model_decode",
        all_live_GPU_format_owners_charged=True,
        CPU_residual_bytes=0,
        captures_retired_before_decode=True,
        labels_or_reference_outputs_used=False,
        no_shared_byte_comparison=True,
        no_headline_latency_evidence=True,
    )
    return logits, measured


def _packet_hash(formats):
    import torch

    state = hashlib.sha256()
    for packet in formats:
        state.update(str((packet.kind, packet.shape)).encode())
        for tensor in packet.tensors():
            state.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            state.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return state.hexdigest()


def worker(root, index):
    import evicastkv_request_validation as parent_runtime
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("exactly two consumed workers")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        index=index,
    )
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one A100 and deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(POLICY["random_seed"])
        from mosaickv.evaluation.graphed_storage import gpu_controls

        write_once(out / "replay_gpu_controls.json", gpu_controls())
        torch.manual_seed(POLICY["random_seed"])
        job = plan["jobs"][index]
        payload = read(job["inputs"][0])
        cfg = RunConfig.from_mapping(job.get("config", plan["config"]))
        if (
            (cfg.model.id, cfg.model.revision) != MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("only pinned full-MHA BF16 LLaVA source transport is supported")
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = dict(
            source_sha=plan["source_sha"],
            config_sha256=config_sha256(cfg),
            policy_sha256=digest(POLICY),
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            backend="huggingface",
            attention="eager",
            seed=POLICY["random_seed"],
            measurement_type="consumed_source_protection_replay_diagnostic",
            sample_id=payload["sample_id"],
            parent_input_sha256=job["input_sha256"],
            baseline_commits=plan["baseline_inventory"],
            plan_sha256=sha(root / "plan.json"),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            model_precision="bf16",
            processor_revision=MODEL[1],
            source_dirty=False,
            cuda_runtime=torch.version.cuda,
            vllm="not_used",
            sglang="not_used",
            baseline_origin="source_and_direct_quantization_diagnostic_not_official_code",
        )
        write_once(out / "manifest.json", manifest)
        prepared = adapter.prepare_inputs(parent_runtime.load_messages(payload))
        checks = {
            name: asdict(function(adapter, prepared, max_new_tokens=16))
            for name, function in (
                ("ordinary_generate", compare_with_generate),
                ("retention_one", compare_cache_reinjection),
            )
        }
        write_once(out / "parity_controls.json", checks)
        if any(
            v["token_agreement"] != 1.0 or v["maximum_logit_difference"] > 0.001
            for v in checks.values()
        ):
            raise RuntimeError("ordinary or retention-one parity failed")
        del prepared
        full_reference, records, saved = None, {}, {}
        with (out / "raw.jsonl").open("x") as stream, torch.inference_mode():
            # FullKV first is a correctness reference. Remaining order is fixed
            # before results; a single trial cannot support a latency claim.
            for arm in ARMS:
                row = dict(
                    sample_id=payload["sample_id"],
                    task=payload["task"],
                    arm=arm,
                    status="failed",
                    source_sha=plan["source_sha"],
                    manifest_path=str(out / "manifest.json"),
                )
                try:
                    logits, record = _request(adapter, payload, cfg, arm)
                    if full_reference is None:
                        if arm != "fullkv_native_reference":
                            raise RuntimeError(
                                "no successful FullKV reference; retain all failures"
                            )
                        if logits.argmax(-1).tolist()[:16] != list(
                            checks["ordinary_generate"]["candidate_token_ids"]
                        ):
                            raise RuntimeError(
                                "isolated FullKV reader changed ordinary greedy output"
                            )
                        full_reference = logits.clone()
                    row.update(
                        record,
                        status="completed",
                        greedy_ids=logits.argmax(-1).tolist(),
                        registered_guard_32=comparison(full_reference[:32], logits[:32]),
                        model_reference_comparison_128=comparison(full_reference, logits),
                    )
                    saved[arm] = logits
                    records[arm] = row
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    result["failures"].append(dict(arm=arm, error=row["error"]))
                stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
                stream.flush()
            torch.save(saved, out / "logits.pt")
        result.update(
            status="completed" if not result["failures"] else "failed", completed_arms=list(records)
        )
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def replay_check(off, on):
    if (off["arm"], on["arm"]) != (
        "source3_attention_optional",
        "source3_attention_optional_replay_on",
    ):
        raise ValueError("registered same-representation replay pairing required")
    same = off["representation_sha256"] == on["representation_sha256"]
    if not same or off["replay_executed"] or not on["replay_executed"]:
        raise ValueError("unchanged encoded source and actual OFF/ON engagement required")
    for row in (off, on):
        storage = row["storage"]
        if (
            storage["graph_private_live_bytes"] != 0
            or storage["active_workspace_bytes"] != 0
            or not storage["graph_private_accounting_verified"]
        ):
            raise ValueError("unretired or uncharged private graph/workspace owner")
    return dict(
        representation_identical=True, actual_replay_engaged=True, replay_owners_retired=True
    )


def analyze(root):
    import torch

    from mosaickv.evaluation.lookm_merge_ablation import comparison

    root = Path(root)
    plan = validate(root)
    rows, failures, bindings, replay = [], [], {}, []
    for index in range(2):
        out = root / f"workers/{index}"
        path = out / "result.json"
        if not path.exists():
            failures.append(dict(index=index, error="missing worker result"))
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
            root / "plan.json"
        ):
            raise ValueError("worker provenance mismatch")
        bindings[str(path)] = sha(path)
        for name, expected in result["outputs"].items():
            if sha(out / name) != expected:
                raise ValueError("worker artifact changed")
            bindings[str(out / name)] = expected
        failures.extend(dict(index=index, **f) for f in result["failures"])
        produced = (
            [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            if (out / "raw.jsonl").exists()
            else []
        )
        if len(produced) != len(ARMS) or {r["arm"] for r in produced} != set(ARMS):
            failures.append(dict(index=index, error="incomplete/duplicate arm grid"))
        rows.extend(produced)
        current = {r["arm"]: r for r in produced if r["status"] == "completed"}
        if len(current) == len(ARMS):
            try:
                check = replay_check(
                    current["source3_attention_optional"],
                    current["source3_attention_optional_replay_on"],
                )
                logits = torch.load(out / "logits.pt", map_location="cpu", weights_only=True)
                paired = comparison(
                    logits["source3_attention_optional"],
                    logits["source3_attention_optional_replay_on"],
                )
                if paired["token_agreement"] != 1 or paired["maximum_logit_difference"] > 0.001:
                    raise ValueError("same-representation replay numerical parity failed")
                replay.append(dict(index=index, **check, numerical=paired))
            except Exception as error:
                failures.append(dict(index=index, error=f"{type(error).__name__}: {error}"))
    complete = (
        len(rows) == plan["expected_rows"]
        and not failures
        and all(r["status"] == "completed" for r in rows)
    )
    write_once(root / "analysis/rows.json", dict(rows=rows))
    write_once(
        root / "analysis/GATE_source_controls.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            complete=complete,
            expected_rows=plan["expected_rows"],
            actual_rows=len(rows),
            failures=failures,
            replay_controls=replay,
            bindings=bindings,
            all_source_variants_pass_registered_guard=complete
            and all(
                r["registered_guard_32"]["registered_guard_pass"]
                for r in rows
                if r["arm"].startswith("source3")
            ),
            raw_predictions_preserved=True,
            independent_quality_N=0,
            common_byte_comparison=False,
            whole_request_KV_peak_certified=False,
            superiority_established=False,
            paper_or_fresh_campaign_authorized=False,
            remaining=dict(
                common_byte=plan["common_byte_campaign"],
                KIVI=plan["KIVI"],
                KVQuant=plan["KVQuant"],
                ZipCache=plan["ZipCache"],
                current_retention_bridge=plan["source_to_current_Mistral_FP16_bridge"],
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.root), sort_keys=True))
    elif args.mode == "worker":
        worker(args.root, args.index)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
