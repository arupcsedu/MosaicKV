#!/usr/bin/env python3
"""Consumed-input, fixed-FullKV-prefix source diagnosis and honest byte ceilings.

This is evaluation-only. The frozen exact-selection study and source production
reader are not changed. In particular, different realized byte counts are NEVER
reported as byte matched merely because a common ceiling was configured.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-v6-parallel-controls-20261110/source"
)
MODEL = ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
ARMS = (
    "fullkv_native_reference",
    "fullkv_forced_clone_control",
    "prompt_topk_raw_bf16_ceiling_local",
    "source3_mandatory_only",
    "source3_random_optional",
    "source3_attention_optional",
    "direct_projected_kv3_attention_optional_local",
)
POLICY = dict(
    id="fox_source_fixed_fullkv_prefix_consumed_v1",
    output_predictions=128,
    continuation_updates=127,
    reference="one_FullKV_greedy_trajectory_per_input_no_benchmark_answers",
    conditioning="identical_FullKV_token_at_each_candidate_decode_step",
    selector_can_read_forced_prefix=False,
    prompt_window=16,
    source_bits=3,
    projected_key_bits=3,
    projected_value_bits=3,
    optional_protection_fractions=[0.0, 0.03, 0.05],
    choose="largest_measured_representation_in_registered_ladder_below_ceiling",
    ceiling="historical_same_input_source_attention_initial_bytes_geometry_only",
    byte_boundary="unique_initial_GPU_cache_backing_storage_before_decode",
    topk_family="uniform_all_layer_all_head_raw_BF16_capacity_no_padding",
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    clone_logit_absolute_tolerance=0.001,
    later_prefix_predictions=[64, 128],
    seed=13,
    replay="reuse_prior_passed_OFF_ON_evidence_do_not_rerun",
    history="existing_lossless_direct_source_append_only_127_updates",
    direct_quantization_when_infeasible="run_smallest_registered_packet_as_uncapped_diagnostic_not_comparison",
    repetitions=1,
    warmups=0,
    classification="consumed_development_diagnostic_not_paper_evidence",
    no_fresh_cohort=True,
    no_calibration=True,
    no_superiority_claim=True,
    no_latency_claim=True,
)

# This policy concerns CPU recomputation of already-saved diagnostic reductions,
# never model outputs, the registered GPU guard, or any experiment tolerance.
ANALYSIS_POLICY = dict(
    schema_version=1,
    id="fixed_prefix_CPU_KL_reduction_verification_v1",
    required_ATEN_CPU_CAPABILITY="avx2",
    required_torch_cpu_capability="AVX2",
    intraop_threads=12,
    fp32_epsilon=2**-23,
    fp64_epsilon=2**-52,
    fp32_epsilon_multiplier=8,
    fp64_epsilon_multiplier=64,
    kl_tolerance_formula="multiplier * epsilon * max(1, abs(saved), abs(recomputed))",
    tolerant_fields_FP32=["per_prefix_KL_FP32", "maximum_KL_FP32"],
    tolerant_fields_FP64=["per_prefix_KL_FP64", "mean_KL_FP64"],
    all_other_fields="identical_types_structure_and_values",
    registered_guard_booleans="exact_match_required_no_threshold_change",
    raw_record_values="preserved_never_replaced_by_CPU_recomputation",
    new_GPU_inference=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def label_blind_payload(path):
    value = read(path)
    if set(value) != {"task", "sample_id", "prompt", "media"}:
        raise ValueError("only consumed prompt/media/task/ID fields allowed; no labels")
    return value


def forced_prefix(reference):
    """Reference logits are used ONLY after construction, by evaluation transport."""
    import torch

    if reference.ndim != 2 or reference.shape[0] != 128 or not torch.isfinite(reference).all():
        raise ValueError("finite complete 128-prediction FullKV reference required")
    return tuple(int(t) for t in reference.argmax(-1).tolist())


def comparisons(reference, candidate):
    from mosaickv.evaluation.lookm_merge_ablation import comparison

    if reference.shape != candidate.shape or reference.shape[0] != 128:
        raise ValueError("identical 128-prediction reference and candidate required")
    return dict(
        registered_guard_32=comparison(reference[:32], candidate[:32]),
        cache_dependent_guard_32=comparison(reference[1:33], candidate[1:33]),
        prefix_64=comparison(reference[:64], candidate[:64]),
        prefix_128=comparison(reference, candidate),
        first_prediction_is_common_prefill=True,
        registered_guard_unchanged=True,
        same_conditioning_prefix=True,
        actual_cache_dependent_prediction_count=127,
        not_free_greedy_sequence_agreement=True,
    )


def verify_recomputed_diagnostics(recomputed, recorded):
    """Verify a saved record without confusing CPU KL roundoff with logit drift.

    Only the four named KL reduction fields receive finite, precision-derived
    comparisons. Structure, token identities/agreement, logit error, first
    divergence and all guard booleans remain exact. A guard flip always fails,
    including when KL drift is smaller than this verification tolerance.
    """
    stats = dict(
        maximum_KL_FP32_recomputation_drift=0.0,
        maximum_KL_FP64_recomputation_drift=0.0,
        verified_KL_FP32_values=0,
        verified_KL_FP64_values=0,
        changed_KL_FP32_values=0,
        changed_KL_FP64_values=0,
        exact_guard_booleans_checked=0,
    )

    def visit(actual, saved, path, precision=None):
        if isinstance(actual, dict):
            if not isinstance(saved, dict) or set(actual) != set(saved):
                raise ValueError(f"fixed-prefix diagnostic structure changed: {path}")
            for key in sorted(actual):
                kind = (
                    "FP32"
                    if key in ANALYSIS_POLICY["tolerant_fields_FP32"]
                    else "FP64"
                    if key in ANALYSIS_POLICY["tolerant_fields_FP64"]
                    else None
                )
                visit(actual[key], saved[key], f"{path}.{key}", kind)
            return
        if isinstance(actual, list):
            if not isinstance(saved, list) or len(actual) != len(saved):
                raise ValueError(f"fixed-prefix diagnostic list changed: {path}")
            for index, (left, right) in enumerate(zip(actual, saved, strict=True)):
                visit(left, right, f"{path}[{index}]", precision)
            return
        if precision is not None:
            if (
                type(actual) not in (int, float)
                or type(saved) not in (int, float)
                or not math.isfinite(actual)
                or not math.isfinite(saved)
            ):
                raise ValueError(f"nonfinite or invalid KL diagnostic: {path}")
            drift = abs(actual - saved)
            prefix = precision.lower()
            tolerance = (
                ANALYSIS_POLICY[f"{prefix}_epsilon_multiplier"]
                * ANALYSIS_POLICY[f"{prefix}_epsilon"]
                * max(1.0, abs(actual), abs(saved))
            )
            if drift > tolerance:
                raise ValueError(f"KL recomputation drift exceeds precision bound: {path}")
            maximum = f"maximum_KL_{precision}_recomputation_drift"
            stats[maximum] = max(stats[maximum], drift)
            stats[f"verified_KL_{precision}_values"] += 1
            stats[f"changed_KL_{precision}_values"] += int(drift != 0)
            return
        if type(actual) is not type(saved) or actual != saved:
            raise ValueError(f"non-KL fixed-prefix diagnostic changed: {path}")
        if isinstance(actual, float) and not math.isfinite(actual):
            raise ValueError(f"nonfinite fixed-prefix diagnostic: {path}")
        if path.endswith(".registered_guard_pass"):
            if type(actual) is not bool:
                raise ValueError("registered guard must be an exact boolean")
            stats["exact_guard_booleans_checked"] += 1

    visit(recomputed, recorded, "diagnostics")
    return stats


def analysis_cpu_controls():
    import torch

    if os.environ.get("ATEN_CPU_CAPABILITY") != ANALYSIS_POLICY["required_ATEN_CPU_CAPABILITY"]:
        raise ValueError("set ATEN_CPU_CAPABILITY=avx2 before Python starts for CPU analysis")
    actual = torch.backends.cpu.get_cpu_capability()
    if actual != ANALYSIS_POLICY["required_torch_cpu_capability"]:
        raise ValueError("actual Torch CPU dispatcher must report AVX2")
    torch.set_num_threads(ANALYSIS_POLICY["intraop_threads"])
    if torch.get_num_threads() != ANALYSIS_POLICY["intraop_threads"]:
        raise ValueError("CPU diagnostic recomputation requires exactly twelve intra-op threads")
    return dict(
        ATEN_CPU_CAPABILITY=os.environ["ATEN_CPU_CAPABILITY"],
        torch_cpu_capability=actual,
        intraop_threads=torch.get_num_threads(),
        torch_version=torch.__version__,
        observed_cpu_reduction_policy_only=True,
    )


def storage_components(formats):
    """Complete backing-storage inventory; aliases are charged exactly once.

    The codec's tensor declaration is authoritative. Dataclass traversal only
    labels those owners; no view numel sum, imaginary allocation or serialized
    Python-header estimate is substituted for actual GPU tensor storage.
    """
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    names = {}
    seen = set()

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            if value.numel():
                owner = (str(value.device), value.untyped_storage().data_ptr())
                names.setdefault(owner, []).append(path)
            return
        if id(value) in seen:
            return
        if dataclasses.is_dataclass(value):
            seen.add(id(value))
            for field in dataclasses.fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, (tuple, list)):
            seen.add(id(value))
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            seen.add(id(value))
            for key, item in sorted(value.items()):
                walk(item, f"{path}.{key}")

    tensors = tuple(t for packet in formats for t in packet.tensors())
    for i, packet in enumerate(formats):
        walk(packet, f"layer[{i}]")
    owners, total, categories = {}, 0, {}
    for tensor in tensors:
        if not tensor.numel():
            continue
        owner = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if owner in owners:
            owners[owner]["declared_alias_count"] += 1
            continue
        paths = names.get(owner, ["unclassified_declared_owner"])
        joined = " ".join(paths)
        category = (
            "numerical_corrections"
            if "patch" in joined or "correction" in joined
            else "quantization_codes"
            if ".codes" in joined
            else "quantization_scales_and_minima"
            if ".scale" in joined or ".minimum" in joined or ".codebook" in joined
            else "logical_positions_masks_and_indices"
            if any(
                s in joined
                for s in (
                    ".row_map",
                    ".row_positions",
                    ".slots",
                    ".union",
                    ".flat_tokens",
                    ".exact_map",
                    ".exact_indices",
                )
            )
            else "rotary_phases"
            if ".cos" in joined or ".sin" in joined
            else "exact_protection_or_original_payload"
            if any(s in joined for s in (".exact", ".protected", ".pair"))
            else "other_declared_tensor_header"
        )
        count = int(tensor.untyped_storage().nbytes())
        owners[owner] = dict(
            category=category,
            device=str(tensor.device),
            bytes=count,
            dtype=str(tensor.dtype),
            observed_view_shape=list(tensor.shape),
            paths=paths,
            declared_alias_count=1,
        )
        total += count
        categories[category] = categories.get(category, 0) + count
    if total != backing_bytes(tensors):
        raise ValueError("component inventory disagrees with unique backing enumeration")
    return dict(
        representation_storage_bytes=total,
        components=list(owners.values()),
        category_bytes=categories,
        includes_codes_scales_indices_rotary_exact_overlays_and_numerical_corrections=True,
        Python_metadata="host_objects_not_fictitious_GPU_header_bytes",
        generated_history="not_yet_allocated_reported_separately_after_decode",
        workspace="not_persistent_representation_observed_separately_in_cache_trace",
        CPU_residual_bytes=0,
        whole_request_peak_certified=False,
    )


def choose_attainable(candidates, cap):
    """Largest physically measured plan in a finite registered family, no labels."""
    if type(cap) is not int or cap < 1 or not candidates:
        raise ValueError("positive byte ceiling and nonempty measured candidate family required")
    if len({item["candidate_id"] for item in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs")
    for item in candidates:
        if type(item["actual_bytes"]) is not int or item["actual_bytes"] <= 0:
            raise ValueError("positive measured actual bytes required")
    eligible = [item for item in candidates if item["actual_bytes"] <= cap]
    chosen = (
        min(eligible, key=lambda item: (-item["actual_bytes"], item["candidate_id"]))
        if eligible
        else None
    )
    actual = None if chosen is None else chosen["actual_bytes"]
    return dict(
        cap_bytes=cap,
        feasible=chosen is not None,
        chosen_candidate_id=None if chosen is None else chosen["candidate_id"],
        actual_bytes=actual,
        shortfall_bytes=None if actual is None else cap - actual,
        utilization=None if actual is None else actual / cap,
        exact_ceiling_fill=actual == cap,
        largest_feasible_in_registered_family=bool(eligible),
        globally_largest_cache_certified=False,
        equal_realized_byte_comparison=False,
        candidates=candidates,
        selection_uses_labels_or_reference_outputs=False,
    )


def raw_topk_capacity(cap, *, layers, heads, length, width, mandatory_count):
    """Exhaustive uniform-capacity family solved analytically, including no padding."""
    if any(type(v) is not int or v < 1 for v in (cap, layers, heads, length, width)) or (
        type(mandatory_count) is not int or not 1 <= mandatory_count <= length
    ):
        raise ValueError("positive complete BF16 MHA geometry and mandatory count required")
    slot_bytes = layers * heads * width * 2 * 2
    count = min(length, cap // slot_bytes)
    return dict(
        feasible=count >= mandatory_count,
        count=count if count >= mandatory_count else None,
        minimum_mandatory_bytes=mandatory_count * slot_bytes,
        actual_bytes=count * slot_bytes if count >= mandatory_count else None,
        cap_bytes=cap,
        physical_slot_bytes=slot_bytes,
        shortfall_bytes=cap - count * slot_bytes if count >= mandatory_count else None,
        physical_padding_bytes=0,
        largest_uniform_dense_capacity_certified=True,
        original_logical_positions_preserved=True,
        equal_realized_byte_comparison=False,
    )


class InfeasibleCeiling(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            "registered representation family cannot fit the initial physical-byte ceiling"
        )


def _topk_formats(adapter, prepared, prefill, captured, cap):
    import torch

    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.request_precision import ranked_positions

    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    h, n, d = source[0][0].shape[1:]
    text = tuple(
        p
        for span in prepared.modality_map
        if getattr(span.modality, "value", span.modality) == "text"
        for p in range(span.start, span.end)
    )
    mandatory = tuple(sorted(set(text) | set(range(min(4, n))) | set(range(max(0, n - 8), n))))
    capacity = raw_topk_capacity(
        cap, layers=len(source), heads=h, length=n, width=d, mandatory_count=len(mandatory)
    )
    if not capacity["feasible"]:
        raise InfeasibleCeiling(capacity)
    count, formats, proof = capacity["count"], [], []
    for layer, pair in enumerate(source):
        if any(tuple(t.shape) != (1, h, n, d) or t.dtype != torch.bfloat16 for t in pair):
            raise ValueError("identical full-MHA BF16 geometry required")
        scores = captured.windows[layer][0].float().mean(-2)
        positions = ranked_positions(scores, count, mandatory)
        if any(not set(mandatory).issubset(row) or len(row) != count for row in positions):
            raise ValueError("Top-K removed mandatory evidence or changed capacity")
        index = torch.tensor(positions, device=pair[0].device)[None, :, :, None].expand(
            1, h, count, d
        )
        selected = tuple(t.gather(2, index) for t in pair)
        packet = ExactFormat(
            "kv",
            tuple(selected[0].shape),
            pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in selected),
        )
        formats.append(packet)
        proof.append(
            dict(
                layer=layer,
                selected_positions_sha256=digest(positions),
                count=count,
                mandatory_preserved=True,
                exact_original_KV=True,
            )
        )
    inventory = storage_components(tuple(formats))
    if inventory["representation_storage_bytes"] != capacity["actual_bytes"]:
        raise RuntimeError("raw dense capacity formula disagrees with actual backing storage")
    return (
        tuple(formats),
        proof,
        dict(family=POLICY["topk_family"], allocation=capacity, inventory=inventory),
    )


def _formats_with_ceiling(
    adapter, prepared, prefill, captured, normalized, phases, arm, cap, evidence
):
    import evicastkv_review_source_controls as previous

    if arm == "prompt_topk_raw_bf16_ceiling_local":
        formats, proof, detail = _topk_formats(adapter, prepared, prefill, captured, cap)
        evidence.update(detail)
        return formats, proof
    old_masks = previous.protection_masks
    candidates, selected, proofs, inventories = [], {}, {}, {}
    fractions = (
        [0.0] if arm == "source3_mandatory_only" else POLICY["optional_protection_fractions"]
    )
    for fraction in fractions:

        def masks(scores, text_positions, *, seed=13, fraction=fraction):
            return old_masks(scores, text_positions, fraction=fraction, seed=seed)

        # This replaces only protection-mask cardinality; the existing codec,
        # validation, RoPE, exact overlay and transport corrections are intact.
        with patch.object(previous, "protection_masks", masks):
            packets, proof = previous._formats(
                adapter, prepared, prefill, captured, normalized, phases, arm
            )
        key = f"optional-{fraction:.2f}"
        inventory = storage_components(packets)
        candidates.append(
            dict(
                candidate_id=key,
                actual_bytes=inventory["representation_storage_bytes"],
                optional_fraction=fraction,
            )
        )
        selected[key], proofs[key], inventories[key] = packets, proof, inventory
    allocation = choose_attainable(candidates, cap)
    evidence.update(
        family="finite_registered_protection_ladder_not_globally_optimal",
        allocation=allocation,
        candidate_inventories=inventories,
    )
    if not allocation["feasible"]:
        if arm != "direct_projected_kv3_attention_optional_local":
            raise InfeasibleCeiling(evidence)
        # Retain a numerical diagnostic for the direct quantizer, but never
        # misrepresent it as a feasible same-cap comparison. No FullKV fallback.
        key = min(candidates, key=lambda item: (item["actual_bytes"], item["candidate_id"]))[
            "candidate_id"
        ]
        evidence.update(
            uncapped_numerical_diagnostic=True,
            physical_ceiling_unsupported=True,
            selected_smallest_registered_candidate=key,
        )
    else:
        key = allocation["chosen_candidate_id"]
        evidence.update(uncapped_numerical_diagnostic=False, physical_ceiling_unsupported=False)
    evidence["inventory"] = inventories[key]
    return selected[key], proofs[key]


def native_reference(adapter, payload):
    """Actual unmodified HF cache and native append-only decode, no source codec.

    FullKV is a quality control. Its raw generated-history representation is
    intentionally not mixed with the source-history runtime's memory results.
    """
    import evicastkv_request_validation as parent_runtime
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    started = time.monotonic()
    prepared = adapter.prepare_inputs(parent_runtime.load_messages(payload))
    prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    state = prefill.state
    logical, next_position = state.logical_sequence_length, state.next_decode_position

    def actual_bytes():
        pairs, _ = adapter._legacy_layers(state.past_key_values)
        return backing_bytes(tuple(t for pair in pairs for t in pair))

    initial = peak = actual_bytes()
    values = [prefill.logits.detach().reshape(-1).float().cpu()]
    token = prefill.next_token_id
    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    decode_started = time.monotonic()
    start_event.record()
    for _ in range(127):
        output = adapter.decode_one_token(token, state, capture_queries=False)
        state, token = output.state, output.next_token_id
        values.append(output.logits.detach().reshape(-1).float().cpu())
        peak = max(peak, actual_bytes())
    end_event.record()
    torch.cuda.synchronize()
    logits = torch.stack(values)
    if not torch.isfinite(logits).all() or state.next_decode_position != next_position + 127:
        raise RuntimeError("native FullKV reference changed logical positions or became nonfinite")
    return logits, dict(
        storage=dict(
            initial_active_kv_storage_bytes=initial,
            final_active_kv_storage_bytes=actual_bytes(),
            peak_active_kv_storage_bytes=peak,
            boundary="native_HF_KV_at_completed_decode_steps_not_whole_request_peak",
            generated_history_lossless_and_append_only=True,
            raw_history_not_source_history=True,
            whole_request_peak_certified=False,
        ),
        decode_wall_seconds=time.monotonic() - decode_started,
        decode_cuda_ms=start_event.elapsed_time(end_event),
        end_to_end_diagnostic_seconds=time.monotonic() - started,
        fullkv_unmodified_HF_cache=True,
        source_codec_or_history_transport=False,
        output_tokens=128,
        continuation_updates=127,
        original_logical_length=logical,
        final_next_position=state.next_decode_position,
        representation_storage_bytes=initial,
        fixed_FullKV_prefix=False,
        free_greedy_sequence=True,
        conditioning_prefix_sha256=None,
        exact_realized_byte_match_claim=False,
        no_headline_latency_evidence=True,
        whole_request_KV_only_peak_certified=False,
        labels_or_reference_outputs_used=False,
    )


def request(adapter, payload, config, arm, *, prefix, cap):
    """Forced-prefix injection occurs at decode only, never in mask/selection APIs."""
    import evicastkv_review_source_controls as previous

    from mosaickv.evaluation import storage_runs

    if (
        arm not in ARMS
        or (arm == ARMS[0] and prefix is not None)
        or (arm != ARMS[0] and prefix is None)
    ):
        raise ValueError("FullKV reference first, then a complete fixed reference prefix required")
    if prefix is not None and (
        len(prefix) != 128 or any(type(t) is not int or t < 0 for t in prefix)
    ):
        raise ValueError("complete nonnegative forced FullKV prefix required")
    if arm == ARMS[0]:
        return native_reference(adapter, payload)
    old_run, old_formats = storage_runs.run, previous._formats
    evidence = {}

    def build(*args):
        # Temporarily restore the original builder while the finite controller
        # calls it, avoiding recursion. Neither prefix nor logits enter it.
        with patch.object(previous, "_formats", old_formats):
            return _formats_with_ceiling(*args, cap, evidence)

    def decode(*args, **kwargs):
        if "forced" in kwargs:
            raise ValueError("nested forcing policy is forbidden")
        kwargs["forced"] = prefix
        if arm not in (
            "fullkv_native_reference",
            "fullkv_forced_clone_control",
        ) and not evidence.get("uncapped_numerical_diagnostic"):
            kwargs["initial_physical_byte_cap"] = cap
        logits, record = old_run(*args, **kwargs)
        return logits, record

    parent_arm = "fullkv_native_reference" if arm == "fullkv_forced_clone_control" else arm
    with patch.object(previous, "_formats", build), patch.object(storage_runs, "run", decode):
        logits, record = previous._request(adapter, payload, config, parent_arm)
    record.update(
        conditioning_prefix_sha256=None if prefix is None else digest(prefix),
        selection_prefix_access=False,
        fixed_FullKV_prefix=prefix is not None,
        free_greedy_sequence=prefix is None,
        physical_ceiling=cap if arm not in ARMS[:2] else None,
        physical_allocation=evidence,
        realized_initial_bytes=record["storage"]["initial_active_kv_storage_bytes"],
        realized_final_bytes=record["storage"]["final_active_kv_storage_bytes"],
        realized_decode_owned_peak_bytes=record["storage"]["peak_active_kv_storage_bytes"],
        exact_realized_byte_match_claim=False,
        no_common_model_bridge_to_Mistral_FP16_GQA=True,
        local_direct_quantization_not_KIVI=True,
    )
    return logits, record


def _bind_parent(parent):
    """Require all prior fourteen observations/replay controls, not their quality."""
    parent = Path(parent)
    plan = read(parent / "plan.json")
    gate = read(parent / "analysis/GATE_source_controls.json")
    if (
        not gate.get("complete")
        or gate.get("failures")
        or gate.get("actual_rows") != 14
        or len(gate.get("replay_controls", [])) != 2
    ):
        raise ValueError("complete fourteen-row parent and two passed replay controls required")
    if gate["plan_sha256"] != sha(parent / "plan.json"):
        raise ValueError("parent gate is not bound to the source plan")
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"parent source binding changed: {path}")
    bound = {
        str(parent / name): sha(parent / name)
        for name in ("plan.json", "analysis/GATE_source_controls.json")
    }
    jobs = []
    for index, original in enumerate(plan["jobs"]):
        payload = label_blind_payload(original["inputs"][0])
        if sha(Path(original["inputs"][0])) != original["input_sha256"]:
            raise ValueError("consumed original input changed")
        for media in payload["media"]:
            if sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("consumed original media changed")
        result_path = parent / f"workers/{index}/result.json"
        result = read(result_path)
        if (
            result["status"] != "completed"
            or result["failures"]
            or result["plan_sha256"] != sha(parent / "plan.json")
        ):
            raise ValueError("parent source worker incomplete")
        bound[str(result_path)] = sha(result_path)
        for name, expected in result["outputs"].items():
            path = result_path.parent / name
            if sha(path) != expected:
                raise ValueError("parent source artifact changed")
            bound[str(path)] = expected
        rows = [
            json.loads(line) for line in (result_path.parent / "raw.jsonl").read_text().splitlines()
        ]
        by_arm = {row["arm"]: row for row in rows}
        cap = by_arm["source3_attention_optional"]["storage"]["initial_active_kv_storage_bytes"]
        if type(cap) is not int or cap < 1 or payload["sample_id"] != rows[0]["sample_id"]:
            raise ValueError("same consumed input and measured initial geometry required")
        jobs.append(dict(original, initial_ceiling_bytes=cap, sample_id=payload["sample_id"]))
        for path in (Path(original["inputs"][0]), *(Path(m["path"]) for m in payload["media"])):
            bound[str(path)] = sha(path)
    if len(jobs) != 2 or len({job["sample_id"] for job in jobs}) != 2:
        raise ValueError("only two distinct consumed inputs permitted")
    return plan, jobs, bound


def prepare(root, parent=DEFAULT_PARENT):
    import evicastkv_review_source_controls as previous

    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch root required")
    source_sha = clean_source(SOURCE)
    old, jobs, bound = _bind_parent(parent)
    for filename in (
        "scripts/evicastkv_review_source_prefix.py",
        "scripts/evicastkv_review_source_controls.py",
        "src/mosaickv/evaluation/storage_runs.py",
        "src/mosaickv/evaluation/shared_source_format.py",
        "src/mosaickv/evaluation/initial_byte_budget.py",
        "src/mosaickv/evaluation/lossless_bf16.py",
        "env/common/requirements.lock",
    ):
        path = SOURCE / "mosaickv" / filename
        bound[str(path)] = sha(path)
    baseline_inventory = previous.baseline_inventory()
    for row in baseline_inventory:
        bound[row["evidence_document"]] = row["evidence_sha256"]
    plan = dict(
        source_sha=source_sha,
        parent=str(parent),
        jobs=jobs,
        config=old["config"],
        policy=POLICY,
        arms=ARMS,
        bound_files=bound,
        baseline_inventory=baseline_inventory,
        reused_replay_gate=str(Path(parent) / "analysis/GATE_source_controls.json"),
        expected_rows=14,
        maximum_workers=2,
        maximum_a100s_total=6,
        cpus_per_worker=12,
        time_limit_seconds=7200,
        fresh_evaluation_authorized=False,
        common_realized_byte_comparison=False,
        paper_superiority_claim=False,
        KIVI_KVQuant_ZipCache="unsupported_no_audited_official_equivalent_source_path",
    )
    write_once(root / "plan.json", plan)
    return dict(workers=2, expected_rows=14, plan_sha256=sha(root / "plan.json"))


def validate(root, *, producer_source=None):
    plan = read(Path(root) / "plan.json")
    producer = SOURCE if producer_source is None else Path(producer_source).resolve(strict=True)
    if not producer.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("producer source must remain under the registered scratch area")
    clean_source(producer, plan["source_sha"])
    if plan["policy"] != POLICY or tuple(plan["arms"]) != ARMS:
        raise ValueError("source fixed-prefix policy drift")
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"source fixed-prefix binding changed: {path}")
    return plan


def worker(root, index):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("two consumed workers only")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        index=index,
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
    )
    saved = {}
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one A100 and deterministic prelaunch CUDA configuration required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(POLICY["seed"])
        job = plan["jobs"][index]
        payload = label_blind_payload(job["inputs"][0])
        cfg = RunConfig.from_mapping(job.get("config", plan["config"]))
        if (
            (cfg.model.id, cfg.model.revision) != MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("pinned BF16 full-MHA LLaVA source transport only")
        manifest = dict(
            source_sha=plan["source_sha"],
            source_dirty=False,
            config_sha256=config_sha256(cfg),
            policy_sha256=digest(POLICY),
            model=dataclasses.asdict(cfg.model),
            dataset=dataclasses.asdict(cfg.dataset),
            model_precision="bf16",
            processor_revision=MODEL[1],
            environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
            environment=environment(),
            gpu=dataclasses.asdict(capture_gpu_environment(torch)),
            backend="huggingface",
            attention="eager",
            seed=POLICY["seed"],
            measurement_type="consumed_fixed_prefix_source_diagnostic",
            result_category="unified_comparison",
            result_origin="locally_measured",
            shared_byte_claim=False,
            baseline_commits=plan["baseline_inventory"],
            sample_id=payload["sample_id"],
            parent_input_sha256=job["input_sha256"],
            plan_sha256=sha(root / "plan.json"),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            cuda_runtime=torch.version.cuda,
            vllm="not_used",
            sglang="not_used",
        )
        write_once(out / "manifest.json", manifest)
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        import evicastkv_request_validation as parent_runtime

        prepared = adapter.prepare_inputs(parent_runtime.load_messages(payload))
        controls = {
            name: dataclasses.asdict(function(adapter, prepared, max_new_tokens=16))
            for name, function in (
                ("ordinary_generate", compare_with_generate),
                ("retention_one", compare_cache_reinjection),
            )
        }
        write_once(out / "parity_controls.json", controls)
        if any(
            c["token_agreement"] != 1.0 or c["maximum_logit_difference"] > 0.001
            for c in controls.values()
        ):
            raise RuntimeError("ordinary/retention-one parity failed unchanged tolerance")
        del prepared
        reference = prefix = None
        with (out / "raw.jsonl").open("x") as stream, torch.inference_mode():
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
                    logits, record = request(
                        adapter, payload, cfg, arm, prefix=prefix, cap=job["initial_ceiling_bytes"]
                    )
                    if reference is None:
                        if arm != ARMS[0]:
                            raise RuntimeError("missing FullKV reference")
                        reference, prefix = logits.clone(), forced_prefix(logits)
                        if list(prefix[:16]) != list(
                            controls["ordinary_generate"]["candidate_token_ids"]
                        ):
                            raise RuntimeError(
                                "FullKV source reader differs from ordinary generation"
                            )
                        write_once(
                            out / "reference_prefix.json",
                            dict(
                                tokens=prefix,
                                sha256=digest(prefix),
                                provenance="this_input_FullKV_only_not_annotations",
                                predictions=128,
                            ),
                        )
                    row.update(
                        record,
                        status="completed",
                        diagnostics=comparisons(reference, logits),
                        reference_prefix_sha256=digest(prefix),
                    )
                    if arm == "fullkv_forced_clone_control" and (
                        row["diagnostics"]["prefix_128"]["token_agreement"] != 1.0
                        or row["diagnostics"]["prefix_128"]["maximum_logit_difference"]
                        > POLICY["clone_logit_absolute_tolerance"]
                    ):
                        raise RuntimeError("fixed-prefix FullKV clone control failed")
                    saved[arm] = logits
                except InfeasibleCeiling as error:
                    row.update(
                        status="unsupported_byte_ceiling",
                        physical_allocation=error.evidence,
                        no_decode_executed=True,
                        no_fullkv_fallback=True,
                        error=str(error),
                    )
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    result["failures"].append(dict(arm=arm, error=row["error"]))
                stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
                stream.flush()
            torch.save(saved, out / "logits.pt")
        result["status"] = "completed" if not result["failures"] else "failed"
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, *, producer_source=None):
    import torch

    cpu_controls = analysis_cpu_controls()
    analysis_source_sha = clean_source(SOURCE)
    root = Path(root)
    plan = validate(root, producer_source=producer_source)
    rows, failures, bindings, pairings, numerical_verifications = [], [], {}, [], []
    for index in range(2):
        out = root / f"workers/{index}"
        path = out / "result.json"
        if not path.exists():
            failures.append(dict(index=index, error="missing worker"))
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
            root / "plan.json"
        ):
            raise ValueError("worker provenance mismatch")
        bindings[str(path)] = sha(path)
        for name, expected in result["outputs"].items():
            p = out / name
            if sha(p) != expected:
                raise ValueError("worker artifact changed")
            bindings[str(p)] = expected
        produced = (
            [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            if (out / "raw.jsonl").exists()
            else []
        )
        if len(produced) != 7 or {r["arm"] for r in produced} != set(ARMS):
            failures.append(dict(index=index, error="incomplete or duplicate arm grid"))
        failures.extend(dict(index=index, **f) for f in result["failures"])
        tensors = (
            torch.load(out / "logits.pt", map_location="cpu", weights_only=True)
            if (out / "logits.pt").exists()
            else {}
        )
        reference = tensors.get(ARMS[0])
        for row in produced:
            if row["status"] == "completed":
                if reference is None or row["arm"] not in tensors:
                    raise ValueError("completed row lacks immutable logit evidence")
                expected = comparisons(reference, tensors[row["arm"]])
                verified = verify_recomputed_diagnostics(expected, row["diagnostics"])
                numerical_verifications.append(dict(index=index, arm=row["arm"], **verified))
                if row["arm"] != ARMS[0] and row["conditioning_prefix_sha256"] != digest(
                    forced_prefix(reference)
                ):
                    raise ValueError("candidate conditioning changed")
                if row["arm"] not in ARMS[:2]:
                    allocation = row["physical_allocation"]
                    inventory = allocation["inventory"]
                    if (
                        row["realized_initial_bytes"] > plan["jobs"][index]["initial_ceiling_bytes"]
                        and not allocation.get("uncapped_numerical_diagnostic")
                    ) or inventory["representation_storage_bytes"] != row[
                        "representation_storage_bytes"
                    ]:
                        raise ValueError("physical ceiling or complete accounting failed")
                    pairings.append(
                        dict(
                            index=index,
                            arm=row["arm"],
                            initial=row["realized_initial_bytes"],
                            final=row["realized_final_bytes"],
                            decode_owned_peak=row["realized_decode_owned_peak_bytes"],
                            cap=plan["jobs"][index]["initial_ceiling_bytes"],
                        )
                    )
            elif row["status"] != "unsupported_byte_ceiling":
                failures.append(
                    dict(index=index, arm=row["arm"], error=row.get("error", "failed row"))
                )
        rows.extend(produced)
    write_once(root / "analysis/rows.json", dict(rows=rows))
    completed = [row for row in rows if row["status"] == "completed"]
    source_rows = [r for r in completed if r["arm"].startswith("source3")]
    write_once(
        root / "analysis/GATE_source_fixed_prefix.json",
        dict(
            source_sha=plan["source_sha"],
            producer_source_sha=plan["source_sha"],
            analysis_source_sha=analysis_source_sha,
            analysis_implementation_path=str(Path(__file__).resolve()),
            analysis_implementation_sha256=sha(Path(__file__).resolve()),
            analysis_policy=ANALYSIS_POLICY,
            analysis_policy_sha256=digest(ANALYSIS_POLICY),
            analysis_cpu_controls=cpu_controls,
            recomputation_verifications=numerical_verifications,
            maximum_KL_FP32_recomputation_drift=max(
                (r["maximum_KL_FP32_recomputation_drift"] for r in numerical_verifications),
                default=0.0,
            ),
            maximum_KL_FP64_recomputation_drift=max(
                (r["maximum_KL_FP64_recomputation_drift"] for r in numerical_verifications),
                default=0.0,
            ),
            raw_record_values_preserved=True,
            registered_guard_thresholds_unchanged=POLICY["guard"],
            plan_sha256=sha(root / "plan.json"),
            complete=len(rows) == 14 and not failures,
            expected_rows=14,
            actual_rows=len(rows),
            completed_rows=len(completed),
            unsupported_rows=sum(r["status"] == "unsupported_byte_ceiling" for r in rows),
            uncapped_numerical_rows=sum(
                bool(r.get("physical_allocation", {}).get("uncapped_numerical_diagnostic"))
                for r in rows
            ),
            failures=failures,
            source_guard_rows_expected=6,
            source_guard_rows_completed=len(source_rows),
            all_source_variants_pass_registered_guard=len(source_rows) == 6
            and all(
                r["diagnostics"]["registered_guard_32"]["registered_guard_pass"]
                for r in source_rows
            ),
            byte_ceiling_records=pairings,
            equal_realized_bytes_certified=False,
            original_replay_evidence_reused=plan["reused_replay_gate"],
            identical_reference_conditioning_verified=len(rows) == 14 and not failures,
            labels_read=False,
            independent_quality_N=0,
            whole_request_KV_peak_certified=False,
            superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
            bindings=bindings,
            remaining=(
                "Local direct-KV quantization may be physically infeasible under this ceiling; "
                "no dummy fill, precision/model bridge or official quantization-baseline claim."
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--producer-source", type=Path)
    args = parser.parse_args()
    if args.producer_source is not None and args.mode != "analyze":
        parser.error("--producer-source is restricted to derived CPU analysis")
    if args.mode == "prepare":
        print(json.dumps(prepare(args.root, args.parent), sort_keys=True))
    elif args.mode == "worker":
        worker(args.root, args.index)
    else:
        analyze(args.root, producer_source=args.producer_source)


if __name__ == "__main__":
    main()
