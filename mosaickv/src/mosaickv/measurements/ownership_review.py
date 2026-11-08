"""CPU-only provenance review; never change allocator lifetimes or predictions."""

from copy import deepcopy

from mosaickv.measurements.kv_timeline import KV_ROLES, ROLES, valid_whole_request_certificate

RULES = (
    ("/mosaickv/evaluation/consolidated_history.py", "consolidate", "kv_workspace"),
    ("/mosaickv/evaluation/exact_minimum_metadata.py", "compact", "kv_workspace"),
    ("/mosaickv/evaluation/streaming_source_constructor.py", "__call__", "kv_workspace"),
    ("/mosaickv/evaluation/joint_value_transaction.py", "begin_key", "kv_workspace"),
    ("/mosaickv/evaluation/joint_incoming_projection.py", "forward", "kv_workspace"),
    ("/mosaickv/evaluation/prompt_graph_reader.py", "reconstruct", "kv_workspace"),
    ("/mosaickv/evaluation/prompt_graph_reader.py", "acquire", "kv_workspace"),
    ("/mosaickv/evaluation/deferred_transport_checks.py", "flush", "kv_workspace"),
    ("/mosaickv/evaluation/source_192_reader.py", "__init__", "kv_workspace"),
    ("/mosaickv/evaluation/input_norm_segment.py", "__init__", "kv_shared_workspace"),
    ("/mosaickv/evaluation/post_attention_segment.py", "__init__", "non_kv"),
    ("/mosaickv/adapters/huggingface/base.py", "_append_decode_attention_mask", "kv_workspace"),
    ("/transformers/models/clip/modeling_clip.py", "forward", "non_kv"),
    ("/torch/nn/functional.py", "gelu", "non_kv"),
    ("/torch/nn/functional.py", "layer_norm", "non_kv"),
    ("/torch/nn/functional.py", "embedding", "non_kv"),
    ("/transformers/masking_utils.py", "eager_mask", "kv_shared_workspace"),
    ("/transformers/masking_utils.py", "sdpa_mask_recent_torch", "kv_shared_workspace"),
    ("/transformers/masking_utils.py", "causal_mask_function", "kv_shared_workspace"),
    ("/mosaickv/evaluation/source_history.py", "seal_source_full_tails", "kv_workspace"),
    ("/mosaickv/evaluation/shared_head_scope.py", "__call__", "kv_workspace"),
    ("/mosaickv/evaluation/eager_core_operator.py", "__init__", "kv_shared_workspace"),
    ("/transformers/masking_utils.py", "_preprocess_mask_arguments", "kv_shared_workspace"),
    ("/transformers/models/llava/modeling_llava.py", "get_placeholder_mask", "non_kv"),
    ("/transformers/activations.py", "forward", "non_kv", 69),
)
RULES += tuple(
    ("/transformers/models/llama/modeling_llama.py", "forward", "kv_shared_workspace", line)
    for line in range(93, 106)
)
RULES += tuple(
    ("/transformers/models/llama/modeling_llama.py", "forward", "non_kv", line)
    for line in (152, 232, 259, 260, 299, 305)
)
RULES += tuple(
    ("/transformers/models/llama/modeling_llama.py", "forward", "kv_workspace", line)
    for line in (233, 234)
)
RULES += tuple(
    ("/mosaickv/evaluation/joint_value_transaction.py", "forward", "non_kv", line)
    for line in (135, 152, 153)
)
RULES += tuple(
    ("/transformers/models/llava/modeling_llava.py", "forward", "non_kv", line)
    for line in (108, 110)
)
RULES += tuple(
    ("/transformers/models/llava/modeling_llava.py", "forward", "non_kv", line)
    for line in (292, 296)
)
RULES += tuple(
    ("/mosaickv/adapters/huggingface/base.py", "prefill", "kv_workspace", line)
    for line in (827, 902)
)
RULES += (
    ("/mosaickv/evaluation/storage_runs.py", "run", "kv_workspace", 279),
    ("/mosaickv/evaluation/storage_runs.py", "run", "non_kv", 288),
)
RULES += (("/mosaickv/evaluation/storage_runs.py", "run", "non_kv", 385),)

# Charge standard language attention's scores, probabilities, and context too,
# not merely custom reconstruction buffers. Vision attention has no KV cache.
# This conservative boundary extension applies identically to both arms.
SHARED_ATTENTION_RULE = (
    "/transformers/models/llama/modeling_llama.py",
    "eager_attention_forward",
    "kv_shared_workspace",
)


def review(report, rules=RULES):
    if any(rule[2] not in ROLES - {"unknown"} for rule in rules):
        raise ValueError("invalid reviewed ownership role")
    result = deepcopy(report)
    stack_table = result["stack_table"]
    resolved = 0
    conservative_reclassifications = 0
    for row in result["records"]:
        if row["role"] == "non_kv" and any(
            filename.endswith(SHARED_ATTENTION_RULE[0]) and name == SHARED_ATTENTION_RULE[1]
            for filename, name, _ in stack_table[row["stack_id"]]
        ):
            row["role"] = SHARED_ATTENTION_RULE[2]
            row["evidence"].append("CPU_review:charge_standard_language_attention_workspace")
            conservative_reclassifications += 1
        if row["role"] != "unknown" or row["baseline"]:
            continue
        for filename, name, line in stack_table[row["stack_id"]]:
            matches = [
                r
                for r in rules
                if filename.endswith(r[0]) and name == r[1] and (len(r) == 3 or line == r[3])
            ]
            if len({r[2] for r in matches}) > 1:
                raise ValueError("conflicting reviewed ownership rules")
            if matches:
                row["role"] = matches[0][2]
                row["evidence"].append(f"CPU_reviewed_source:{filename}:{name}:{line}")
                resolved += 1
                break
    unknown = sum(row["role"] == "unknown" for row in result["records"])
    original = result["blocks_exact_certification"]
    blockers = [
        b
        for b in original
        if b
        not in ("unclassified_allocation_generations", "opaque_allocator_workspace_not_classified")
    ]
    if unknown:
        blockers += [
            "unclassified_allocation_generations",
            "opaque_allocator_workspace_not_classified",
        ]
    elif "ambiguous_deferred_owner_timestamp" in blockers:
        # Rejected timestamp observations assigned NO role. Every generation
        # now has an independent, exact source-stack attribution. Do not use
        # those rejected marks or alter any allocation start/end/index.
        blockers.remove("ambiguous_deferred_owner_timestamp")
        result["rejected_timestamp_marks_not_used_for_final_attribution"] = True
        result["rejected_timestamp_mark_count"] = None
        result["rejected_timestamp_mark_count_status"] = "not_recorded_by_producer"
    delta, upper_delta = {}, {}
    for row in result["records"]:
        for mapping, accepted in (
            (delta, row["role"] in KV_ROLES),
            (upper_delta, row["role"] in KV_ROLES | {"unknown"}),
        ):
            if accepted:
                mapping[row["start"]] = mapping.get(row["start"], 0) + row["bytes"]
                if row["end"] is not None:
                    mapping[row["end"]] = mapping.get(row["end"], 0) - row["bytes"]

    def maximum(mapping):
        peak = live = 0
        for key in sorted(mapping):
            live += mapping[key]
            peak = max(peak, live)
        return peak

    peak = maximum(delta)
    result.update(
        original_blocks_exact_certification=original,
        reviewed_allocations=resolved,
        conservative_attention_reclassifications=conservative_reclassifications,
        unknown_allocation_generations=unknown,
        blocks_exact_certification=blockers,
        opaque_workspace_classified=not unknown,
        known_KV_peak_lower_bound_bytes=peak,
        KV_peak_upper_bound_including_unknown_bytes=maximum(upper_delta),
        exact_KV_peak_bytes=peak if not blockers else None,
        exact_scoped_KV_peak_certified=not blockers,
        exact_whole_request_KV_peak_certified=not blockers and result["scope"] == "whole_request",
    )
    if result["exact_whole_request_KV_peak_certified"] and not valid_whole_request_certificate(
        result
    ):
        raise ValueError("reviewed allocator interval certificate failed independent recomputation")
    return result
