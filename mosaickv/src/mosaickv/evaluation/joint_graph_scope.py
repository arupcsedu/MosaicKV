"""Accurate graph-scope metadata; old raw measurements are never rewritten."""

from copy import deepcopy


def graph_scope(policy):
    if type(policy) is not str or policy not in ("none", "keys", "kv"):
        raise ValueError("registered explicit graph projection policy required")
    active = policy != "none"
    return dict(
        only_immutable_prompt_captured=not active,
        incoming_frozen_weight_projections_captured=active,
        captured_incoming_components=["key", "value"] if active else [],
        reused_incoming_components={"none": [], "keys": ["key"], "kv": ["key", "value"]}[policy],
        attention_softmax_and_model_layers_captured=False,
    )


def reconcile(measured):
    """Derived metadata only, retaining the originally recorded stale flag."""
    result = deepcopy(measured)
    state = result.get("joint_incoming_projection", {})
    policy = state.get("policy", "none")
    scope = graph_scope(policy if state.get("eligible_layers", 0) else "none")
    graph = result["prompt_graph_reader"]
    original = graph["only_immutable_prompt_captured"]
    graph.update(scope)
    result["graph_scope_metadata_correction"] = dict(
        original_only_immutable_prompt_captured=original,
        corrected_only_immutable_prompt_captured=scope["only_immutable_prompt_captured"],
        correction_required=original != scope["only_immutable_prompt_captured"],
        no_timing_storage_or_numerical_measurement_changed=True,
        original_raw_artifact_must_remain_immutable=True,
        reason=(
            "joint tail graphs capture the current incoming hidden projection; "
            "they are not immutable-prompt-only"
        ),
    )
    return result
