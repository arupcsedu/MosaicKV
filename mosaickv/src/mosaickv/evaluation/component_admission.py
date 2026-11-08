"""Explicit reduced-component variant; never relabel it as the full method.

An additive overlay with a greedy-token lock cannot improve the parent's
answer and cannot reduce shared-source backing without an ownership transfer.
For a parent-preservation objective the useful transaction is the validated
source reader itself, without installing any inactive overlay hooks. Research
overlays remain available explicitly and are not removed from the codebase.
"""

from contextlib import contextmanager

from mosaickv.evaluation.storage_six_overlay import integrated_storage

SIX_COMPONENTS = (
    "future_query_forecasting",
    "cross_modal_evidence_graph",
    "value_aware_block_utility",
    "budgeted_submodular_selection",
    "exact_prototype_residual_overlay",
    "uncertainty_guided_residual_repair",
)


@contextmanager
def component_scope(adapter, policy, *, mode, reader_options=None):
    if mode not in ("storage_preservation", "experimental_overlay"):
        raise ValueError("explicit source-only or experimental-overlay variant required")
    if mode == "experimental_overlay":
        with integrated_storage(adapter, policy, reader_options=reader_options) as report:
            report["component_admission_mode"] = mode
            yield report
        return
    if policy.enabled or policy.quality_objective != "bitwise_storage_only":
        raise ValueError("source-only preservation requires explicitly disabled lossy overlays")
    if policy.replace_source_codes or policy.readout_selection != "none":
        raise ValueError("unverified source transformations cannot enter the hook-free variant")
    # Do not create caches, tensors, hooks, monkey patches or a second prefill.
    # The caller applies the same independently verified common codec to all
    # source-only controls. Parent reader validation/accounting remain intact.
    report = dict(
        variant="evicastkv_storage_only_hook_free_v1",
        component_admission_mode=mode,
        original_six_mechanism_design_active=False,
        disabled_components=list(SIX_COMPONENTS),
        active_six_component_overlays=[],
        additive_overlay_admitted=False,
        reason="no_certified_shared_source_retirement_and_no_parent_answer_gain",
        numerical_contract="bitwise_parent_logits_required_by_external_validation",
        reader_validation_and_physical_storage_accounting_unchanged=True,
        extra_gpu_bytes=0,
        residual_cpu_bytes=0,
        repair_events=[],
        no_extra_prefill_or_draft=True,
        whole_request_peak_certified=False,
        benchmark_labels_accessed=False,
        quality_improvement_claim_authorized=False,
    )
    yield report
