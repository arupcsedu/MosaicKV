from dataclasses import asdict

import pytest

from mosaickv.evaluation.consumed_component_panel import frozen_policies, select_inputs
from mosaickv.evaluation.storage_six_overlay import COMPONENT_MODES, OverlayPolicy


def test_id_only_selection_is_order_invariant_and_discards_all_labels():
    samples = [
        dict(sample_id=f"id:{i}", prompt="caption", media=[], references=["secret"], score=i)
        for i in range(40)
    ]
    actual = select_inputs(samples, ("id:0", "id:1"))
    changed = [dict(s, references=["other"], score=-s["score"]) for s in reversed(samples)]
    assert select_inputs(changed, ("id:0", "id:1")) == actual
    assert len(actual) == 16
    assert all(set(s) == {"sample_id", "prompt", "media"} for s in actual)
    assert not {"id:0", "id:1"} & {s["sample_id"] for s in actual}
    with pytest.raises(ValueError):
        select_inputs(samples, ("unknown",))
    with pytest.raises(ValueError):
        select_inputs(samples, (), count=128)
    with pytest.raises(ValueError):
        select_inputs([*samples, samples[0]], ())


def test_policies_freeze_budget_runtime_and_thresholds_without_lossy_source_change():
    base = OverlayPolicy(
        inline_repair=True,
        graph_positional_search="indexed",
        graph_cosine_topk="partition",
        graph_accumulation="segmented_tensor_topk",
        forecast_distribution="individual_queries",
        prototype_representation="parent_residual_delta",
        precision_transaction="prototype",
        record_phase_timings=True,
    )
    policies = frozen_policies(asdict(base))
    assert tuple(policies) == COMPONENT_MODES
    for mode, cfg in policies.items():
        actual = OverlayPolicy(**dict(cfg, layers=tuple(cfg["layers"])))
        assert actual.maximum_extra_gpu_bytes == base.maximum_extra_gpu_bytes
        assert actual.entropy_threshold == base.entropy_threshold
        assert actual.prototype_members == base.prototype_members
        assert actual.extra_exact_positions == base.extra_exact_positions
        assert not actual.replace_source_codes
        assert actual.component_mode == mode
    with pytest.raises(ValueError):
        frozen_policies(asdict(OverlayPolicy()))
