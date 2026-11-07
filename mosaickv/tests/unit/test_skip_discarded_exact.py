import pytest
import torch

from mosaickv.evaluation.precision_transaction_study import plan_precision_transaction
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def payload(seed):
    g = torch.Generator().manual_seed(seed)
    original = tuple(torch.randn(32, 16, generator=g).bfloat16() for _ in range(2))
    parent = tuple(
        (t.float() + 0.03 * torch.randn(t.shape, generator=g)).bfloat16() for t in original
    )
    return dict(
        original_pair=original,
        parent_pair=parent,
        prompt_queries=torch.randn(8, 16, generator=g).bfloat16(),
        phases=(torch.ones(32, 16).bfloat16(), torch.zeros(32, 16).bfloat16()),
        evidence=dict(
            selected_extra=[2, 4, 6, 8],
            prototype_members=[11, 12],
            original_protected_positions=[0, 31],
            prototype_anchor_position=10,
            prototype_representation="parent_residual_delta",
        ),
        overlay=dict(
            member_keys=torch.zeros(2, 16).bfloat16(), prototype_value=torch.zeros(1, 16).bfloat16()
        ),
    )


@pytest.mark.parametrize("seed", range(8))
def test_skipping_discarded_search_preserves_every_returned_payload_and_guard(seed, monkeypatch):
    import mosaickv.evaluation.precision_transaction_study as module

    p = payload(seed)
    old = plan_precision_transaction(p, mode="prototype")

    def must_not_run(*args, **kwargs):
        raise AssertionError("discarded exact search still ran")

    monkeypatch.setattr(module, "restoration_errors", must_not_run)
    new = plan_precision_transaction(p, mode="prototype", skip_discarded_exact=True)
    for k in (
        "baseline",
        "accepted_exact",
        "accepted_prototype",
        "accepted_meaningful_change",
        "final_metrics",
        "returned_exact",
        "joint_proposal_rejected",
        "final_joint_guard_passed",
        "fixed_prototype_members_preserved",
        "query_weighted_prototype",
    ):
        assert old[k] == new[k]
    assert set(old["replacement"]) == set(new["replacement"])
    for k, v in old["replacement"].items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, new["replacement"][k])
        elif k == "prototype_pair":
            assert all(torch.equal(a, b) for a, b in zip(v, new["replacement"][k], strict=True))
        else:
            assert v == new["replacement"][k]
    assert new["discarded_exact_search_evaluated"] is False
    assert old["discarded_exact_search_evaluated"] is True


def test_required_exact_search_cannot_be_skipped_or_hidden():
    for mode in ("exact", "joint_guarded"):
        with pytest.raises(ValueError, match="discarded"):
            plan_precision_transaction(payload(1), mode=mode, skip_discarded_exact=True)
    with pytest.raises(ValueError):
        OverlayPolicy(skip_discarded_exact=True)
    assert OverlayPolicy(
        skip_discarded_exact=True,
        precision_transaction="prototype",
        prototype_representation="parent_residual_delta",
    )


def test_live_constructor_engages_skip_without_changing_members_or_allocations(monkeypatch):
    from dataclasses import replace

    from test_storage_error_aware import arguments

    import mosaickv.evaluation.precision_transaction_study as module
    from mosaickv.evaluation.storage_six_overlay import _bytes, plan_layer

    args, parent = arguments()
    p = OverlayPolicy(
        forecast_distribution="individual_queries",
        prototype_representation="parent_residual_delta",
        precision_transaction="prototype",
        record_phase_timings=True,
    )
    old = plan_layer(*args, p, pin_residuals=False, parent_pair=parent)

    def must_not_run(*args, **kwargs):
        raise AssertionError("unused search still runs in live constructor")

    monkeypatch.setattr(module, "restoration_errors", must_not_run)
    new = plan_layer(
        *args, replace(p, skip_discarded_exact=True), pin_residuals=False, parent_pair=parent
    )
    assert _bytes(old.gpu_tensors()) == _bytes(new.gpu_tensors()) == 6713
    assert old.cpu_bytes == new.cpu_bytes == 1024
    for key in ("selected_extra", "prototype_members", "prototype_anchor_position"):
        assert old.evidence[key] == new.evidence[key]
    for attr in ("positions", "exact_ids", "member_keys"):
        assert torch.equal(getattr(old, attr), getattr(new, attr))
    for attr in ("exact_pair", "prototype_pair"):
        assert all(
            torch.equal(a, b) for a, b in zip(getattr(old, attr), getattr(new, attr), strict=True)
        )


def test_conditional_runtime_followup_cannot_broaden_the_graph_study(tmp_path):
    from storage_six_ablation import prepare

    with pytest.raises(ValueError, match="registered graph parity"):
        prepare(tmp_path / "no_graph", paired_after_graph_root=tmp_path / "runtime")
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError, match="registered graph parity"):
        prepare(
            tmp_path / "graph",
            graph_runtime_study=True,
            paired_after_graph_root=tmp_path / "existing",
        )
