from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation.consumed_guard_check import (
    compare,
    consumed_inputs,
    finite_fault,
    parent_export,
    same_storage,
)
from mosaickv.evaluation.source_hybrid_validation import BOUNDARIES


def fixture():
    p = torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3)
    g = dict(
        steps=127,
        observed_parent_logit_steps=127,
        fullkv_fallback=False,
        benchmark_labels_accessed=False,
        observed=[[0.0, 0.0, 1.0, 1.0]] * 127,
        prototype_or_exact_changes_vetoed=0,
        tail_graph=dict(records_validated=True, record_validation_failed=False, recorded_steps=127),
    )
    return p, {"live_anchor_guard": g}


def test_strip_all_annotation_fields_without_reordering():
    rows = [
        dict(sample_id=str(i), prompt="image", media=[{"path": str(i)}], references=["secret"])
        for i in range(16)
    ]
    stripped = consumed_inputs(rows)
    assert [r["sample_id"] for r in stripped] == [str(i) for i in range(16)]
    assert all(set(r) == {"sample_id", "prompt", "media"} for r in stripped)
    assert "references" in rows[0]
    with pytest.raises(ValueError):
        consumed_inputs(rows[:15])
    with pytest.raises(ValueError):
        consumed_inputs([rows[0]] * 16)


def test_exact_records_and_parent_counterfactual():
    p, r = fixture()
    assert compare(p, p, p, p[1:], r, r)["all_127_guard_records_bitwise"]


@pytest.mark.parametrize("field,value", [("steps", 126), ("fullkv_fallback", True)])
def test_incomplete_or_fallback_fails(field, value):
    p, r = fixture()
    r["live_anchor_guard"][field] = value
    with pytest.raises(ValueError):
        compare(p, p, p, p[1:], r, r)


def test_no_tolerance_for_numeric_changes():
    p, r = fixture()
    altered = p.clone()
    altered[45, 1] += 0.001
    with pytest.raises(ValueError, match="changed model"):
        compare(p, p, altered, p[1:], r, r)
    with pytest.raises(ValueError, match="counterfactual"):
        compare(p, p, p, altered[1:], r, r)
    bad = p.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare(p, p, bad, p[1:], r, r)


def test_storage_uses_all_recorded_boundaries():
    d = {k: 100 for k in BOUNDARIES}
    assert same_storage(d, d) == d
    for key in BOUNDARIES:
        with pytest.raises(ValueError):
            same_storage(d, dict(d, **{key: 101}))


def test_finite_fault_injects_once_and_restores_method():
    from mosaickv.evaluation.guard_tail_graph import GuardTail

    def original(owner, parent, active):
        owner.calls += 1
        return active.clone()

    owner = SimpleNamespace(calls=0)
    with patch.object(GuardTail, "run", original):
        with finite_fault(True) as injections:
            outputs = [GuardTail.run(owner, None, torch.tensor(1.0)) for _ in range(7)]
        assert GuardTail.run is original
    assert injections == [3]
    assert [float(v) for v in outputs] == [1, 1, 1, 129, 1, 1, 1]


def test_parent_export_is_independent_and_patch_restores_on_error():
    from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard

    owner = SimpleNamespace(parent_logits=[torch.tensor([1.0]), torch.tensor([2.0])])

    def original(*args):
        return None

    with patch.object(LiveAnchorGuard, "__exit__", original):
        with pytest.raises(RuntimeError), parent_export() as captured:
            LiveAnchorGuard.__exit__(owner, None, None, None)
            raise RuntimeError("test abort")
        assert LiveAnchorGuard.__exit__ is original
    owner.parent_logits[0].zero_()
    assert captured["logits"].tolist() == [[1.0], [2.0]]
