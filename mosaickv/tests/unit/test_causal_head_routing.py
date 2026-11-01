import pytest
import torch

from mosaickv.evaluation.causal_head_routing import choose, proposals
from mosaickv.evaluation.last_layer_tail import statistics


def test_frozen_bounded_schedule_includes_each_layer_and_ties_deterministically():
    evidence = [dict(layer=i, head=h, score=float(i + h)) for i in range(32) for h in range(32)]
    a = proposals(evidence, layers=32)
    assert len(a) == 32 and all(r == dict(layer=i, head=31) for i, r in enumerate(a))
    assert proposals(list(reversed(evidence)), layers=32) == a
    evidence[0]["score"] = float("nan")
    with pytest.raises(ValueError):
        proposals(evidence, layers=32)
    with pytest.raises(ValueError):
        proposals([dict(layer=0, head=0, score=1.0)], layers=2)


def test_actual_guard_not_proxy_authorizes_and_fail_closed_never_fullkv():
    ref = torch.tensor([[0.0, 2.0, 1.0]])
    good, bad = statistics(ref, ref), statistics(ref, ref.flip(-1))
    rows = [dict(layer=0, head=3, statistics=bad), dict(layer=1, head=2, statistics=good)]
    assert choose(rows, bad)["layer"] == 1
    assert choose(rows[:1], bad) is None
    with pytest.raises(ValueError):
        choose([rows[0], rows[0]], bad)
