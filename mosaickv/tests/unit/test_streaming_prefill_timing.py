from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.request_precision import REQUEST_POLICY, online_frozen
from mosaickv.evaluation.streaming_prefill_timing import order, summary


def test_minimal_window_and_full_prior_preserve_original_selection():
    torch.manual_seed(821)
    probabilities = tuple(torch.softmax(torch.randn(1, 4, 64, 64), -1).bfloat16() for _ in range(3))
    profile = (0.8, 0.2, 0.5)
    original, _ = online_frozen(
        SimpleNamespace(attention_weights=probabilities),
        profile,
        0.5,
        method="mosaickv_exact",
        allocation_mode="balanced",
    )
    window = REQUEST_POLICY["prompt_window"]
    compact = tuple(p[..., -window:, :] for p in probabilities)
    prior = {i: p.mean(1, keepdim=True).float().mean(-2)[0] for i, p in enumerate(probabilities)}
    actual, _ = online_frozen(
        SimpleNamespace(attention_weights=compact),
        profile,
        0.5,
        method="mosaickv_exact",
        allocation_mode="balanced",
        priors=prior,
    )
    assert original == actual


def fixtures():
    return [
        dict(
            **r,
            total_wall_seconds=2.0 if r["variant"] == "offline" else 1.5,
            logits_bitwise=True,
            tensor_result_auditing=False,
        )
        for r in order(37)
    ]


def test_paired_randomized_warmups_and_endtoend_statistics():
    rows = fixtures()
    assert rows == fixtures() and len(rows) == 50
    assert sum(r["warmup"] for r in rows) == 10
    assert len({r["variant"] for r in rows if r["order_index"] == 0}) == 2
    result = summary(rows, seed=37)
    assert result["timed_pairs"] == 20 and result["warmup_pairs"] == 5
    assert result["mean_time_ratio"] == 0.75 and result["paired_ratio_CI95"] == [0.75, 0.75]
    assert result["similar_time_gate_upper_ratio_1_05"]
    assert result["independent_quality_N"] == 0 and not result["baseline_comparison"]


@pytest.mark.parametrize("bad", ["duplicate", "missing", "logit", "audit", "nan"])
def test_invalid_trials_fail_closed(bad):
    rows = fixtures()
    if bad == "duplicate":
        rows[-1] = rows[0]
    elif bad == "missing":
        rows.pop()
    elif bad == "logit":
        rows[-1]["logits_bitwise"] = False
    elif bad == "audit":
        rows[-1]["tensor_result_auditing"] = True
    else:
        rows[-1]["total_wall_seconds"] = float("nan")
    with pytest.raises(ValueError):
        summary(rows, seed=37)
