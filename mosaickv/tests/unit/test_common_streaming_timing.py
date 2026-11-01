import pytest

from mosaickv.evaluation.common_streaming_timing import METHODS, order, summarize


def rows():
    return [
        dict(
            **r,
            total_wall_seconds=2.0 if r["method"] == "mosaickv_exact" else 3.0,
            logits_bitwise=True,
            tensor_result_auditing=False,
        )
        for r in order(73)
    ]


def test_randomized_bounded_same_engine_methods_and_paired_intervals():
    records = rows()
    assert len(records) == 125 and sum(r["warmup"] for r in records) == 25
    assert len({r["method"] for r in records if r["order_index"] == 0}) == 5
    assert records == rows()
    output = summarize(records, 73)
    assert set(output["distributions"]) == set(METHODS)
    assert len(output["pairs"]) == 4
    for pair in output["pairs"].values():
        assert pair["mean_ratio"] == 2 / 3
        assert pair["paired_ratio_CI95"] == [2 / 3, 2 / 3]
        assert pair["familywise_eight_comparison_ratio_bounds"] == [2 / 3, 2 / 3]
        assert pair["paired_difference_CI95"] == [-1.0, -1.0]
        assert pair["similar_time_gate_upper_ratio_1_05"]
    assert not output["paper_superiority_established"] and output["independent_quality_N"] == 0
    assert not output["equal_encoded_byte_comparison"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "logit", "audit", "nan", "negative"])
def test_incomplete_or_invalid_requests_fail_closed(bad):
    records = rows()
    if bad == "missing":
        records.pop()
    elif bad == "duplicate":
        records[-1] = records[0]
    elif bad == "logit":
        records[-1]["logits_bitwise"] = False
    elif bad == "audit":
        records[-1]["tensor_result_auditing"] = True
    else:
        records[-1]["total_wall_seconds"] = float("nan") if bad == "nan" else -1
    with pytest.raises(ValueError):
        summarize(records, 73)
