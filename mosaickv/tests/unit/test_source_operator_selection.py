from copy import deepcopy

from mosaickv.evaluation.source_operator_microbench import select_group


def fixture():
    return [
        dict(
            microbenchmark=dict(
                medians_ms=dict(fused_h1=1.0, fused_h2=0.7, fused_h4=0.8, native_bmm_256=0.1),
                numerical={
                    name: [dict(bitwise_equal=True, history_sentinel_unchanged=True)] * 2
                    for name in ("fused_h1", "fused_h2", "fused_h4", "native_bmm_256")
                },
            )
        )
        for _ in range(2)
    ]


def test_only_bitwise_grouped_strategy_can_be_selected_not_fast_native():
    cases = fixture()
    assert select_group(cases)["chosen_grouped_operator"] == "fused_h2"
    cases[0]["microbenchmark"]["numerical"]["fused_h2"] = [
        dict(bitwise_equal=False, history_sentinel_unchanged=True)
    ]
    assert select_group(cases)["chosen_grouped_operator"] == "fused_h4"
    for c in cases:
        c["microbenchmark"]["medians_ms"]["fused_h4"] = 0.95
    assert select_group(cases)["chosen_grouped_operator"] == "fused_h1"


def test_missing_or_unsupported_operators_are_not_silent_passes():
    cases = deepcopy(fixture())
    del cases[0]["microbenchmark"]["medians_ms"]["fused_h2"]
    assert select_group(cases)["chosen_grouped_operator"] == "fused_h4"
    del cases[0]["microbenchmark"]["medians_ms"]["fused_h1"]
    assert select_group(cases)["chosen_grouped_operator"] is None
    assert select_group([])["chosen_grouped_operator"] is None
