import pytest
import torch

from mosaickv.evaluation.source_rank_feasibility import (
    choose_codec_coupled,
    cpu_serialized_storage,
    screen,
)


def test_optimistic_rank_bound_rejects_full_rank_noise_without_fabricating_factor_results():
    g = torch.Generator().manual_seed(919)
    x = torch.randn(1, 32, 128, generator=g).bfloat16()
    result = screen(x, torch.zeros(32, dtype=torch.bool), (), ranks=(2, 4))
    for row in result["rows"]:
        assert row["optimal_unquantized_rank_mse"] > row["baseline_source_mse"]
        assert row["actual_factor_source_mse"] is None
        assert not row["factor_numerical_screen_passed"]
    assert result["candidates"] == {}


def test_exact_low_rank_source_has_zero_optimistic_error_and_charged_payload():
    # Exactly repeated BF16 source rows: no learned parameters or labels.
    x = torch.zeros(1, 32, 128).bfloat16()
    x[0, ::2, :64] = 1
    x[0, 1::2, 64:] = 1
    result = screen(x, torch.zeros(32, dtype=torch.bool), (), ranks=(2,))
    row = result["rows"][0]
    assert row["optimal_unquantized_rank_mse"] < 1e-12
    assert row["optimistic_screen_passed"]
    assert row["factor_cpu_payload_bytes"] > 0
    with pytest.raises(ValueError):
        screen(x, torch.zeros(32, dtype=torch.bool), (), ranks=(33,))


def test_cpu_codec_accounts_for_offsets_scratch_and_raw_fallback():
    codes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    raw = cpu_serialized_storage([dict(codes=codes, other_bytes=12)])
    assert raw["net_cpu_payload_bytes"] == 268
    assert not raw["gpu_kv_storage_measured"]
    shared = cpu_serialized_storage([dict(codes=codes.clone(), other_bytes=12) for _ in range(8)])
    assert shared["charged_replay_bytes"] == 256
    assert shared["net_cpu_payload_bytes"] < 8 * 268


def test_codec_coupled_selection_matches_exhaustive_cost_not_raw_greed():
    import itertools

    common = torch.arange(1024).byte()
    g = torch.Generator().manual_seed(44)
    choices = []
    for _ in range(5):
        choices.append(
            [
                ("parent", dict(codes=common.clone(), other_bytes=4)),
                (
                    "shorter",
                    dict(
                        codes=torch.randint(0, 256, (896,), generator=g, dtype=torch.uint8),
                        other_bytes=4,
                    ),
                ),
            ]
        )
    actual, names = choose_codec_coupled(choices)
    exhaustive = min(
        (
            cpu_serialized_storage([layer[i][1] for layer, i in zip(choices, path, strict=True)])[
                "net_cpu_payload_bytes"
            ],
            path,
        )
        for path in itertools.product(range(2), repeat=5)
    )
    assert cpu_serialized_storage(actual)["net_cpu_payload_bytes"] == exhaustive[0]
    assert names == [choices[k][i][0] for k, i in enumerate(exhaustive[1])]
    greedy = cpu_serialized_storage([layer[1][1] for layer in choices])
    assert cpu_serialized_storage(actual)["net_cpu_payload_bytes"] < greedy["net_cpu_payload_bytes"]


def test_codec_coupled_ties_prefer_parent_and_empty_scope_rejected():
    body = dict(codes=torch.zeros(128, dtype=torch.uint8), other_bytes=0)
    _, names = choose_codec_coupled([[("parent", body), ("alternative", body)] for _ in range(3)])
    assert names == ["parent"] * 3
    with pytest.raises(ValueError):
        choose_codec_coupled([])
