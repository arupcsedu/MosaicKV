"""Synthetic fixtures only; no fabricated observations in measured tables."""

import pandas as pd
import pytest

from mosaickv.evaluation.equal_quality_memory import exploratory_memory_screen


def fixture():
    return pd.DataFrame(
        [
            dict(
                task="ci",
                sample_id=sample_id,
                method=method,
                cap_mib=cap,
                nll_sum=nll,
                tokens=10,
                rouge_l_f1=rouge,
                end_to_end_time=2.0,
                **{
                    f"{b}_active_kv_storage_bytes": cap * 2**20
                    for b in ("initial", "final", "peak")
                },
            )
            for sample_id in ["a", "b", "c", "d"]
            for method, cap, nll, rouge in [
                ("candidate", 32, 10.0, 0.8),
                ("baseline", 40, 12.0, 0.7),
            ]
        ]
    )


def test_screen_preserves_category_and_never_promotes_development():
    result = exploratory_memory_screen(
        fixture(), candidate="candidate", baselines=("baseline",), replicates=1000, seed=7
    )[0]
    assert result["exploratory_memory_quality_pass"]
    assert result["final_memory_reduction"] == pytest.approx(0.2)
    assert result["paper_claim_eligible"] is False
    assert result["nearest_competitor_established"] is False
    assert "NOT_byte_matched" in result["classification"]


def test_quality_loss_is_not_hidden_by_smaller_memory():
    frame = fixture()
    frame.loc[frame.method == "candidate", "nll_sum"] = 15.0
    result = exploratory_memory_screen(
        frame, candidate="candidate", baselines=("baseline",), replicates=1000, seed=7
    )[0]
    assert result["strictly_smaller_memory"]
    assert not result["quality_screen_pass"]
    assert not result["exploratory_memory_quality_pass"]


def test_mismatched_identities_fail_closed():
    frame = fixture()
    frame.loc[0, "sample_id"] = "wrong"
    with pytest.raises(ValueError, match="identical paired"):
        exploratory_memory_screen(
            frame, candidate="candidate", baselines=("baseline",), replicates=1000, seed=7
        )
