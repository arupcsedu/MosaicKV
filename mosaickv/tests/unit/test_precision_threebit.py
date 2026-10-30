import json

import evicastkv_precision_threebit as extension
import evicastkv_precision_tier as diagnostic
import pytest


@pytest.mark.parametrize("passes", [False, True])
def test_selected_precision_not_hardcoded_four(monkeypatch, tmp_path, passes):
    root = tmp_path / "comparison"
    root.mkdir()
    diagnostic.write_once(root / "plan.json", dict(jobs=[{}], comparison_precision_bits=3))

    def original(out, expected):
        diagnostic.write_once(
            out / "analysis/GATE.json",
            dict(
                failures=[],
                controls=[
                    dict(
                        controls={
                            "3": dict(numerical_gate=passes, smaller_initial_final_peak=True),
                            "4": dict(numerical_gate=not passes, smaller_initial_final_peak=True),
                        },
                        trials=[],
                    )
                ],
            ),
        )

    monkeypatch.setattr(diagnostic, "original_analyze", original)
    diagnostic.analyze(root, "sha")
    decision = json.loads((root / "analysis/precision_decision.json").read_text())
    assert decision["registered_precision_passed"] == passes
    assert decision["registered_precision_bits"] == 3
    assert not decision["four_bit_canary_passed"]
    assert not decision["paper_campaign_authorized"]


def test_extension_never_materializes_a_fresh_cohort(tmp_path):
    with pytest.raises(ValueError, match="consumed"):
        extension.prepare(tmp_path / "fresh", "baselines")
