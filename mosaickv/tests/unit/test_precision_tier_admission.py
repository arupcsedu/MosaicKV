import json
from pathlib import Path

import evicastkv_precision_tier as diagnostic
import pytest


@pytest.mark.parametrize("numerical,memory", [(False, True), (True, False)])
def test_failed_quality_or_memory_never_submits(monkeypatch, tmp_path, numerical, memory):
    root = tmp_path / "canary"
    root.mkdir()
    diagnostic.write_once(root / "plan.json", {"jobs": [{}, {}]})

    def original(out, expected):
        diagnostic.write_once(
            out / "analysis/GATE.json",
            dict(
                failures=[],
                controls=[
                    dict(
                        controls={
                            "4": dict(numerical_gate=numerical, smaller_initial_final_peak=memory)
                        },
                        trials=[],
                    )
                ]
                * 2,
            ),
        )

    monkeypatch.setattr(diagnostic, "original_analyze", original)
    monkeypatch.setattr(
        diagnostic.base.launch, "submit", lambda *a, **kw: pytest.fail("must not submit")
    )
    diagnostic.analyze(root, "sha")
    decision = json.loads((root / "analysis/precision_decision.json").read_text())
    assert not decision["four_bit_canary_passed"]
    assert not decision["paper_campaign_authorized"]


def test_only_new_consumed_roots_allowed(tmp_path):
    with pytest.raises(ValueError, match="consumed"):
        diagnostic.prepare(tmp_path / "fresh", "baselines")
    with pytest.raises(ValueError, match="consumed"):
        diagnostic.prepare(Path("/tmp"), "baselines")
