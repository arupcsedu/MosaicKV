import json

import evicastkv_source_hotpath_followup as driver
import pytest


def test_promotion_uses_upper_bounds_on_both_inputs_not_only_mean():
    def summary(upper):
        return dict(
            within_method=dict(
                mosaickv_exact=dict(mean_ratio=0.9, familywise_ratio_bounds=[0.8, upper])
            )
        )

    assert driver.improvement([summary(0.99), summary(0.98)], [])
    assert not driver.improvement([summary(0.99001), summary(0.98)], [])
    assert not driver.improvement([summary(0.98)], [])
    assert not driver.improvement([summary(0.98), summary(0.98)], ["unsafe"])


@pytest.mark.parametrize("passed", [False, True])
def test_second_attempt_is_bounded_and_uses_only_promoted_anchor(monkeypatch, tmp_path, passed):
    plan = dict(
        source_sha="s",
        hotpath_attempt=0,
        chain_root=str(tmp_path),
        bound_files={},
        candidate_hotpath="prepared",
        anchor_hotpath="original",
    )
    monkeypatch.setattr(driver.pair.common.previous.common, "inputs", lambda *a: plan)
    summaries = [
        dict(
            within_method=dict(
                mosaickv_exact=dict(familywise_ratio_bounds=[0.8, 0.98 if passed else 1.01])
            )
        )
        for _ in (0, 1)
    ]
    monkeypatch.setattr(driver.pair, "read_producers", lambda *a: (summaries, [], []))
    monkeypatch.setattr(driver.pair, "reuse_cpu", lambda *a: None)
    calls = []
    monkeypatch.setattr(driver, "precheck", lambda *a: calls.append(a))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    driver.analyze(tmp_path, "h")
    saved = json.loads((tmp_path / "restore/plan.json").read_text())
    assert saved["hotpath_attempt"] == 1
    assert saved["anchor_hotpath"] == ("prepared" if passed else "original")
    assert saved["candidate_hotpath"] == ("prepared_restore" if passed else "restore")
    assert len(calls) == 1 and calls[0][0] == tmp_path / "restore"


def test_unsafe_producer_stops_and_final_dict_distributions_are_supported(monkeypatch, tmp_path):
    plan = dict(
        source_sha="s",
        hotpath_attempt=0,
        chain_root=str(tmp_path),
        anchor_hotpath="original",
        candidate_hotpath="prepared",
    )
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(driver.pair.common.previous.common, "inputs", lambda *a: plan)
    monkeypatch.setattr(driver.pair, "read_producers", lambda *a: ([], [], ["bad logits"]))
    monkeypatch.setattr(driver, "precheck", lambda *a: pytest.fail("no next GPU run"))
    driver.analyze(tmp_path, "h")
    final = json.loads((tmp_path / "engineering_closeout.json").read_text())
    assert not final["complete"] and final["failures"] == ["bad logits"]
    assert not final["fresh_or_large_campaign_submitted"]
    root = tmp_path / "completed"
    root.mkdir()
    decision = dict(
        complete=True,
        failures=[],
        all_method_similar_time_gate=False,
        summaries=[
            dict(
                distributions={"mosaickv_exact": dict(mean=2, p50=2, p95=3)},
                pairs={"prefixkv_reimpl": dict(familywise_eight_comparison_ratio_bounds=[1, 1.2])},
            )
        ],
    )
    (root / "plan.json").write_text(json.dumps(dict(source_sha="s", chain_root=str(root))))
    (root / "analysis").mkdir()
    (root / "analysis/decision.json").write_text(json.dumps(decision))
    driver.closeout(root, decision)
    assert "| mosaickv_exact " in (root / "engineering_report.md").read_text()


def test_failed_primitive_precheck_gets_terminal_closeout(monkeypatch, tmp_path):
    plan = dict(source_sha="s", chain_root=str(tmp_path))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(driver.pair.common.previous.common, "inputs", lambda *a: plan)

    def failure(*args):
        driver.write_once(tmp_path / "precheck/blocked.json", dict(error="bad primitive"))

    monkeypatch.setattr(driver.pair, "begin", failure)
    driver.begin(tmp_path, "h")
    final = json.loads((tmp_path / "engineering_closeout.json").read_text())
    assert not final["complete"] and final["failures"] == ["bad primitive"]
