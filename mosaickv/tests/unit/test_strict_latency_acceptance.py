import copy
import json
from pathlib import Path

import evicastkv_strict_latency as controller
import pytest
from evicastkv_strict_latency import summarize


def trials(ratio=1.0, all_methods=False):
    arms = [
        ("mosaickv_exact", "direct", ratio),
        ("mosaickv_exact", "cached", 1.2),
        ("prefixkv_reimpl", "direct", 1.0),
        ("prefixkv_reimpl", "cached", 1.0),
        ("prefixkv_reimpl", "whole-6291456", 1.0),
    ]
    if all_methods:
        arms += [
            (m, "direct", 1.0) for m in ("prompt_attention_topk", "vl_cache_reimpl", "lookm_reimpl")
        ]
    return [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(total_wall_seconds=v, logits_bitwise=True, no_FullKV_fallback=True),
        )
        for m, a, v in arms
        for t in range(-5, 20)
    ]


def test_equal_is_pass_and_five_percent_slower_is_only_tradeoff():
    equal = summarize(trials(), kind="direct", seed=3)
    slower = summarize(trials(1.04), kind="direct", seed=3)
    assert equal["strict_latency_gate"] is True
    assert slower["strict_latency_gate"] is False
    assert slower["within_five_percent_tradeoff"] is True
    assert slower["acceptance_ratio_upper_bound"] == 1.0


def test_fastest_measured_control_cannot_be_hidden():
    rows = trials(0.99)
    for r in rows:
        if r["arm"] == "whole-6291456":
            r["measurement"]["total_wall_seconds"] = 0.9
    assert summarize(rows, kind="direct", seed=3)["strict_latency_gate"] is False


def test_stronger_faster_gate_and_determinism():
    result = summarize(trials(0.94), kind="direct", seed=3)
    assert result["five_percent_faster_gate"] is True
    assert result == summarize(trials(0.94), kind="direct", seed=3)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "nan", "parity", "fallback", "warmup"]
)
def test_invalid_raw_observations_fail_closed(mutation):
    rows = copy.deepcopy(trials())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "nan":
        rows[0]["measurement"]["total_wall_seconds"] = float("nan")
    elif mutation == "parity":
        rows[0]["measurement"]["logits_bitwise"] = False
    elif mutation == "fallback":
        rows[0]["measurement"]["no_FullKV_fallback"] = False
    else:
        rows[0]["warmup"] = False
    with pytest.raises(ValueError):
        summarize(rows, kind="direct", seed=3)


def test_all_baseline_evidence_required_at_final_stage():
    with pytest.raises(ValueError, match="all five"):
        summarize(trials(), kind="direct", seed=3, final=True)
    assert summarize(trials(all_methods=True), kind="direct", seed=3, final=True)[
        "strict_latency_gate"
    ]


def test_cached_stage_can_be_reclassified_without_model_execution():
    rows = trials(1.04)
    for r in rows:
        r["arm"] = {"direct": "text-6291456", "cached": "reuse"}.get(r["arm"], r["arm"])
    assert not summarize(rows, kind="cached", seed=3)["strict_latency_gate"]


@pytest.mark.parametrize("ratio,launches", [(1.04, False), (0.99, True)])
def test_only_strict_gate_can_launch_registered_followup(tmp_path, monkeypatch, ratio, launches):
    plan = dict(source_sha="execution", seed=3)
    monkeypatch.setattr(controller, "validate", lambda *_: plan)
    monkeypatch.setattr(controller.direct.driver, "load_producers", lambda *_: ([{}, {}], []))
    called = []
    monkeypatch.setattr(controller, "launch_child", lambda *args: called.append(args))
    for index in (0, 1):
        folder = tmp_path / f"workers/{index}"
        folder.mkdir(parents=True)
        raw = [dict(r, source_sha="execution", plan_sha256="plan") for r in trials(ratio)]
        (folder / "trials.jsonl").write_text("\n".join(map(json.dumps, raw)))
    controller.analyze(tmp_path, "plan", Path("runtime"), "analyzer", "timing", "direct")
    report = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert bool(called) is launches
    assert report["all_input_latency_gate"] is launches
    assert report["execution_source_sha"] != report["analyzer_source_sha"]
    assert not report["paper_superiority_established"]


def test_failed_producer_cannot_advance(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "validate", lambda *_: dict(source_sha="execution"))
    monkeypatch.setattr(controller.direct.driver, "load_producers", lambda *_: ([], ["failed"]))
    monkeypatch.setattr(controller, "launch_child", lambda *_: pytest.fail("must not launch"))
    controller.analyze(tmp_path, "plan", Path("runtime"), "analyzer", "study", "direct")
    report = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not report["complete"] and not report["all_input_latency_gate"]
