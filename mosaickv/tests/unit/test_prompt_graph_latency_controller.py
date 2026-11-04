import copy
import json

import evicastkv_prompt_graph_latency as controller
import pytest
from evicastkv_prompt_graph_latency import METHODS, make_records, summarize


def fixture(ratio=0.95, final=False):
    methods = METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
    arms = [(m, a) for m in methods for a in ("direct", "graph")]
    arms += [("prefixkv_reimpl", "whole-6291456"), ("prefixkv_reimpl", "whole-graph")]
    return [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=(ratio if (m, a) == ("mosaickv_exact", "graph") else 1.0),
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in arms
        for t in range(-5, 20)
    ]


@pytest.mark.parametrize("ratio,passes", [(0.95, True), (0.961, False), (1.0, False)])
def test_strict_four_percent_target_is_not_equal_or_faster(ratio, passes):
    d = summarize(fixture(ratio), 3)
    assert d["four_percent_latency_gate"] is passes
    assert d["equal_or_faster_gate"]
    assert d["acceptance_ratio_upper_bound"] == 0.96


def test_fastest_new_baseline_reader_cannot_be_hidden():
    rows = fixture()
    for x in rows:
        if x["method"] == "prefixkv_reimpl" and x["arm"] == "graph":
            x["measurement"]["total_wall_seconds"] = 0.9
    assert not summarize(rows, 3)["equal_or_faster_gate"]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "nan", "bool", "warmup", "parity", "fallback"]
)
def test_invalid_trials_fail_closed(mutation):
    rows = fixture()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "warmup":
        rows[0]["warmup"] = False
    else:
        key, value = {
            "nan": ("total_wall_seconds", float("nan")),
            "bool": ("total_wall_seconds", True),
            "parity": ("logits_bitwise", False),
            "fallback": ("no_FullKV_fallback", False),
        }[mutation]
        rows[0]["measurement"][key] = value
    with pytest.raises(ValueError):
        summarize(rows, 3)


def test_final_requires_all_five_methods_and_deterministic_statistics():
    with pytest.raises(ValueError):
        summarize(fixture(), 3, final=True)
    rows = fixture(final=True)
    assert summarize(rows, 3, final=True) == summarize(rows, 3, final=True)


def test_record_builder_gives_every_method_both_readers_without_changing_parent():
    parent = dict(
        records={
            m: dict(
                direct=dict(direct_policy={"fixed": True}),
                **{"whole-6291456": dict(whole_policy={"fixed": True})},
            )
            for m in METHODS
        }
    )
    old = copy.deepcopy(parent)
    records = make_records(parent, METHODS)
    assert parent == old
    for arms in records.values():
        assert not arms["direct"]["prompt_graph_reader"]
        assert arms["graph"]["prompt_graph_reader"]
    assert records["prefixkv_reimpl"]["whole-graph"]["prompt_graph_reader"]


@pytest.mark.parametrize("ratio,advance", [(0.95, True), (0.97, False)])
def test_only_four_percent_gate_releases_all_baseline_extension(
    tmp_path, monkeypatch, ratio, advance
):
    monkeypatch.setattr(controller.driver, "checked", lambda *_: dict(seed=3, source_sha="source"))
    monkeypatch.setattr(controller.driver, "load_producers", lambda *_: ([{}, {}], []))
    called = []
    monkeypatch.setattr(controller, "stage", lambda *a: called.append(a))
    monkeypatch.setattr(controller.direct.common.previous, "child", lambda *_: tmp_path / "child")
    for i in (0, 1):
        folder = tmp_path / f"workers/{i}"
        folder.mkdir(parents=True)
        rows = [dict(r, source_sha="source", plan_sha256="plan") for r in fixture(ratio)]
        (folder / "trials.jsonl").write_text("\n".join(map(json.dumps, rows)))
    controller.analyze(tmp_path, "plan", "timing_analysis")
    d = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert d["complete"] and d["four_percent_latency_gate"] is advance
    assert bool(called) is advance
    assert d["independent_quality_N"] == 0 and not d["paper_superiority_established"]


def test_failed_cpu_proof_gets_terminal_decision_and_no_gpu_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.driver, "checked", lambda *_: dict(source_sha="source"))

    def invalid(*_):
        raise ValueError("incomplete CPU proof")

    monkeypatch.setattr(controller.driver, "cpu_proof", invalid)
    monkeypatch.setattr(controller, "stage", lambda *_: pytest.fail("must not launch"))
    controller.analyze(tmp_path, "plan", "cpu_analysis")
    d = json.loads((tmp_path / "cpu-analysis/decision.json").read_text())
    assert not d["complete"] and "incomplete CPU proof" in d["failures"]
