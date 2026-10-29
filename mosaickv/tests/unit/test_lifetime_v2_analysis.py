"""Analysis must retain byte mismatch/failures and never promote a diagnostic."""

import json

import evicastkv_lifetime_v2 as module
import pytest

from mosaickv.evaluation.fox_campaign import sha


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
def test_v2_analysis_checks_realized_bytes_and_never_authorizes_paper(
    tmp_path, monkeypatch, mismatch, automatic
):
    labels = ["full_kv-native-control", "prefixkv_reimpl-native-control"]
    for cap in (192, 208):
        for method in ("prompt_attention_topk", "mosaickv_exact"):
            for policy in ("sink_recent", "recent_importance") if cap == 208 else ("sink_recent",):
                labels.append(f"{method}-{policy}-{cap}mib-v2")
    labels += [
        "mosaickv_exact-append-anchor-137.5mib-v2",
        "mosaickv_exact-sink_recent-208mib-cloned-control-v2",
    ]
    plan = dict(
        source_sha="fixture",
        automatic_consumed_followup=automatic,
        jobs=[dict(index=i, arms=[dict(label=k) for k in labels]) for i in (0, 1)],
    )
    monkeypatch.setattr(module.previous, "inputs", lambda *_args: plan)
    calls = []
    monkeypatch.setattr(module, "prepare_followup", lambda *args: "fixture-child")
    monkeypatch.setattr(module, "submit", lambda *args: calls.append(args))
    for index, task in enumerate(("llava_description", "mmvet_prefixkv")):
        worker = tmp_path / f"workers/{index}"
        (worker / "rows").mkdir(parents=True)
        (worker / "result.json").write_text(json.dumps(dict(failures=[])))
        trace = worker / "fixture.json"
        trace.write_text(json.dumps(dict(measurement_type="synthetic_test_only")))
        for label in labels:
            native = label.startswith("prefixkv")
            metrics = {k: 200 if native else 100 for k in module.previous.STORAGE}
            metrics["end_to_end_time"] = 1.0
            if mismatch and label == "mosaickv_exact-sink_recent-192mib-v2":
                metrics["peak_active_kv_storage_bytes"] += 1
            row = dict(
                task=task,
                sample_id=f"fixture-{index}",
                label=label,
                status="completed",
                metrics=metrics,
                traces={},
                memory_audit_traces={str(trace): sha(trace)},
                teacher_forced_ppl=3.1 if native else 3.0,
                rouge_l_f1=0.5 if native else 0.6,
                request_owned_peak_bytes=100,
                audited_request_owned_peak_bytes=100,
                pytorch_request_peak_allocated_bytes=200,
                memory_observer_token_parity=True,
            )
            (worker / f"rows/{label}.json").write_text(json.dumps(row))
    module.analyze(tmp_path, "fixture")
    gate = json.loads((tmp_path / "analysis/GATE.json").read_text())
    assert gate["status"] == ("failed" if mismatch else "complete_consumed_diagnostic")
    assert bool(gate["consumed_followup_eligible_labels"]) is not mismatch
    assert not gate["paper_campaign_authorized"]
    assert not gate["fresh_evaluation_authorized"]
    assert not gate["superiority_established"]
    assert len(calls) == int(automatic and not mismatch)
    if automatic:
        decision = json.loads((tmp_path / "recursive_decision.json").read_text())
        assert decision["no_fresh_or_paper_authorization"] is True


def test_followup_materializes_only_remaining_consumed_ids(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    label = "mosaickv_exact-sink_recent-192mib-v2"
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            dict(
                failures=[],
                consumed_followup_eligible_labels=[label],
            )
        )
    )
    (parent / "analysis/summary.parquet").write_bytes(b"synthetic_test_only")
    plan = dict(bound_files={}, jobs=[])
    for i in (0, 1):
        ledger = parent / f"ledger-{i}.json"
        ledger.write_text(
            json.dumps(
                dict(samples=[dict(sample_id=f"consumed-{i}-{k}", media=[]) for k in range(8)])
            )
        )
        plan["jobs"].append(
            dict(
                index=i,
                ledger=str(ledger),
                task=f"fixture-{i}",
                arms=[
                    dict(label=arm_label)
                    for arm_label in (
                        "full_kv-native-control",
                        "prefixkv_reimpl-native-control",
                        label,
                        label.replace("mosaickv_exact-", "prompt_attention_topk-"),
                    )
                ],
            )
        )
    (parent / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(module.previous, "inputs", lambda *_args: plan)
    monkeypatch.setattr(module, "clean_source", lambda *_args: "fixture")
    child = tmp_path / "child"
    module.prepare_followup(parent, sha(parent / "plan.json"), label, child)
    result = json.loads((child / "plan.json").read_text())
    assert len(result["jobs"]) == 14
    assert all(j["sample_id"] not in {"consumed-0-0", "consumed-1-0"} for j in result["jobs"])
    assert all(len(j["arms"]) == 4 for j in result["jobs"])
    assert result["automatic_consumed_followup"] is False
    assert result["fresh_evaluation_authorized"] is False
    assert result["paper_campaign_authorized"] is False
