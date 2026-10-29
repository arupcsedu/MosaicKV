"""A positive consumed-input diagnostic must never become a paper gate."""

import json

import evicastkv_lifetime_history as module


def test_complete_diagnostic_cannot_authorize_confirmation(tmp_path, monkeypatch):
    plan = {"source_sha": "fixture", "jobs": []}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(module, "inputs", lambda *args: plan)
    labels = [
        "full_kv-native-control",
        "prefixkv_reimpl-native-control",
        "evicast-append-anchor-90mib",
        "prompt_attention_topk-sink_recent-192mib",
        "prompt_attention_topk-recent_importance-192mib",
        "mosaickv_exact-sink_recent-192mib",
        "mosaickv_exact-recent_importance-192mib",
    ]
    for i, task in enumerate(("llava_description", "mmvet_prefixkv")):
        out = tmp_path / f"workers/{i}/rows"
        out.mkdir(parents=True)
        (out.parent / "result.json").write_text(json.dumps({"failures": []}))
        for label in labels:
            baseline = label.startswith("prefixkv")
            r = dict(
                task=task,
                sample_id=f"consumed-{i}",
                label=label,
                status="completed",
                teacher_forced_ppl=3.1 if baseline else 3.0,
                rouge_l_f1=0.6 if baseline else 0.7,
                metrics={k: 200 if baseline else 100 for k in module.STORAGE},
                request_owned_peak_bytes=300,
            )
            r["metrics"]["end_to_end_time"] = 1.0
            (out / (label + ".json")).write_text(json.dumps(r))
    module.analyze(tmp_path, "fixture")
    gate = json.loads((tmp_path / "analysis/GATE.json").read_text())
    assert gate["status"] == "complete_consumed_diagnostic"
    assert all(
        c["smaller_all_decode_boundaries"]
        and c["ppl_point_not_worse"]
        and c["rouge_point_not_worse"]
        for c in gate["checks"]
    )
    assert not gate["quality_preservation_statistically_established"]
    assert not gate["superiority_established"]
    assert not gate["whole_request_peak_certified"]
    assert not gate["fresh_evaluation_authorized"]
    assert not gate["large_campaign_authorized"]
