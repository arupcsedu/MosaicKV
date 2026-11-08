import gzip
import json

import exact_scale_replay_study as study
import torch

from mosaickv.evaluation.fox_campaign import sha


def test_feasibility_counts_replay_and_does_not_double_count_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "PARENT", tmp_path)
    bound = {}
    for i in (0, 1):
        folder = tmp_path / f"model/{i}"
        folder.mkdir(parents=True)
        rows = {
            (layer, "source"): dict(
                minimum=torch.ones(512, dtype=torch.bfloat16),
                scale=(torch.arange(512) % 7 + 1).float(),
            )
            for layer in range(32)
        }
        path = folder / "headers.pt"
        torch.save(rows, path)
        bound[str(path)] = sha(path)
    proof = study.header_feasibility(dict(bound_files=bound))
    assert proof["both_inputs_positive"]
    for r in proof["cases"]:
        assert r["charged_replay_bytes"] == 512 * 4
        assert r["net_saving_bytes"] == (
            r["original_scale_bytes"] - r["encoded_scale_bytes"] - r["charged_replay_bytes"]
        )
    assert proof["online_encoding_not_yet_timed"]


def test_missing_workers_never_promote_or_claim_superiority(tmp_path):
    study.analyze(tmp_path, {})
    proof = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not proof["complete"]
    assert not proof["all_initial_final_peak_savings"]
    assert not proof["micro_prototype_or_history_followup_authorized"]
    assert not proof["strict_latency_established"]
    assert not proof["six_component_superiority_established"]


def test_prepare_reuses_only_valid_controls_and_submits_cpu(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "plan.json").write_text(json.dumps(dict(bound_files={}, guard_tokens=32)))
    monkeypatch.setattr(study, "PARENT", parent)
    monkeypatch.setattr(study, "clean_source", lambda *_: "a" * 40)
    monkeypatch.setattr(
        "mosaickv.measurements.kv_timeline.valid_whole_request_certificate", lambda _: True
    )
    for i in (0, 1):
        folder = parent / f"model/{i}"
        folder.mkdir(parents=True)
        (folder / "headers.pt").write_bytes(b"CPU fixture only")
        ledger = folder / "ledger.gz"
        with gzip.open(ledger, "wt") as stream:
            json.dump({}, stream)
        row = dict(
            arm="control_common",
            status="completed",
            numerically_admissible=True,
            whole_request_certificate_valid=True,
            headers_sha=sha(folder / "headers.pt"),
            ledger_sha=sha(ledger),
            ledger_path=str(ledger),
        )
        (folder / "raw.jsonl").write_text(json.dumps(row) + "\n")
    calls = []
    monkeypatch.setattr(
        study.common, "submit", lambda root, action, *_, **__: calls.append(action) or "fixture"
    )
    root = tmp_path / "output"
    study.prepare(root)
    p = json.loads((root / "plan.json").read_text())
    assert calls == ["cpu", "after_cpu"]
    assert p["guard_tokens"] == 32
    assert p["arms"] == study.ARMS
    assert not p["fresh_or_paper_campaign_authorized"]
    assert not p["independent_quality_validation_authorized"]
