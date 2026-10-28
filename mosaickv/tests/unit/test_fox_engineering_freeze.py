import json
from pathlib import Path

import pytest


def test_engineering_freeze_has_no_paper_approval_or_automatic_successor(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_engineering_freeze as module

    from mosaickv.evaluation.fox_campaign import sha, write_once

    parent, pilot, output = tmp_path / "parent", tmp_path / "pilot", tmp_path / "frozen"
    ledger = parent / "ledger.json"
    write_once(ledger, dict(samples=[dict(sample_id=str(i)) for i in range(3)]))
    template = module.SOURCE / "mosaickv/configs/experiments/evicastkv_fox/coco/mosaickv_exact.json"
    write_once(
        parent / "protocol_lock.json",
        dict(
            authorization={},
            resources={},
            templates=dict(coco=dict(path=str(template))),
            cohorts=dict(coco=dict(validation=dict(path=str(ledger), sha256=sha(ledger)))),
        ),
    )
    write_once(pilot / "protocol_lock.json", dict(source_root=str(module.SOURCE)))
    write_once(
        pilot / "analysis/full_graph_pilot/GATE_full_graph_pilot.json",
        dict(status="passed", tests_actually_executed=dict(completed=16)),
    )
    monkeypatch.setattr(module, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(module, "load_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(module, "submit", lambda *a: calls.append(a))
    module.freeze(output, parent, pilot)
    frozen = json.loads((output / "engineering_config_lock.json").read_text())
    lock = json.loads((output / "protocol_lock.json").read_text())
    assert frozen["status"] == "engineering_frozen_not_paper_approved"
    assert not frozen["confirmation_authorized"] and not frozen["human_reviewed"]
    assert not lock["authorization"]["automatic_stage_progression"]
    assert len(frozen["configs"]) == 7
    assert lock["cohorts"]["coco"]["validation"]["sample_ids"] == ["0", "1"]
    assert len(json.loads(ledger.read_text())["samples"]) == 3
    assert calls == [(output, "correctness")]


def test_freeze_rejects_non_scratch_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_engineering_freeze as module

    with pytest.raises(ValueError, match="scratch-only"):
        module.freeze(Path("/tmp/not-a-run"), Path("/tmp/a"), Path("/tmp/b"))
