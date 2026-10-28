import json
from pathlib import Path

import pytest


def test_regate_is_bounded_common_policy_and_never_launches_scoring(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_transport_regate as module

    from mosaickv.evaluation.fox_campaign import sha, write_once

    parent, diag, root = tmp_path / "parent", tmp_path / "diag", tmp_path / "retry"
    ids = ["ActionPrediction-0035", "ActionPrediction-0014"]
    ledger = parent / "ledger.json"
    write_once(
        ledger, dict(samples=[dict(sample_id=s) for s in ids] + [dict(sample_id="not_selected")])
    )
    write_once(
        parent / "protocol_lock.json",
        dict(
            cohorts={"milebench": {"validation": dict(path=str(ledger), sha256=sha(ledger))}},
            methods={
                "milebench": ["full_kv", "prompt_attention_topk", "lookm_reimpl", "mosaickv_exact"]
            },
            authorization={},
            resources={},
        ),
    )
    write_once(
        diag / "analysis.json",
        dict(
            transport_controls_passed=True,
            cases=[dict(sample_id=s, padding_control_passed=True, parity_passed=True) for s in ids],
        ),
    )
    monkeypatch.setattr(module, "clean_source", lambda *a: "synthetic")
    calls = []
    monkeypatch.setattr(module, "submit", lambda *a: calls.append(a))
    module.prepare(root, parent, diag)
    lock = json.loads((root / "protocol_lock.json").read_text())
    assert lock["arithmetic"]["attention_layout_policy"] == "valid_slot_decode"
    assert not lock["authorization"]["automatic_stage_progression"]
    assert not lock["authorization"]["confirmation_authorized"]
    assert lock["cohorts"]["milebench"]["validation"]["sample_ids"] == ids
    assert calls == [(root, "correctness")]
    with pytest.raises(FileExistsError):
        module.prepare(root, parent, diag)
