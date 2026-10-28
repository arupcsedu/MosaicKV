import json
from pathlib import Path


def test_successful_correctness_cannot_override_disabled_promotion(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_analyze as module

    (tmp_path / "protocol_lock.json").write_text("{}")
    monkeypatch.setattr(
        module,
        "load_lock",
        lambda p: dict(source_sha="a" * 40, authorization=dict(automatic_stage_progression=False)),
    )
    monkeypatch.setattr(module, "collect", lambda *a: (dict(jobs=[]), [], [], []))
    calls = []
    monkeypatch.setattr(module, "submit", lambda *a, **k: calls.append(a))
    module.analyze(tmp_path, "correctness")
    assert not calls
    decision = json.loads((tmp_path / "analysis/correctness/completion_decision.json").read_text())
    assert decision["action"] == "stop_at_registered_stage_boundary"
