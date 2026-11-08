import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/complete_owner_study.py"
    spec = importlib.util.spec_from_file_location("owner_study_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ("pass", "under_threshold", "missing", "failed", "bits"))
def test_gpu_progression_requires_both_complete_unchanged_cpu_gates(tmp_path, monkeypatch, outcome):
    study = load()
    monkeypatch.setattr(
        study,
        "checked",
        lambda *args: dict(layers=2, minimum_source_stream_plus_replay_saving_fraction=0.02),
    )
    submitted = []

    def submit(root, action, after=None):
        submitted.append((action, after))
        return action + "_job"

    monkeypatch.setattr(study, "submit", submit)
    for i in (0, 1):
        if outcome == "missing" and i == 1:
            continue
        folder = tmp_path / f"cpu/{i}"
        folder.mkdir(parents=True)
        result = dict(
            status="completed",
            errors=[],
            rows=[
                dict(
                    layer=layer,
                    common_bytes=1000,
                    additive_or_common_bytes=995 if outcome == "under_threshold" else 900,
                    source_replay_bytes=200,
                    original_code_bits_exact=outcome != "bits",
                    all_shared_consumers_covered=True,
                )
                for layer in range(2)
            ],
        )
        if outcome == "failed" and i == 1:
            result.update(status="failed", errors=["synthetic failure"])
        (folder / "result.json").write_text(json.dumps(result))
    study.follow(tmp_path, "unused")
    decision = json.loads((tmp_path / "cpu-analysis/decision.json").read_text())
    assert decision["CPU_source_gate_passed"] == (outcome == "pass")
    if outcome == "pass":
        assert submitted == [("gpu", None), ("controller", None), ("analyze", "gpu_job")]
        assert (tmp_path / "admission.json").exists()
    else:
        assert submitted == []
        assert not (tmp_path / "gpu-submission.json").exists()
