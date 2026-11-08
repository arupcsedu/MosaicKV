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


@pytest.mark.parametrize(
    "invalid", ("incomplete", "failed_gate", "changed_threshold", "changed_code")
)
def test_reuse_rejects_incompatible_cpu_evidence(tmp_path, monkeypatch, invalid):
    study = load()
    (tmp_path / "plan.json").write_text(
        json.dumps(
            dict(
                source_sha="fixture_sha",
                bound_files={},
                source_archive_root=str(study.CAPTURES),
                GPU_predictor="delta_add_only",
                cpu_predictors=["xor1", "subtract1", "xor8", "subtract8", "delta_add"],
                minimum_source_stream_plus_replay_saving_fraction=0.01
                if invalid == "changed_threshold"
                else 0.02,
            )
        )
    )
    (tmp_path / "cpu-analysis").mkdir()
    (tmp_path / "cpu-analysis/decision.json").write_text(
        json.dumps(
            dict(
                complete=invalid != "incomplete",
                errors=[],
                CPU_source_gate_passed=invalid != "failed_gate",
            )
        )
    )
    monkeypatch.setattr(study.subprocess, "check_output", lambda *args, **kwargs: b"changed-code")
    with pytest.raises(ValueError):
        study.reusable_cpu(tmp_path)


@pytest.mark.parametrize("audit_valid", (True, False))
def test_analysis_cannot_promote_unaudited_counters_to_audited_peak(
    tmp_path, monkeypatch, audit_valid
):
    study = load()
    monkeypatch.setattr(
        study,
        "checked",
        lambda *args: dict(
            warmups=1, repetitions=20, seed=17, bootstrap_replicates=100, multiplicity_comparisons=2
        ),
    )
    for i in (0, 1):
        folder = tmp_path / f"gpu/{i}"
        folder.mkdir(parents=True)
        rows = []
        for phase, count in (("warmup", 1), ("timed", 20), ("memory_audit", 1)):
            for trial in range(count):
                for a, arm in enumerate(study.ARMS):
                    rows.append(
                        dict(
                            phase=phase,
                            trial=trial,
                            arm=arm,
                            status="completed",
                            sample_id=f"synthetic_fixture_{i}",
                            all_128_frozen_logits_bitwise=True,
                            request_wall_seconds=1.0 - a * 0.1,
                            detail=dict(
                                initial_encoded_bytes=100 - a,
                                final_encoded_bytes=200 - a,
                                reported_counter_peak_bytes=300 - a,
                                peak_measurement_admissible=audit_valid,
                                tensor_result_auditing=audit_valid,
                                whole_request_peak_allocated_bytes=1000,
                                used_fullkv_fallback=False,
                            ),
                        )
                    )
        (folder / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (folder / "result.json").write_text(json.dumps(dict(status="completed", errors=[])))
    study.analyze(tmp_path, "unused")
    decision = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert decision["complete"] == audit_valid
    assert decision["engineering_memory_and_latency_gate_passed"] == audit_valid
    assert decision["whole_request_peak_certified"] is False
