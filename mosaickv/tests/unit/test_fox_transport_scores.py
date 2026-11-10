"""Synthetic controls for CPU-only scoring of immutable consumed predictions."""

import copy
import json

import evicastkv_transport_scores as module
import pytest

from mosaickv.evaluation.fox_campaign import TASKS, sha, write_once


@pytest.fixture
def grid(tmp_path):
    samples, rows = [], []
    for cap in module.CAPS:
        path = tmp_path / f"config-{cap}.json"
        write_once(
            path,
            dict(
                model=dict(id="synthetic", revision="a" * 40, precision="fp16"),
                dataset=dict(id="synthetic", revision="b" * 40),
                generation=dict(max_new_tokens=64),
                execution=dict(backend="huggingface"),
                cache=dict(
                    accounting_spec_sha="s",
                    block_size=1,
                    budget_unit="bytes",
                    budget_value=cap,
                    exact_gather_mode="source_views",
                    packing_mode="per_head_reference",
                    physical_budget_block_size=1,
                    physical_budget_mandatory_policy="first_and_last",
                    total_storage_budget_bytes=None,
                ),
            ),
        )
    for task in TASKS:
        for n in range(8):
            sample = dict(
                sample_id=f"{task}-{n:04d}",
                milebench_dataset=task,
                milebench_question_type="multi-choice" if task in TASKS[:2] else "open-ended",
                references=["fixture"],
                choices=[],
            )
            samples.append(sample)
            for cap in module.CAPS:
                path = tmp_path / f"config-{cap}.json"
                for method in module.METHODS:
                    rows.append(
                        dict(
                            sample_id=sample["sample_id"],
                            subtask=task,
                            method=method,
                            actual_method=method,
                            effective_method=method,
                            budget_bytes=cap,
                            status="completed",
                            error=None,
                            arithmetic_policy=copy.deepcopy(module.ARITHMETIC),
                            references=sample["references"],
                            choices=[],
                            question_type=sample["milebench_question_type"],
                            emitted_tokens=64,
                            forward_steps=63,
                            token_ids=[1] * 64,
                            resolved_config_path=str(path),
                            resolved_config_file_sha256=sha(path),
                            input_sha=sample["sample_id"],
                            processor_sha="processor",
                            scorer_sha="scorer",
                            workload="milebench",
                            repeat=0,
                            initial_kv_bytes=cap,
                            final_kv_bytes=cap + 63,
                            peak_kv_bytes=cap + 63,
                        )
                    )
    return rows, samples


def test_complete_exact_byte_grid_is_accepted_independently_of_answers(grid):
    rows, samples = grid
    assert len(module.validate_rows(rows, samples)) == 64
    for r in rows:
        r["answer"] = "an unrelated output must not influence admission"
    assert len(module.validate_rows(rows, samples)) == 64


@pytest.mark.parametrize(
    "change",
    [
        dict(effective_method="full_kv"),
        dict(status="failed"),
        dict(references=["wrong"]),
        dict(question_type="wrong"),
        dict(emitted_tokens=63),
        dict(forward_steps=62),
        dict(arithmetic_policy=dict(attention_layout_policy="native")),
    ],
)
def test_invalid_or_incompatible_rows_fail_closed(grid, change):
    rows, samples = grid
    rows[0].update(change)
    with pytest.raises(ValueError, match="mismatch"):
        module.validate_rows(rows, samples)


def test_missing_and_duplicate_predictions_cannot_form_a_surviving_intersection(grid):
    rows, samples = grid
    with pytest.raises(ValueError, match="denominator"):
        module.validate_rows(rows[:-1], samples)
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_rows(rows + rows[:1], samples)


def test_initial_final_and_peak_realized_bytes_all_must_match(grid):
    rows, samples = grid
    for field in ("initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes"):
        broken = copy.deepcopy(rows)
        broken[0][field] -= 1
        with pytest.raises(ValueError, match="storage mismatch"):
            module.validate_rows(broken, samples)


@pytest.fixture
def quality_contrast():
    return dict(
        reference_value=0.2,
        candidate_value=0.25,
        difference=0.05,
        relative_difference=0.25,
        ordinary_95_ci=[-0.01, 0.1],
        simultaneous_ci=[-0.02, 0.12],
        speedup_ratio_of_means=0.8,
        speedup_95_ci=[0.7, 0.9],
        speedup_simultaneous_ci=[0.6, 1.0],
        speedup_future_field=123,
        percentage_reduction=-25,
        family_size=6,
        resamples=10000,
        seed=20271014,
        unique_inputs=16,
    )


def test_quality_statistics_never_present_quality_ratios_as_latency(quality_contrast):
    original = copy.deepcopy(quality_contrast)
    corrected = module.quality_statistics("rouge_l_f1", quality_contrast)
    assert quality_contrast == original
    assert not any(key.startswith("speedup_") for key in corrected)
    assert "percentage_reduction" not in corrected
    assert "relative_difference" not in corrected
    assert corrected["relative_quality_difference"] == pytest.approx(0.25)
    assert corrected["metric_direction"] == "higher"
    assert corrected["timing_inference_allowed"] is False
    for field in (
        "reference_value",
        "candidate_value",
        "difference",
        "ordinary_95_ci",
        "simultaneous_ci",
    ):
        assert corrected[field] == original[field]


def test_zero_quality_reference_has_no_relative_effect(quality_contrast):
    quality_contrast.update(reference_value=0.0, candidate_value=0.05)
    assert (
        module.quality_statistics("accuracy", quality_contrast)["relative_quality_difference"]
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(candidate_value=float("nan")),
        dict(reference_value=-0.1),
        dict(candidate_value=1.1),
        dict(difference=0.2),
        dict(ordinary_95_ci=[0.1, -0.1]),
        dict(simultaneous_ci=[float("-inf"), 0.1]),
    ],
)
def test_invalid_quality_statistics_are_rejected(quality_contrast, change):
    quality_contrast.update(change)
    with pytest.raises(ValueError, match="quality"):
        module.quality_statistics("accuracy", quality_contrast)


def test_latency_metrics_cannot_enter_quality_report(quality_contrast):
    with pytest.raises(ValueError, match="quality metrics"):
        module.quality_statistics("end_to_end_seconds", quality_contrast)


@pytest.fixture
def saved_quality(tmp_path, monkeypatch, quality_contrast):
    monkeypatch.setattr(module, "clean_source", lambda *args: "c" * 40)
    root = tmp_path / "source"
    evidence = tmp_path / "evidence.json"
    write_once(evidence, dict(synthetic_fixture=True))
    registration = dict(
        expected_rows=128,
        bindings={str(evidence): sha(evidence)},
        statistical_family_size=6,
        bootstrap_resamples=10000,
        seed=20271014,
    )
    write_once(root / "registration.json", registration)
    write_once(
        root / "GATE_scoring.json",
        dict(
            status="passed_bounded_retrospective_scoring_not_confirmation",
            registration_sha256=sha(root / "registration.json"),
            source_sha="d" * 40,
            completed=128,
            expected=128,
            failed=0,
            official_scorer_fixtures_passed=True,
            timing_inference_allowed=False,
            fresh_confirmation_authorized=False,
            superiority_claim_allowed=False,
        ),
    )
    rows = [
        dict(
            quality_contrast,
            metric_name=metric,
            budget_bytes=cap,
            reference=module.METHODS[0],
            candidate=module.METHODS[1],
        )
        for metric in ("accuracy", "rouge_l_f1")
        for cap in module.CAPS
    ]
    write_once(root / "paired_statistics.json", rows)
    return root, tmp_path / "derived", evidence


def test_quality_reexport_is_derived_hash_bound_and_write_once(saved_quality):
    root, output, evidence = saved_quality
    originals = {p: p.read_bytes() for p in root.iterdir()}
    module.reexport(root, output)
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    rows = json.loads((output / "paired_quality_statistics.json").read_text())
    gate = json.loads((output / "GATE_quality_statistics.json").read_text())
    assert len(rows) == 4
    assert all(not any(k.startswith("speedup_") for k in row) for row in rows)
    assert gate["new_GPU_inference"] == 0
    assert gate["bootstrap_recomputed"] is False
    assert gate["superiority_claim_allowed"] is False
    for path in (*originals, evidence):
        assert gate["bindings"][str(path)] == sha(path)
    with pytest.raises(FileExistsError, match="must be new"):
        module.reexport(root, output)


@pytest.mark.parametrize(
    "defect",
    ["failed", "authorization", "registration", "evidence", "missing", "duplicate", "wrong_method"],
)
def test_quality_reexport_rejects_invalid_evidence_or_incomplete_grid(saved_quality, defect):
    root, output, evidence = saved_quality
    gate_path = root / "GATE_scoring.json"
    gate = json.loads(gate_path.read_text())
    if defect == "failed":
        gate["failed"] = 1
    elif defect == "authorization":
        gate["fresh_confirmation_authorized"] = True
    elif defect == "registration":
        gate["registration_sha256"] = "wrong"
    elif defect == "evidence":
        evidence.write_text("changed synthetic evidence")
    else:
        path = root / "paired_statistics.json"
        rows = json.loads(path.read_text())
        if defect == "missing":
            rows.pop()
        elif defect == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[0]["candidate"] = "full_kv"
        path.write_text(json.dumps(rows))
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        module.reexport(root, output)
    assert not output.exists()
