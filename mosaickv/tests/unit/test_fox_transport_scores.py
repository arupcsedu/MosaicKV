"""Synthetic controls for CPU-only scoring of immutable consumed predictions."""

import copy

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
