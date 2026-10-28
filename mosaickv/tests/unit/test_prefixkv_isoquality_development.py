"""CPU-only controls for the bounded parallel native-metric comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import prefixkv_isoquality_development as development
import pytest
import yaml
from prefixkv_isoquality_development import (
    PROTOCOL,
    STORAGE,
    choose_ids,
    controls,
    latency_interval,
    paired_metrics,
    validate_protocol,
    validate_rows,
)


def test_protocol_static_gpu_split():
    p = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(p)
    p["resources"]["prefixkv_array_concurrency"] = 6
    with pytest.raises(ValueError, match="six A100s"):
        validate_protocol(p)


@pytest.mark.parametrize(
    "field,value",
    [("output_tokens", 64), ("samples_per_task", 128), ("paper_campaign_authorized", True)],
)
def test_scope_cannot_silently_expand(field, value):
    p = yaml.safe_load(PROTOCOL.read_text())
    p[field] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_identity_selection_does_not_read_labels():
    class LabelsForbidden(dict):
        def __getitem__(self, key):
            if key != "sample_id":
                raise AssertionError("labels cannot select a development cohort")
            return super().__getitem__(key)

    samples = [LabelsForbidden(sample_id=str(i)) for i in range(40)]
    selected = choose_ids(samples, {"1", "2"}, count=8, seed=17)
    assert len(set(selected)) == 8
    assert not set(selected) & {"1", "2"}
    assert selected == choose_ids(list(reversed(samples)), {"1", "2"}, count=8, seed=17)
    with pytest.raises(ValueError, match="not enough"):
        choose_ids(samples, {str(i) for i in range(40)}, count=1, seed=17)


def fixture_rows():
    return [
        dict(
            sample_id="a",
            status="completed",
            error=None,
            generated_tokens=1024,
            prefill_time=1.0,
            compression_time=0.2,
            decode_time=2.0,
            ttft=1.2,
            end_to_end_time=3.2,
            residual_kv_bytes=0,
            repair_count=0,
            **{STORAGE[0]: 80 * 2**20, STORAGE[1]: 591 * 2**20, STORAGE[2]: 591 * 2**20},
        )
    ]


def test_append_only_final_bytes_need_not_equal_initial_cap():
    validate_rows(fixture_rows(), ["a"], 80)


@pytest.mark.parametrize(
    "field,value",
    [
        (STORAGE[0], 80 * 2**20 - 1),
        (STORAGE[2], 2),
        ("end_to_end_time", float("nan")),
        ("generated_tokens", 64),
        ("repair_count", 1),
    ],
)
def test_bad_storage_timing_or_truncation_fails(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_rows(rows, ["a"], 80)


def test_missing_duplicate_samples_fail():
    rows = fixture_rows()
    with pytest.raises(ValueError):
        validate_rows(rows + rows, ["a", "b"], 80)


def test_ppl_is_corpus_reduction_not_mean_sample_ppl():
    c = pd.DataFrame(dict(sample_id=["a", "b"], numerator=[1.0, 300.0], denominator=[1, 100]))
    r = c.copy()
    r["numerator"] = [2.0, 400.0]
    stats = paired_metrics(c, r, "teacher_forced_ppl", replicates=128)
    assert stats["difference"] == pytest.approx(np.exp(301 / 101) - np.exp(402 / 101))
    assert stats["passes"] is True


def test_metric_directions_and_pairing():
    r = pd.DataFrame(dict(sample_id=["a", "b"], metric_value=[0.5, 0.5]))
    c = r.copy()
    c["metric_value"] = [0.6, 0.7]
    assert paired_metrics(c, r, "rouge_l_f1", replicates=128)["passes"] is True
    assert paired_metrics(r, c, "rouge_l_f1", replicates=128)["passes"] is False
    c["sample_id"] = ["a", "c"]
    with pytest.raises(ValueError, match="identical"):
        paired_metrics(c, r, "rouge_l_f1", replicates=128)


def test_paired_hierarchical_timing():
    c = np.ones((2, 20)) * 9
    r = np.ones((2, 20)) * 10
    result = latency_interval(c, r, replicates=128)
    assert result["mean_latency_ratio"] == pytest.approx(0.9)
    assert result["no_slower_gate"]
    assert not latency_interval(r, c, replicates=128)["no_slower_gate"]
    with pytest.raises(ValueError):
        latency_interval(c[:, :19], r[:, :19])


def test_shared_controls_ignore_only_algorithm_and_cap():
    data = dict(
        model={"revision": "abc"},
        dataset={"revision": "def"},
        execution={"attention": "eager"},
        generation={"max_new_tokens": 1024},
        cache={
            key: 1
            for key in (
                "block_size",
                "physical_budget_block_size",
                "physical_budget_mandatory_policy",
                "packing_mode",
                "exact_gather_mode",
            )
        },
    )
    changed = copy.deepcopy(data)
    changed["cache"]["budget_value"] = 123
    assert controls(data) == controls(changed)
    changed["generation"]["max_new_tokens"] = 64
    assert controls(data) != controls(changed)


def test_all_historical_compressed_configs_have_identical_controls():
    root = Path("/scratch/djy8hg/runs/mosaickv/prefixkv/full-campaign-e4de390-20261019-v6")
    if not root.exists():
        pytest.skip("external immutable fixtures not available")
    for task in ("llava_description", "mmvet_prefixkv"):
        values = [
            json.loads((root / f"configs/{task}/shared_byte_40mib-{m}.json").read_text())
            for m in ("full_kv", "prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact")
        ]
        assert all(controls(v) == controls(values[0]) for v in values)


def test_failed_references_never_release_compressed_array(tmp_path, monkeypatch):
    plan = dict(jobs=[dict(index=0, method="full_kv")])
    monkeypatch.setattr(development, "inputs", lambda *args: plan)

    def forbidden(*args, **kwargs):
        raise AssertionError("failed prerequisite must not submit a GPU job")

    monkeypatch.setattr(development, "sbatch", forbidden)
    development.advance(tmp_path, "test-only", "after-references")
    assert json.loads((tmp_path / "decisions/references-failed.json").read_text())["status"] == (
        "blocked_reference_failure"
    )


def test_missing_worker_denominators_are_preserved(tmp_path, monkeypatch):
    plan = dict(
        jobs=[dict(index=0, task="llava_description", method="full_kv", sample_ids=["a", "b"])]
    )
    monkeypatch.setattr(development, "inputs", lambda *args: plan)
    development.analyze(tmp_path, "test-only")
    failures = json.loads((tmp_path / "analysis/failures.json").read_text())
    assert failures["planned_sample_rows"] == 2
    assert failures["failed_or_missing_sample_rows"] == 2
    assert json.loads((tmp_path / "analysis/GATE.json").read_text())["status"] == "failed_execution"


def test_submission_uses_afterany_and_static_three_gpu_cap(tmp_path, monkeypatch):
    protocol = yaml.safe_load(PROTOCOL.read_text())
    (tmp_path / "plan.json").write_text(json.dumps(dict(protocol=protocol)))
    captured = []

    def fake_submit(command, **kwargs):
        captured.append(command)
        return "12345\n"

    monkeypatch.setattr(development.subprocess, "check_output", fake_submit)
    development.sbatch(tmp_path, "test-only", mode="worker", indices=[1, 2, 3])
    assert "--array=1,2,3%3" in captured[-1]
    assert "--reservation=bi_fox_dgx" in captured[-1]
    assert "--cpus-per-task=12" in captured[-1]
    assert "--time=02:00:00" in captured[-1]
    development.sbatch(tmp_path, "test-only", mode="analyze", dependency="12345")
    assert "--dependency=afterany:12345" in captured[-1]
    assert "--gres=none" in captured[-1]
