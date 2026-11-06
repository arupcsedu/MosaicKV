"""Synthetic CPU fixtures, never measured experiment numbers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation.coco_all_methods import (
    caption_text,
    complete_grid,
    endpoints,
    validate_row,
)
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.packed_attention_decode import AppendAudit


def row(**changes):
    return (
        dict(
            status="completed",
            used_fullkv_fallback=False,
            generated_token_ids=[1] * 128,
            answer=dict(
                answer="caption",
                answer_tokens=1,
                stopped_by_eos=True,
                output_limit_reached=False,
                prefix_token_cap=128,
                token_ids=[1],
            ),
            end_to_end_seconds=1.0,
            initial_encoded_bytes=100,
            final_encoded_bytes=120,
            decode_owned_peak_bytes=130,
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"decode_owned_peak_bytes": None},
        {"decode_owned_peak_bytes": 119},
        {"initial_encoded_bytes": True},
        {"end_to_end_seconds": float("nan")},
        {"end_to_end_seconds": 0},
        {"status": "failed"},
        {"used_fullkv_fallback": True},
        {"generated_token_ids": [1] * 127},
    ],
)
def test_missing_or_invalid_measurements_never_promoted(changes):
    with pytest.raises(ValueError):
        validate_row(row(**changes), audited=True)


def test_fullkv_and_encoded_endpoints_and_unaudited_unknown_peak():
    validate_row(row(), audited=True)
    validate_row(row(decode_owned_peak_bytes=None), audited=False)
    assert endpoints(
        dict(
            storage=dict(initial_active_kv_storage_bytes=10, final_active_kv_storage_bytes=12),
            decode_owned_peak_bytes=15,
        )
    ) == (10, 12, 15)


def test_structured_eos_caption_and_legacy_text_are_scored_without_losing_metadata():
    record = row()
    before = dict(record["answer"])
    assert caption_text(record) == "caption"
    assert record["answer"] == before and record["answer"]["stopped_by_eos"]
    assert caption_text(row(answer="legacy")) == "legacy"
    assert caption_text(row(answer={"answer": ""})) == ""
    for invalid in (None, {}, {"answer": None}, {"answer": ["caption"]}):
        with pytest.raises(ValueError, match="caption text required"):
            caption_text(row(answer=invalid))


def test_exact_grid_rejects_missing_duplicate_and_extra_trials():
    data = [
        dict(method=m, sample_index=i, trial=t)
        for m in ("a", "b")
        for i in (1, 2)
        for t in range(2)
    ]
    complete_grid(data, ("a", "b"), (1, 2), trials=2)
    for invalid in (data[:-1], [*data, data[0]], [*data, dict(method="c", sample_index=1)]):
        with pytest.raises(ValueError):
            complete_grid(invalid, ("a", "b"), (1, 2), trials=2)


def test_native_append_peak_counts_old_and_new_storage_without_mutating_values():
    key = torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 3, 2)
    layer = SimpleNamespace(keys=key.clone(), values=key.clone())
    cache = SimpleNamespace(layers=[layer])
    audit = AppendAudit(cache)
    with audit:
        layer.keys = torch.cat([layer.keys, key[..., :1, :]], dim=2)
        layer.values = torch.cat([layer.values, key[..., :1, :]], dim=2)
    assert audit.appends == 2
    final = backing_bytes((layer.keys, layer.values))
    assert audit.peak > final
    assert torch.equal(layer.keys, torch.cat([key, key[..., :1, :]], dim=2))


def test_canary_success_launches_only_registered_memory_grid(tmp_path):
    import evicastkv_coco_all_methods as driver

    protocol = {"timing_subset": [2, 45, 88, 131, 174, 217]}
    rows = [
        row(
            method=m,
            sample_id="x",
            sample_index=2,
            trial=0,
            audit_fast_128_logits_bitwise_identical=True,
            manifest_path="manifest.json",
        )
        for m in driver.METHODS
    ]
    with (
        patch.object(driver, "check", return_value=({"source_sha": "fixed"}, protocol)),
        patch.object(driver, "collect", return_value=(rows, [])),
        patch.object(driver, "launch") as launch,
    ):
        driver.analyze(tmp_path, "hash", "canary")
    phase, jobs = launch.call_args.args[1:]
    assert phase == "memory" and len(jobs) == 96
    assert all(len(j["samples"]) == 16 and len(j["methods"]) == 1 for j in jobs)
    assert {i for j in jobs for i in j["samples"]} == set(range(2, 258))


def test_failed_canary_is_recorded_and_blocks_all_gpu_progression(tmp_path):
    import evicastkv_coco_all_methods as driver

    with (
        patch.object(driver, "check", return_value=({"source_sha": "fixed"}, {})),
        patch.object(driver, "collect", return_value=([], ["parity failed"])),
        patch.object(driver, "launch") as launch,
    ):
        driver.analyze(tmp_path, "hash", "canary")
    launch.assert_not_called()
    assert not driver.campaign.read(tmp_path / "canary/analysis/decision.json")["complete"]


def test_scope_never_claims_byte_matching_or_inactive_components():
    import evicastkv_coco_all_methods as driver

    protocol = driver.campaign.read(driver.CONFIG)
    assert not protocol["shared_realized_byte_budget"]
    assert not protocol["original_six_mechanisms_active"]
    assert not protocol["whole_request_KV_peak_certified"]
    assert protocol["maximum_a100s"] == 6 and protocol["worker_minutes"] == 120
    script = driver.SCRIPT.read_text()
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "cache_env.sh" in script and "assert_clean_worktree.sh" in script


def test_complete_summary_uses_all_references_and_keeps_metric_boundaries(tmp_path):
    import evicastkv_coco_all_methods as driver
    import numpy as np

    protocol = driver.campaign.read(driver.CONFIG)
    memory = [
        row(method=m, sample_id=f"id-{i}", sample_index=i, trial=0)
        for m in driver.METHODS
        for i in driver.INDICES
    ]
    repeated = [
        row(method=m, sample_id=f"id-{i}", sample_index=i, trial=t)
        for m in driver.METHODS[1:]
        for i in protocol["timing_subset"]
        for t in range(20)
    ]
    ledger = dict(
        samples=[dict(references=["a", "b", "c", "d", "e", "f", "g"]) for _ in range(258)]
    )
    for phase in ("memory", "timing", "repeated"):
        driver.write_once(tmp_path / phase / "analysis/all_attempts.json", [])

    class FixtureEngine:
        def __init__(self, groups):
            assert all(len(samples) == 256 for samples in groups.values())
            assert all(len(s.references) == 7 for samples in groups.values() for s in samples)
            self.points = tuple(sorted(groups, key=str))

        def score(self, counts):
            return np.ones((len(counts), len(self.points)))

        def validate_official(self, counts, scores):
            assert counts.shape == (1, 256) and scores.shape == (1, 6)

    with (
        patch.object(driver, "check", return_value=({"source_sha": "fixed"}, protocol)),
        patch.object(driver, "collect", side_effect=[(memory, []), (memory, []), (repeated, [])]),
        patch.object(driver.campaign, "load_ledger", return_value=ledger),
        patch.object(driver.campaign, "timing_comparisons", return_value=[]),
        patch("mosaickv.evaluation.cider_resampling.ResampledCider", FixtureEngine),
        patch(
            "mosaickv.evaluation.cider_resampling.paired_draw_counts",
            return_value=np.ones((3, 256), dtype=np.int64),
        ),
    ):
        driver.final_analysis(tmp_path, "hash")
    result = driver.campaign.read(tmp_path / "summary/summary.json")
    assert len(result["rows"]) == 6 and len(result["quality"]) == 5
    assert not result["same_realized_byte_budget"]
    assert result["audited_timing_excluded_from_latency"]
    assert not result["whole_request_KV_peak_certified"]
    assert all(r["completed"] == 256 and r["timing_trials"] == 256 for r in result["rows"])
