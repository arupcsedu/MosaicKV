"""Synthetic controls/bytes only; no benchmark labels, downloads or GPU work."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mosaickv.evaluation.shared_byte_contract import (
    CONTROL_FIELDS,
    digest,
    load_contract,
    reuse_decision,
    validate_group,
    validate_ledger,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle

CONTRACT = (
    Path(__file__).resolve().parents[2] / "configs/experiments/shared_byte_comparison_v2.yaml"
)

CERTIFICATE = dict(
    physical_byte_cap=1000,
    largest_feasible_initial_bytes=960,
    boundary="unique_live_backing_storage",
    status="passed",
)


def test_routes_caps_resources_and_separate_metrics():
    c = load_contract(CONTRACT)
    assert c["resources"]["max_global_a100s"] == 4
    assert c["resources"]["cpus_per_worker"] == 12
    assert not c["execution_authorized"] and not c["fullkv_latency_comparison"]
    prefix = c["lanes"]["prefixkv"]["tasks"]
    assert prefix["mmvet_prefixkv"]["split"] == "official_218"
    assert prefix["llava_description"]["metrics"] == ["teacher_forced_ppl", "rouge_l_f1"]
    assert c["lanes"]["vl_cache"]["tasks"]["coco_caption"]["split"] == "val"
    assert c["lanes"]["lookm"]["tasks"]["milebench"]["dataset_revision"] == (
        "53c7a58051ef88bacf76541d91f03f5ba2d71e7d"
    )


def test_backing_storage_counts_preallocated_views_and_deduplicates_aliases():
    # Two disjoint views into one allocation: storage is once, not twice.
    slab = torch.zeros(2, 1, 1, 16, 4, dtype=torch.float16)
    layers = ((slab[0, :, :, :4], slab[1, :, :, :4]),)
    tracker = CacheLifecycle.start(layers)
    assert tracker.initial_payload_bytes == 64
    assert tracker.initial_storage_bytes == 256
    tracker.observe(((slab[0, :, :, :12], slab[1, :, :, :12]),))
    tracker.observe(layers)  # shrinking later must not erase peak
    result = tracker.record()
    assert result["final_active_kv_bytes"] == 64
    assert result["peak_active_kv_bytes"] == 192
    assert result["peak_active_kv_storage_bytes"] == 256


def test_dynamic_append_tracks_peak_and_initial_separately():
    def layers(n):
        return ((torch.zeros(1, 1, n, 4), torch.zeros(1, 1, n, 4)),)

    tracker = CacheLifecycle.start(layers(4))
    tracker.observe(layers(5))
    tracker.observe(layers(6))
    assert tracker.initial_storage_bytes == 128
    assert tracker.final_storage_bytes == tracker.peak_storage_bytes == 192
    assert tracker.observations == 3


def rows():
    return [
        dict(
            sample_id=sid,
            method=method,
            lane="vl_cache",
            task="coco_caption",
            metric_name="cider",
            status="completed",
            result_category="unified_comparison",
            control_hashes={k: "a" * 64 for k in CONTROL_FIELDS},
            physical_byte_cap=1000,
            initial_active_kv_storage_bytes=960,
            final_active_kv_storage_bytes=1120,
            peak_active_kv_storage_bytes=1120,
            largest_feasible_initial_bytes=960,
            capacity_certificate_sha256=digest(CERTIFICATE),
            decode_evictions=0,
            used_fullkv_fallback=False,
        )
        for sid in ["a", "b"]
        for method in ["full_kv", "prompt_attention_topk", "mosaickv_exact"]
    ]


def check(data):
    return validate_group(
        data,
        sample_ids=["a", "b"],
        methods=["full_kv", "prompt_attention_topk", "mosaickv_exact"],
        cap_bytes=1000,
        metric="cider",
        expected_controls={s: {k: "a" * 64 for k in CONTROL_FIELDS} for s in ["a", "b"]},
        capacity_certificates={
            (m, s): CERTIFICATE
            for m in ["prompt_attention_topk", "mosaickv_exact"]
            for s in ["a", "b"]
        },
    )


def test_fullkv_quality_only_and_feasible_shared_shortfall():
    data = rows()
    data[0]["initial_active_kv_storage_bytes"] = 10000
    result = check(data)
    assert result["latency_methods"] == ["prompt_attention_topk", "mosaickv_exact"]
    assert result["planned_samples"] == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "failed/historical"),
        ("result_category", "paper_reproduction", "failed/historical"),
        ("initial_active_kv_storage_bytes", 1001, "invalid append-only"),
        ("final_active_kv_storage_bytes", 1080, "realized bytes differ"),
        ("peak_active_kv_storage_bytes", None, "missing measured"),
        ("peak_active_kv_storage_bytes", 1152, "realized bytes differ"),
        ("largest_feasible_initial_bytes", 992, "underfilled"),
        ("capacity_certificate_sha256", None, "certificate"),
        ("used_fullkv_fallback", True, "fallback"),
        ("decode_evictions", 1, "append-only"),
        ("metric_name", "anls", "mix lanes"),
    ],
)
def test_reject_false_comparisons(field, value, message):
    data = rows()
    data[2][field] = value
    with pytest.raises(ValueError, match=message):
        check(data)


@pytest.mark.parametrize("control", CONTROL_FIELDS)
def test_every_input_control_is_required_and_identical(control):
    data = rows()
    del data[2]["control_hashes"][control]
    with pytest.raises(ValueError, match="missing controlled"):
        check(data)
    data[2]["control_hashes"][control] = "b" * 64
    with pytest.raises(ValueError, match="controls differ"):
        check(data)


def test_missing_duplicate_and_wrong_ids_never_disappear():
    data = rows()
    with pytest.raises(ValueError, match="missing rows"):
        check(data[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        check([*data, data[0]])
    data[0]["sample_id"] = "unexpected"
    with pytest.raises(ValueError, match="unexpected"):
        check(data)


def test_reuse_requires_full_identity_and_prior_comparison_audit():
    planned = rows()[1] | {"candidate_content_sha256": "c" * 64, "baseline_commit": "d" * 40}
    assert reuse_decision(None, planned) == "retry_missing"
    existing = deepcopy(planned)
    assert reuse_decision(existing, planned) == "preserve_unverified_requires_artifact_audit"
    existing["comparison_validation_passed"] = True
    assert reuse_decision(existing, planned) == "reuse_completed"
    existing["status"] = "failed"
    assert reuse_decision(existing, planned) == "retry_failed"
    existing["physical_byte_cap"] = 2000
    assert reuse_decision(existing, planned) == "preserve_incompatible_requires_new_plan"


def test_ledger_hash_identity_duplicates(tmp_path):
    import json

    from mosaickv.evaluation.shared_byte_contract import file_digest

    p = tmp_path / "ledger.json"
    route = dict(dataset_id="fixture", dataset_revision="a" * 40, split="test")
    p.write_text(json.dumps(dict(task="fixture", **route, sample_ids=["x", "y"])))
    assert validate_ledger(p, file_digest(p), route, "fixture") == ("x", "y")
    with pytest.raises(ValueError, match="hash"):
        validate_ledger(p, "b" * 64, route, "fixture")
    with pytest.raises(ValueError, match="route"):
        validate_ledger(p, file_digest(p), route, "other")


def test_missing_legacy_storage_is_not_fabricated_and_new_fields_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    from mosaickv.evaluation.model import GenerationMetrics
    from mosaickv.evaluation.results import EvaluationResult
    from mosaickv.evaluation.storage import write_parquet_aggregate

    row = EvaluationResult.from_generation(
        run_id="fixture",
        sample_id="a",
        task="synthetic_ci",
        model="synthetic/model",
        backend="synthetic",
        method="ci_fixture",
        retention_ratio=1.0,
        answer="blue",
        reference="blue",
        task_score=1.0,
        metrics=GenerationMetrics(
            initial_active_kv_storage_bytes=100,
            final_active_kv_storage_bytes=200,
            peak_active_kv_storage_bytes=300,
        ),
        manifest_path="synthetic-manifest.json",
    )
    path = tmp_path / "rows.parquet"
    write_parquet_aggregate((row,), path)
    stored = pq.read_table(path).to_pylist()[0]
    assert stored["initial_active_kv_storage_bytes"] == 100
    assert stored["final_active_kv_storage_bytes"] == 200
    assert stored["peak_active_kv_storage_bytes"] == 300
    old = row.to_json_object()
    for field in tuple(old):
        if field.endswith("kv_storage_bytes"):
            old.pop(field)
    legacy = EvaluationResult.from_json_object(old)
    assert legacy.initial_active_kv_storage_bytes is None
    assert legacy.final_active_kv_storage_bytes is None
    assert legacy.peak_active_kv_storage_bytes is None
