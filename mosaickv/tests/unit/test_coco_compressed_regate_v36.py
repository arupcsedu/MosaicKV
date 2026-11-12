"""Synthetic CPU orchestration fixtures; no real model or quality evidence."""

import json
from copy import deepcopy

import evicastkv_coco_compressed_regate_v36 as regate
import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return regate.sha(path)


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    ledger = tmp_path / "existing-ledger.json"
    write(ledger, dict(metadata_only_fixture=True))
    monkeypatch.setattr(regate, "PARENT", parent)
    monkeypatch.setattr(regate, "LEDGER", ledger)
    monkeypatch.setattr(regate, "LEDGER_SHA", regate.sha(ledger))
    lock = dict(
        cohorts={
            "coco": {
                "validation": dict(
                    path=str(ledger), sha256=regate.sha(ledger), sample_ids=list(regate.IDS)
                )
            }
        },
        arithmetic=dict(allow_fp16_reduced_precision_reduction=True),
        tolerances=dict(retention_one_max_logit=1e-4),
        budgets_mib=dict(coco=[20, 40]),
        seed=20271014,
    )
    write(parent / "protocol_lock.json", lock)
    monkeypatch.setattr(regate, "PARENT_LOCK_SHA", regate.sha(parent / "protocol_lock.json"))
    monkeypatch.setattr(regate, "clean_source", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        regate, "cpu_gate", lambda *_args: dict(status="passed", source_sha="a" * 40)
    )
    monkeypatch.setattr(regate, "environment", lambda: dict(synthetic=True))
    monkeypatch.setattr(regate, "load_lock", lambda root: regate.read(root / "protocol_lock.json"))
    cpu_root = tmp_path / "cpu"
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        write(cpu_root / "full-cpu" / name, dict(synthetic=True))
    root = tmp_path / "regate"
    result = regate.prepare(root, cpu_root)
    assert result == dict(status="prepared_not_submitted", jobs=2, requests=12, jobs_submitted=0)
    return root


def parity():
    item = dict(
        generated_tokens=16,
        token_agreement=1.0,
        maximum_logit_difference=0.0,
        reference_token_ids=list(range(16)),
        candidate_token_ids=list(range(16)),
    )
    return {
        name: deepcopy(item)
        for name in ("ordinary_generate", "retention_one", "mosaickv_runtime_retention_one")
    }


def representation():
    return dict(
        status="passed",
        forced_steps=16,
        compact_absolute_tolerance=1e-4,
        retention_one_absolute_tolerance_unchanged=1e-4,
        original_layout_normwise_tolerance=8 * 2**-10,
        source_entries_bitwise=True,
        independent_mask_builder=True,
        unpruned_control_passed=True,
        independent_compact_passed=True,
        original_masked_normwise_passed=True,
        per_step=[
            dict(
                step=i,
                compact_max_absolute=0.0,
                masked_normalized_linf=0.0,
                compact_greedy_equal=True,
            )
            for i in range(16)
        ],
        unpruned_padding_control=[dict(normalized_linf=0.0) for _ in range(16)],
    )


def completed(root):
    _plan, lock = regate.load(root)
    for index, sid in enumerate(regate.IDS):
        output = root / "outputs/correctness" / f"{index:03d}" / "synthetic-attempt"
        manifest = output / "environment.json"
        write(manifest, dict(source_sha="a" * 40, synthetic=True))
        write(output / "retention_one.json", parity())
        records = []
        for method, cap, variant in regate.ARMS:
            key = f"{index:03d}-{method}-{cap}-complete-r0"
            config = regate.resolved_config(lock, "coco", method, cap)
            mapping = regate.canonical_config(config)
            resolved = output / "resolved-configs" / f"{key}.json"
            trace = output / "traces" / f"{key}.json"
            write(resolved, mapping)
            write(
                trace, dict(status="completed", synthetic=True, generated_token_ids=list(range(64)))
            )
            write(output / "correctness" / f"{key}.json", representation())
            row = dict(
                run_id=key,
                sample_id=sid,
                method=method,
                effective_method=method,
                budget_bytes=cap * 2**20,
                variant=variant,
                source_sha="a" * 40,
                runtime_config_sha256=regate.config_sha256(config),
                config_sha=regate.digest(
                    dict(runtime_config=mapping, arithmetic=lock["arithmetic"])
                ),
                resolved_config_path=str(resolved),
                resolved_config_file_sha256=regate.sha(resolved),
                status="completed",
                workload="coco",
                input_sha=sid,
                processor_sha="p",
                scorer_sha="s",
                forward_steps=63,
                cache_append_steps=63,
                emitted_tokens=64,
                repeat=0,
                initial_kv_bytes=cap * 2**20,
                final_kv_bytes=cap * 2**20 + 63 * 131072,
                peak_kv_bytes=cap * 2**20 + 63 * 131072,
                e2e_wall_seconds=1.0,
                boundary="A",
                manifest_path=str(manifest),
                manifest_sha256=regate.sha(manifest),
                trace_path=str(trace),
                trace_sha256=regate.sha(trace),
                timings=dict(per_token_forward_wall=[0.01] * 63),
                metrics=dict(
                    initial_active_kv_storage_bytes=cap * 2**20,
                    final_active_kv_storage_bytes=cap * 2**20 + 63 * 131072,
                    peak_active_kv_storage_bytes=cap * 2**20 + 63 * 131072,
                ),
            )
            path = output / "records" / f"{key}.json"
            write(path, row)
            records.append(dict(path=str(path), sha256=regate.sha(path), status="completed"))
        write(
            output / "worker.json",
            dict(source_sha="a" * 40, status="passed", errors=[], records=records),
        )


def test_prepare_uses_only_two_consumed_ids_and_native_exact_controls(prepared):
    plan, lock = regate.load(prepared)
    assert plan["policy"]["consumed_ids"] == list(regate.IDS)
    assert lock["authorization"]["automatic_stage_progression"] is False
    assert lock["arithmetic"] == dict(allow_fp16_reduced_precision_reduction=True)
    assert lock["templates"]["coco"]["sha256"] == regate.admission.CANDIDATE_SHA
    assert all(
        len(job["arms"]) == 6 for job in regate.read(prepared / "jobs/correctness.json")["jobs"]
    )
    assert not (prepared / "jobs/regate-gpu-submitted.json").exists()


def test_control_drift_cannot_be_submitted(prepared):
    plan = regate.read(prepared / "regate_plan.json")
    plan["policy"]["maximum_absolute_logit_error"] = 0.01
    write(prepared / "regate_plan.json", plan)
    with pytest.raises(ValueError, match="policy/configuration changed"):
        regate.load(prepared)


def test_incomplete_workers_write_failure_not_proofs(prepared):
    result = regate.analyze(prepared)
    assert result["complete"] is False
    assert result["actual_requests"] == 0
    assert result["failures"]
    assert not (
        prepared / "analysis/v36-compressed-regate/qualified_correctness_proof.json"
    ).exists()


def test_all_three_executed_methods_emit_actual_schedule_and_byte_proofs(prepared):
    completed(prepared)
    result = regate.analyze(prepared)
    assert result["complete"] is True
    assert result["actual_requests"] == 12 and len(result["per_sample_storage_pairing"]) == 4
    output = prepared / "analysis/v36-compressed-regate"
    proof = regate.read(output / "qualified_correctness_proof.json")
    assert proof["kind"] == "correctness" and proof["complete"] is True
    assert proof["model_scope"] == "FP16_GQA_COCO_exact_selection"
    byte = regate.read(output / "qualified_byte_matching_proof.json")
    assert byte["actual_compared_methods_pass"] is True
    assert byte["complete"] is True and byte["baseline_fidelity_not_inferred"] is True
    assert byte["compared_methods"] == [
        "prompt_attention_topk",
        "vl_cache_reimpl",
        "mosaickv_exact",
    ]
    assert result["benchmark_quality_tested"] is result["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "fallback",
        "schedule",
        "config",
        "representation",
        "vl_cache",
        "token_trace",
        "forward_timings",
        "physical_metric",
    ],
)
def test_one_invalid_executed_arm_blocks_all_qualified_proofs(prepared, kind):
    completed(prepared)
    worker_path = next((prepared / "outputs/correctness/000").glob("*/worker.json"))
    worker = regate.read(worker_path)
    binding = worker["records"][1 if kind == "vl_cache" else 0]
    row_path = regate.Path(binding["path"])
    row = regate.read(row_path)
    if kind in {"fallback", "vl_cache"}:
        row["effective_method"] += "__fallback"
    elif kind == "schedule":
        row["forward_steps"] = 62
    elif kind == "config":
        row["config_sha"] = "0" * 64
    elif kind == "representation":
        path = worker_path.parent / "correctness" / f"{row['run_id']}.json"
        record = regate.read(path)
        record["per_step"][0]["compact_max_absolute"] = 0.01
        write(path, record)
    elif kind == "token_trace":
        path = regate.Path(row["trace_path"])
        trace = regate.read(path)
        trace["generated_token_ids"].pop()
        write(path, trace)
        row["trace_sha256"] = regate.sha(path)
    elif kind == "forward_timings":
        row["timings"]["per_token_forward_wall"].pop()
    else:
        row["metrics"]["peak_active_kv_storage_bytes"] += 1
    write(row_path, row)
    binding["sha256"] = regate.sha(row_path)
    write(worker_path, worker)
    result = regate.analyze(prepared)
    assert result["complete"] is False
    assert not (
        prepared / "analysis/v36-compressed-regate/qualified_correctness_proof.json"
    ).exists()


@pytest.mark.parametrize(
    "field,value",
    [("generated_tokens", 15), ("maximum_logit_difference", 0.001), ("token_agreement", 0.99)],
)
def test_retention_one_unchanged_gate(field, value):
    record = parity()
    record["retention_one"][field] = value
    with pytest.raises(ValueError):
        regate.check_parity(record)


def test_submit_builds_afterany_chain_and_preserves_gpu_receipt(prepared, monkeypatch):
    calls = []

    def dispatch(command, **_kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return "gpu:a100:4\n"
        return "12345\n" if "--array=0-1%2" in command else "12346\n"

    monkeypatch.setattr(regate.subprocess, "check_output", dispatch)
    result = regate.submit(prepared)
    assert result == dict(gpu_array="12345", analysis="12346")
    assert "--dependency=afterany:12345" in calls[2]
    assert "--array=0-1%2" in calls[1]
    assert (prepared / "jobs/regate-gpu-submitted.json").exists()


def test_no_gpu_submission_over_aggregate_ceiling(prepared, monkeypatch):
    monkeypatch.setattr(regate.subprocess, "check_output", lambda *_a, **_k: "gpu:a100:6\n")
    with pytest.raises(ValueError, match="six-GPU ceiling"):
        regate.submit(prepared)
    assert not (prepared / "jobs/regate-gpu-request.json").exists()


def test_shared_engineering_qualification_requires_actual_completed_regate(prepared):
    result = regate.qualify_shared_engineering(prepared)
    assert result["complete"] is False
    assert result["failures"]
    output = prepared / "analysis/v36-compressed-regate"
    assert not (output / "qualified_metric_proof.json").exists()
    assert not (output / "qualified_baseline_fidelity_proof.json").exists()


def test_shared_engineering_success_records_old_producer_not_new_gpu_execution(
    prepared, monkeypatch
):
    completed(prepared)
    monkeypatch.setattr(regate, "HISTORICAL_PRODUCER", "b" * 40)
    monkeypatch.setattr(
        regate,
        "shared_engineering_evidence",
        lambda _root, _plan: dict(synthetic_metadata_fixture=True),
    )
    regate.analyze(prepared)
    output = prepared / "analysis/v36-compressed-regate"
    for name in ("qualified_metric_proof.json", "qualified_baseline_fidelity_proof.json"):
        proof = regate.read(output / name)
        assert proof["complete"] is True
        assert proof["actual_historical_GPU_producer"] == "b" * 40
        assert proof["actual_current_cache_regate_producer"] == "a" * 40
        assert proof["proof_origin"] == "derived_validation_not_new_historical_GPU_execution"
        assert proof["official_reproduction_validated"] is False
        assert proof["published_point_reproduced"] is False
        assert proof["strong_official_baseline_superiority_claim_allowed"] is False


@pytest.mark.parametrize(
    "failure",
    ["environment differs", "source byte closure differs", "native metric gate incomplete"],
)
def test_shared_qualification_preserves_failures_and_does_not_manufacture_proofs(
    prepared, monkeypatch, failure
):
    completed(prepared)

    def blocked(*_args):
        raise ValueError(failure)

    monkeypatch.setattr(regate, "shared_engineering_evidence", blocked)
    regate.analyze(prepared)
    output = prepared / "analysis/v36-compressed-regate"
    result = regate.read(output / "shared_engineering_qualification.json")
    assert result["complete"] is False
    assert failure in result["failures"][0]["error"]
    assert not (output / "qualified_metric_proof.json").exists()
    assert not (output / "qualified_baseline_fidelity_proof.json").exists()


def test_prepare_optionally_hash_binds_existing_original_review(prepared, monkeypatch):
    review_root = prepared.parent / "original-review"
    write(review_root / "original_complete_review.json", dict(source_sha="a" * 40))
    monkeypatch.setattr(
        regate.original_review, "load_review", lambda _root: (dict(source_sha="a" * 40), None)
    )
    root = prepared.parent / "linked-regate"
    regate.prepare(root, prepared.parent / "cpu", review_root)
    plan, _lock = regate.load(root)
    assert plan["original_review_root"] == str(review_root.resolve())
    write(review_root / "original_complete_review.json", dict(source_sha="b" * 40))
    with pytest.raises(ValueError, match="binding changed"):
        regate.load(root)


def test_linked_review_freezes_only_after_actual_regate_and_four_proofs(prepared, monkeypatch):
    completed(prepared)
    actual_load = regate.load
    review_root = prepared.parent / "mock-original-review"

    def linked_load(root):
        plan, lock = actual_load(root)
        plan["original_review_root"] = str(review_root)
        return plan, lock

    monkeypatch.setattr(regate, "load", linked_load)
    monkeypatch.setattr(regate, "shared_engineering_evidence", lambda *_args: dict(synthetic=True))
    calls = []

    def freeze(root, paths):
        calls.append((root, paths))
        assert all(path.is_file() for path in paths.values())
        return dict(candidate_parameters_frozen=True)

    monkeypatch.setattr(regate.original_review, "freeze_candidate", freeze)
    regate.analyze(prepared)
    assert len(calls) == 1
    assert calls[0][0] == str(review_root)
    assert set(calls[0][1]) == {"correctness", "metric", "byte_matching", "baseline_fidelity"}
    result = regate.read(
        prepared / "analysis/v36-compressed-regate/shared_engineering_qualification.json"
    )
    assert result["candidate_parameters_frozen"] is True
    assert result["fresh_confirmation_authorized"] is False
