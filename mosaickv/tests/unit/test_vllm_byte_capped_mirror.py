"""CPU-only comparison/control checks for the mirror launcher."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
PREP = runpy.run_path(str(SCRIPTS / "prepare_vllm_byte_capped_mirror.py"))
RUN = runpy.run_path(str(SCRIPTS / "run_vllm_byte_capped_mirror.py"))
BUILD = runpy.run_path(str(SCRIPTS / "build_vllm_selector_plan.py"))


def test_identity_does_not_charge_unexecuted_compressed_source_planning():
    trace = {"timing_breakdown": {"forecast": 1.0, "graph": 2.0, "utility_and_selection": 3.0}}
    assert BUILD["_planning_seconds"](trace, identity=True) == 0.0
    assert BUILD["_planning_seconds"](trace) == 6.0


def test_shared_controls_include_physical_layout_and_sampling():
    source = {
        "model": {"revision": "a" * 40},
        "dataset": {"revision": "b" * 40},
        "execution": {"seed": 42},
        "generation": {"max_new_tokens": 64},
        "cache": dict.fromkeys(
            (
                "budget_unit",
                "budget_value",
                "block_size",
                "physical_budget_block_size",
                "physical_budget_mandatory_policy",
                "packing_mode",
                "exact_gather_mode",
            ),
            "fixed",
        ),
        "method": "mosaickv_exact",
    }
    baseline = copy.deepcopy(source)
    baseline["method"] = "prompt_attention_topk"
    assert PREP["controls"](source) == PREP["controls"](baseline)
    baseline["cache"]["physical_budget_mandatory_policy"] = "changed"
    assert PREP["controls"](source) != PREP["controls"](baseline)


def test_immutable_writer_refuses_overwrite_and_reader_rejects_tampering(tmp_path):
    path = tmp_path / "input.json"
    PREP["write_once"](path, {"value": 1})
    PREP["write_once"](path, {"value": 1})
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert RUN["bound"](str(path), expected) == path
    with pytest.raises(ValueError, match="conflict"):
        PREP["write_once"](path, {"value": 2})
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="changed"):
        RUN["bound"](str(path), expected)


@pytest.mark.parametrize("stage", ["preflight", "campaign"])
def test_failed_hf_completion_never_mirrored(tmp_path, stage):
    path = tmp_path / "completion.json"
    path.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="completed"):
        PREP["prepare"](
            path,
            tmp_path / "out",
            cap=40 * 2**20,
            sample_limit=2,
            stage=stage,
            source_root=tmp_path,
            attestation=None,
        )


def _fixture_outputs(tmp_path):
    """Synthetic artifact fixtures, never experimental measurements."""
    sample_ids = ["textvqa:a", "textvqa:b"]
    job = {
        "sample_ids": sample_ids, "source_git_sha": "a" * 40,
        "classification": "development_only", "stage": "preflight", "task": "textvqa",
        "physical_byte_cap": 32, "shared_controls": {"generation": {"max_new_tokens": 2}},
        "arms": [],
    }
    outputs = {}
    for name in ("full_kv", "native_identity", "prompt_attention_topk", "mosaickv_exact"):
        root = tmp_path / name
        run_id = name
        (root / run_id).mkdir(parents=True)
        trace_root = root / "traces" / run_id
        trace_root.mkdir(parents=True)
        config = {"method": name}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        rows = [
            {
                "sample_id": sid, "status": "completed", "answer": "different" if name ==
                "mosaickv_exact" else "answer", "reference": '["answer"]',
                "task_score": 0.1 if name == "mosaickv_exact" else 0.6,
                "ttft": 0.1, "prefill_time": 0.1, "compression_time": 0.1,
                "decode_time": 0.2, "end_to_end_time": 0.4, "active_kv_bytes": 32,
                "generated_tokens": 2,
            } for sid in sample_ids
        ]
        raw = root / run_id / "results.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = hashlib.sha256()
        for i, row in enumerate(rows):
            trace = {
                "sample_id": row["sample_id"], "status": "completed",
                "prompt_sha256": str(i), "media_sha256": str(i),
                "kv_cache_accounting": {"active_kv_bytes": 32},
                "trials": [{"token_ids": [1, 2], "request_latency_seconds": 0.3,
                            "native_audit": {"selection": {"retention_ratio": 0.5}}}],
            }
            if i == 0:
                trace["request_isolation"] = {
                    "performed": True, "token_ids_match_anchor": True,
                    "intervening_distinct_input_fingerprints": 1,
                    "anchor_token_ids": [1, 2], "probe_token_ids": [1, 2],
                }
            path = trace_root / f"{i}.json"
            path.write_text(json.dumps(trace))
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        manifest = {
            "resolved_config": config,
            "source": {"git_sha": "a" * 40, "git_dirty": True, "canonical_eligible": False},
            "artifacts": {"raw_output_sha": RUN["sha"](raw), "log_sha": digest.hexdigest()},
            "execution": {"request_isolation_verified": True},
        }
        (root / run_id / "manifest.json").write_text(json.dumps(manifest))
        hf = root / "hf"
        hf.mkdir()
        hf_rows = [dict(row, answer="answer", task_score=0.6) for row in rows]
        hf_raw = hf / "results.jsonl"
        hf_raw.write_text("".join(json.dumps(row) + "\n" for row in hf_rows))
        hf_manifest = hf / "manifest.json"
        hf_manifest.write_text(json.dumps({"artifacts": {"raw_output_sha": RUN["sha"](hf_raw)}}))
        entries = []
        for i, sid in enumerate(sample_ids):
            source_trace = hf / f"{i}.trace.json"
            source_trace.write_text(json.dumps({"sample_id": sid, "generated_token_ids": [1, 2]}))
            entries.append({
                "sample_id": sid, "bytes_per_position": 1,
                "projection": {"block_size": 16, "selected_blocks": [0, 1],
                               "retention_ratio": 0.5},
                "source_trace": str(source_trace), "source_trace_sha256": RUN["sha"](source_trace),
            })
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps({"entries": entries}))
        job["arms"].append({
            "name": name, "config": str(config_path), "config_sha256": RUN["sha"](config_path),
            "catalog": str(catalog), "catalog_sha256": RUN["sha"](catalog),
            "hf_manifest": str(hf_manifest), "hf_manifest_sha256": RUN["sha"](hf_manifest),
        })
        outputs[name] = (root, run_id)
    return job, outputs


def test_execution_pass_does_not_claim_quality_pass(tmp_path):
    job, outputs = _fixture_outputs(tmp_path)
    result = RUN["validate_outputs"](job, outputs)
    assert result["status"] == "passed"
    assert result["telemetry_checks_passed"]
    assert "not_quality" in result["status_scope"]
    quality = result["quality_comparisons"]["prompt_attention_topk"]
    assert quality["mosaickv_minus_baseline_score"] == -0.5
    assert not quality["observed_mean_no_worse"]
    assert not quality["statistical_noninferiority_established"]
    assert not result["full_campaign_authorized_by_this_report"]
    assert result["hf_comparisons"]["mosaickv_exact"]["exact_answer_agreement"] == 0
    assert "full_kv" not in result["compressed_timing"]


def test_audit_ratio_mismatch_is_reported_without_hiding_measured_scores(tmp_path):
    job, outputs = _fixture_outputs(tmp_path)
    arm = next(a for a in job["arms"] if a["name"] == "mosaickv_exact")
    catalog = Path(arm["catalog"])
    payload = json.loads(catalog.read_text())
    payload["entries"][0]["projection"]["retention_ratio"] = 0.25
    catalog.write_text(json.dumps(payload))
    arm["catalog_sha256"] = RUN["sha"](catalog)
    result = RUN["validate_outputs"](job, outputs)
    assert not result["telemetry_checks_passed"]
    assert len(result["telemetry_issues"]) == 1
    assert result["telemetry_issues"][0]["audit_ratio"] == 0.5
    assert result["mean_task_scores"]["mosaickv_exact"] == 0.1


def test_validator_rejects_tampered_trace_even_when_manifest_claims_isolation(tmp_path):
    job, outputs = _fixture_outputs(tmp_path)
    root, run_id = outputs["mosaickv_exact"]
    path = root / "traces" / run_id / "0.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="trace artifact hash"):
        RUN["validate_outputs"](job, outputs)


@pytest.mark.parametrize("defect", ["isolation", "token_count", "bytes", "duplicate_trace"])
def test_validator_checks_evidence_not_only_boolean_assertions(tmp_path, defect):
    job, outputs = _fixture_outputs(tmp_path)
    root, run_id = outputs["mosaickv_exact"]
    trace_root = root / "traces" / run_id
    path = trace_root / "0.json"
    trace = json.loads(path.read_text())
    if defect == "isolation":
        trace["request_isolation"]["probe_token_ids"] = [9, 9]
    elif defect == "token_count":
        trace["trials"][0]["token_ids"] = []
        trace["request_isolation"]["anchor_token_ids"] = []
        trace["request_isolation"]["probe_token_ids"] = []
    elif defect == "bytes":
        trace["kv_cache_accounting"]["active_kv_bytes"] = 64
    else:
        (trace_root / "duplicate.json").write_text(path.read_text())
    path.write_text(json.dumps(trace))
    digest = hashlib.sha256()
    for item in sorted(trace_root.glob("*.json")):
        digest.update(item.name.encode())
        digest.update(item.read_bytes())
    manifest_path = root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["log_sha"] = digest.hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        RUN["validate_outputs"](job, outputs)


def test_canonical_classification_rejects_dirty_output_manifest(tmp_path):
    job, outputs = _fixture_outputs(tmp_path)
    job["classification"] = "canonical_vllm_projection"
    with pytest.raises(ValueError, match="noncanonical"):
        RUN["validate_outputs"](job, outputs)


def test_validator_rejects_mismatched_executed_config(tmp_path):
    job, outputs = _fixture_outputs(tmp_path)
    root, run_id = outputs["mosaickv_exact"]
    path = root / run_id / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["resolved_config"]["method"] = "another"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="executed configuration"):
        RUN["validate_outputs"](job, outputs)
