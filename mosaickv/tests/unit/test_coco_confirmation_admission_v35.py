"""Synthetic metadata fixtures only; never measured confirmation evidence."""

import json
from copy import deepcopy

import evicastkv_coco_confirmation_admission_v35 as admission
import pytest


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return dict(path=str(path), sha256=admission.file_sha(path))


def candidate():
    return dict(
        model=dict(
            id=admission.CONTROLS["model_id"],
            revision=admission.CONTROLS["model_revision"],
            precision="fp16",
        ),
        dataset=dict(
            id=admission.CONTROLS["dataset_id"],
            revision=admission.CONTROLS["dataset_revision"],
            split="val",
        ),
        execution=dict(attention_implementation="eager", backend="huggingface"),
        method="mosaickv_exact",
        cache=dict(block_size=1),
        generation=dict(max_new_tokens=64),
    )


def setup_metadata(monkeypatch, tmp_path):
    config = tmp_path / "candidate.json"
    write(config, candidate())
    expected = admission.file_sha(config)
    monkeypatch.setattr(admission, "CANDIDATE", config)
    monkeypatch.setattr(admission, "REVIEW_TEMPLATE", config)
    monkeypatch.setattr(admission, "CANDIDATE_SHA", expected)
    negative = dict(
        status="passed_execution_development_only",
        source_sha="b" * 40,
        confirmation_authorized=False,
        quality_decisions={
            name: dict(required_contrasts=4, positive_development_signal=False)
            for name in ("evidence-routing-v1", "evidence-coverage-v1", "evidence-both-v1")
        },
    )
    gate = tmp_path / "negative.json"
    write(gate, negative)
    monkeypatch.setattr(admission, "SELECTION_GATES", ((gate, admission.file_sha(gate)),))
    monkeypatch.setattr(admission, "inspect", lambda: dict(checked_paths=[], discovered_paths=[]))
    return expected


def frozen_fixture(monkeypatch, tmp_path):
    expected = setup_metadata(monkeypatch, tmp_path)
    history = tmp_path / "history.json"
    media = tmp_path / "media.json"
    write(
        history,
        dict(
            complete=True,
            errors=[],
            global_historical_inventory_complete=True,
            complete_historical_id_inventory=True,
            frontier_exhausted=True,
            missing_original_worker_indices=[],
        ),
    )
    write(
        media,
        dict(
            complete=True,
            errors=[],
            global_historical_media_inventory_complete=True,
            underlying_media_source_group_inventory_complete=True,
            unresolved_source_groups=[],
        ),
    )
    frozen = dict(
        schema=admission.FROZEN_SCHEMA,
        frozen=True,
        parameters_locked_before_confirmation=True,
        confirmation_outcomes_used_for_selection=False,
        source_sha="a" * 40,
        candidate_config_sha256=expected,
        controls=deepcopy(admission.CONTROLS),
        history_report_sha256=admission.file_sha(history),
        media_report_sha256=admission.file_sha(media),
    )
    common = dict(
        complete=True,
        failures=[],
        source_sha=frozen["source_sha"],
        candidate_config_sha256=expected,
        controls=deepcopy(admission.CONTROLS),
    )
    records = dict(
        selection_gate=common | dict(positive_development_signal=True),
        numerical_gate=common
        | dict(
            source_scope="FP16_GQA_COCO_compressed_exact_selection",
            full_retention_identity_pass=True,
            retained_cache_fidelity_pass=True,
            tolerance=1e-4,
        ),
        byte_matching_gate=common
        | dict(
            per_sample_initial_byte_match_pass=True,
            per_sample_final_byte_match_pass=True,
            per_sample_peak_byte_match_pass=True,
            physical_backing_padding_preallocation_charged=True,
            matched_forward_step_schedule=True,
        ),
        baseline_fidelity_gate=common
        | dict(method="vl_cache_reimpl", qualified_reimplementation=True, fidelity_validated=True),
        cohort_disjointness_gate=dict(
            complete=True,
            failures=[],
            metadata_only=True,
            candidate_count=256,
            candidate_config_sha256=expected,
            id_disjointness_pass=True,
            underlying_media_source_group_disjointness_pass=True,
            history_report_sha256=frozen["history_report_sha256"],
            media_report_sha256=frozen["media_report_sha256"],
        ),
    )
    for key, record in records.items():
        frozen[key] = write(tmp_path / f"{key}.json", record)
    path = tmp_path / "selected_config.json"
    write(path, frozen)
    return path, frozen, records, history, media


def test_absent_freeze_and_coverage_block_without_erasing_negative_evidence(monkeypatch, tmp_path):
    setup_metadata(monkeypatch, tmp_path)
    report = admission.evaluate()
    assert report["status"] == "blocked_confirmation_prerequisites"
    assert report["verdicts"]["registered_candidate_unchanged"] is True
    assert report["verdicts"]["pinned_negative_selection_evidence_preserved"] is True
    assert report["verdicts"]["explicit_frozen_candidate_valid"] is False
    assert report["facts"]["historical_selection"][0]["confirmation_authorized"] is False
    assert report["jobs_submitted"] == 0


def test_all_passed_synthetic_contract_still_never_launches_or_authorizes(monkeypatch, tmp_path):
    path, _frozen, _records, history, media = frozen_fixture(monkeypatch, tmp_path)
    report = admission.evaluate(history_report=history, media_report=media, freeze_artifact=path)
    assert all(report["verdicts"].values())
    assert report["failures"] == []
    assert report["status"] == "prerequisites_satisfied_not_execution_authorization"
    assert report["new_sample_ids_accessed"] is False
    assert report["dataset_or_prediction_content_accessed"] is False
    assert report["fresh_confirmation_authorized"] is False
    assert report["materialization_authorized"] is False
    assert report["paper_claim_allowed"] is False
    assert report["GPU_jobs_submitted"] == report["jobs_submitted"] == 0
    assert len(report["evidence_bindings"]) == 10


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("selection_gate", "positive_development_signal", False),
        ("selection_gate", "positive_development_signal", 1),
        ("numerical_gate", "source_sha", "c" * 40),
        ("numerical_gate", "complete", False),
        ("numerical_gate", "tolerance", 0.001),
        ("numerical_gate", "source_scope", "FP16_GQA_COCO_FullKV_evaluation_only_boundary_control"),
        ("numerical_gate", "source_scope", "BF16_LLaVA_1_5_full_MHA_source_reader"),
        ("byte_matching_gate", "per_sample_final_byte_match_pass", False),
        ("byte_matching_gate", "per_sample_peak_byte_match_pass", False),
        ("byte_matching_gate", "matched_forward_step_schedule", False),
        ("baseline_fidelity_gate", "qualified_reimplementation", False),
        ("cohort_disjointness_gate", "candidate_count", 255),
        ("cohort_disjointness_gate", "underlying_media_source_group_disjointness_pass", False),
        ("cohort_disjointness_gate", "history_report_sha256", "0" * 64),
    ],
)
def test_negative_wrong_source_scope_or_incomplete_proofs_reject(
    monkeypatch, tmp_path, key, field, value
):
    _path, frozen, records, _history, _media = frozen_fixture(monkeypatch, tmp_path)
    records[key][field] = value
    frozen[key] = write(tmp_path / f"{key}.json", records[key])
    with pytest.raises(ValueError):
        admission.validate_freeze(frozen, admission.MetadataReader())


def test_checkpoint_or_precision_mismatch_rejects_even_positive_label(monkeypatch, tmp_path):
    _path, frozen, records, _history, _media = frozen_fixture(monkeypatch, tmp_path)
    records["numerical_gate"]["controls"]["precision"] = "bf16"
    frozen["numerical_gate"] = write(tmp_path / "numerical_gate.json", records["numerical_gate"])
    with pytest.raises(ValueError, match="model/source-matched"):
        admission.validate_freeze(frozen, admission.MetadataReader())


def test_changed_link_is_rejected_by_hash_before_booleans(monkeypatch, tmp_path):
    _path, frozen, records, _history, _media = frozen_fixture(monkeypatch, tmp_path)
    records["selection_gate"]["positive_development_signal"] = False
    write(tmp_path / "selection_gate.json", records["selection_gate"])
    with pytest.raises(ValueError, match="hash mismatch"):
        admission.validate_freeze(frozen, admission.MetadataReader())


@pytest.mark.parametrize("kind", ["history", "media"])
def test_subset_completion_and_saved_hash_reduction_do_not_prove_freshness(kind):
    assert (
        admission.coverage(
            dict(
                complete=True,
                errors=[],
                subset_frontier_exhausted=True,
                complete_saved_hash_reduction=True,
                global_historical_inventory_complete=False,
            ),
            kind,
        )
        is False
    )


def test_history_and_media_reports_are_bound_by_freeze(monkeypatch, tmp_path):
    path, _frozen, _records, history, media = frozen_fixture(monkeypatch, tmp_path)
    record = json.loads(history.read_text())
    record["extra_metadata"] = "changed"
    write(history, record)
    report = admission.evaluate(history_report=history, media_report=media, freeze_artifact=path)
    assert report["verdicts"]["explicit_frozen_candidate_valid"] is False
    assert "both supplied" in report["failures"][-1]["error"]


def test_known_root_inspection_never_reads_or_selects_discovered_freeze(tmp_path):
    path = tmp_path / "analysis/selected_config.json"
    path.parent.mkdir()
    path.write_text("not parsed and not automatically admitted")
    result = admission.inspect([tmp_path])
    assert result["discovered_paths"] == [str(path)]
    assert len(result["checked_paths"]) == 12


def test_write_once_report_and_binding_sidecar(monkeypatch, tmp_path):
    setup_metadata(monkeypatch, tmp_path)
    report = admission.evaluate()
    output = tmp_path / "report.json"
    admission.write_report(output, report)
    sidecar = json.loads((tmp_path / "report.json.bindings.json").read_text())
    assert sidecar["artifact_bindings"] == report["evidence_bindings"]
    assert sidecar["report_sha256"] == admission.file_sha(output)
    with pytest.raises(FileExistsError):
        admission.write_report(output, report)
    output.unlink()
    with pytest.raises(FileExistsError):
        admission.write_report(output, report)


def test_reader_rejects_nonobject_metadata(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        admission.MetadataReader().read(path)
