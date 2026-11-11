"""Synthetic fixed-receipt import; data bodies deliberately do not exist."""

import copy
import json
from pathlib import Path

import evicastkv_provenance_import as importer
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return importer.sha(path)


@pytest.fixture
def parent(tmp_path, monkeypatch):
    base, parent = tmp_path / "old-run", tmp_path / "old-run/provenance"
    code = tmp_path / "source.py"
    code.write_text("# recorded immutable source\n")
    producer = dict(recorded_producer_commit=importer.PRODUCER, historical_source_executed=False)
    queue, proofs, workers = [], [], []
    for index in range(2):
        path = str(tmp_path / f"NEVER_READ_{index}.parquet")
        artifact_sha = str(index) * 64
        row = dict(
            path=path,
            index=index,
            signature=[1, 2, 12, 4],
            original_error_sha256s=[chr(97 + index) * 64],
            expected_artifact_sha256=artifact_sha,
        )
        proof = dict(
            path=path,
            signature=row["signature"],
            original_error_sha256s=row["original_error_sha256s"],
            whole_file_sha256_before=artifact_sha,
            whole_file_sha256_after=artifact_sha,
            source_proof=producer,
            resolved_as="derived_no_request_or_media_identity",
            parquet_footer=dict(rows_decoded=0),
            bindings={str(code): importer.sha(code)},
        )
        queue.append(row)
        proofs.append(proof)
    imported = [dict(original_error_sha256s=["c" * 64], retained_old_proof=True)]
    plan = dict(
        source_sha=importer.PARENT_SOURCE,
        cpu_gate_root=str(base),
        shard_count=2,
        queue=queue,
        imported_proofs=imported,
        producer=producer,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        bindings={str(code): importer.sha(code)},
        covered_error_hashes=["c" * 64],
        original_errors_path="immutable_parent_errors",
    )
    psha = put(parent / "plan.json", plan)
    for index in range(2):
        value = dict(
            source_sha=importer.PARENT_SOURCE,
            index=index,
            plan_sha256=psha,
            committed=[proofs[index]],
            pending=[],
            blocked=[],
            fresh_confirmation_authorized=False,
        )
        workers.append(put(parent / "workers" / f"shard-{index}.json", value))
    overlay = dict(
        source_sha=importer.PARENT_SOURCE,
        imported_proofs=imported,
        new_proofs=proofs,
        parent_bindings=plan["bindings"],
        original_errors_path=plan["original_errors_path"],
    )
    osha = put(parent / "coverage_overlay.json", overlay)
    net = dict(
        original_error_count=6,
        unique_covered_before=1,
        newly_covered_error_hashes=2,
        net_unresolved_after=3,
        net_unresolved_categories=dict(remaining=3),
        pending_frontier_items=9002,
        producer_recipe_subset_of_yaml=0,
        scientific_scope="provenance_coverage_only_not_identity_or_media_freshness",
    )
    nsha = put(parent / "net_error_report.json", net)
    gate = dict(
        **net,
        source_sha=importer.PARENT_SOURCE,
        status="native_v4_provenance_recovery_complete",
        registered_files=2,
        newly_proved_files=2,
        previously_proved_files=1,
        pending_files=0,
        blocked_files=0,
        missing_files=0,
        missing_shards=[],
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        error_waivers=0,
        original_errors_preserved=True,
        external_video_journals_unresolved=True,
        roots_restarted=False,
        coverage_overlay_sha256=osha,
        net_error_report_sha256=nsha,
        worker_bindings={str(parent / "workers" / f"shard-{i}.json"): workers[i] for i in range(2)},
    )
    put(parent / importer.GATE, gate)
    cpu_suite_sha = put(base / "full-cpu/cpu-tests.json", dict(source_sha=importer.PARENT_SOURCE))
    (base / "full-cpu/cpu-tests.log").write_text("complete synthetic CPU suite\n")
    (base / "full-cpu/cpu-tests.xml").write_text(
        '<testsuite tests="3" failures="0" errors="0" skipped="0"/>'
    )
    cpu = dict(
        status="passed",
        source_sha=importer.PARENT_SOURCE,
        counts=dict(tests=3, failures=0, errors=0, skipped=0),
        cpu_suite_sha256=cpu_suite_sha,
    )
    csha = put(base / "GATE_cpu_suite.json", cpu)
    monkeypatch.setattr(importer, "PARENT", parent)
    monkeypatch.setattr(importer, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(importer, "PARENT_SOURCE_ROOT", tmp_path / "old-code")
    monkeypatch.setattr(
        importer,
        "EXPECTED",
        dict(
            registered=2,
            imported=1,
            raw=6,
            before=1,
            net=3,
            frontier=9002,
            recipe_subset=0,
            cpu_tests=3,
        ),
    )
    monkeypatch.setattr(
        importer,
        "EXPECTED_HASHES",
        {name: importer.sha(parent / name) for name in importer.EXPECTED_HASHES},
    )
    monkeypatch.setattr(importer, "PARENT_CPU_GATE_SHA", csha)
    monkeypatch.setattr(importer, "PARENT_CPU_SUITE_SHA", cpu_suite_sha)
    monkeypatch.setattr(importer, "clean_source", lambda root, expected=None: expected or "e" * 40)
    monkeypatch.setattr(
        importer,
        "cpu_gate",
        lambda root, revision: (
            copy.deepcopy(cpu) if Path(root) == base else dict(status="passed", source_sha=revision)
        ),
    )
    return parent


def rebind(parent, name, value):
    actual = put(parent / name, value)
    importer.EXPECTED_HASHES[name] = actual
    return actual


def test_byte_exact_gate_preserves_original_source_without_any_data_body(parent, tmp_path):
    root = tmp_path / "new-run/provenance"
    gate_before = (parent / importer.GATE).read_bytes()
    result = importer.import_completed(root, parent)
    assert result["original_source_sha"] == importer.PARENT_SOURCE
    assert result["reader_source_sha"] == "e" * 40
    assert (root / importer.GATE).read_bytes() == gate_before
    copied = json.loads((root / importer.GATE).read_bytes())
    assert copied["source_sha"] == importer.PARENT_SOURCE
    reuse = json.loads((root / "reuse.json").read_bytes())
    assert reuse["new_proof_executions"] == reuse["parquet_body_reads"] == reuse["new_scans"] == 0
    assert not reuse["old_producer_relabelled"] and not reuse["fresh_confirmation_authorized"]
    assert len(reuse["parent_bindings"]) == 11
    assert not (tmp_path / "NEVER_READ_0.parquet").exists()


@pytest.mark.parametrize("name", list(importer.EXPECTED_HASHES))
def test_fixed_receipt_byte_tampering_blocks_import(parent, name):
    path = parent / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact hash changed"):
        importer.validate_saved(parent)


@pytest.mark.parametrize(
    "field,value",
    [("pending_files", 1), ("registered_files", 3), ("fresh_confirmation_authorized", True)],
)
def test_gate_denominators_and_disabled_authorizations_are_checked(parent, field, value):
    gate = json.loads((parent / importer.GATE).read_bytes())
    gate[field] = value
    rebind(parent, importer.GATE, gate)
    with pytest.raises(ValueError, match="endpoint denominators"):
        importer.validate_saved(parent)


def test_rebound_worker_cannot_drop_a_proof(parent):
    name = "workers/shard-0.json"
    worker = json.loads((parent / name).read_bytes())
    worker["committed"] = []
    worker_sha = rebind(parent, name, worker)
    gate = json.loads((parent / importer.GATE).read_bytes())
    gate["worker_bindings"][str(parent / name)] = worker_sha
    rebind(parent, importer.GATE, gate)
    with pytest.raises(ValueError, match="dropped, duplicated"):
        importer.validate_saved(parent)


def test_saved_worker_cannot_reference_another_plan(parent):
    name = "workers/shard-0.json"
    worker = json.loads((parent / name).read_bytes())
    worker["plan_sha256"] = "f" * 64
    worker_sha = rebind(parent, name, worker)
    gate = json.loads((parent / importer.GATE).read_bytes())
    gate["worker_bindings"][str(parent / name)] = worker_sha
    rebind(parent, importer.GATE, gate)
    with pytest.raises(ValueError, match="worker receipt differs"):
        importer.validate_saved(parent)


def test_original_cpu_count_must_match_exact_saved_suite(parent, monkeypatch):
    value = json.loads((parent.parent / "GATE_cpu_suite.json").read_bytes())
    value["counts"]["tests"] = 4
    monkeypatch.setattr(
        importer, "PARENT_CPU_GATE_SHA", put(parent.parent / "GATE_cpu_suite.json", value)
    )
    with pytest.raises(ValueError, match="CPU receipt differs"):
        importer.validate_saved(parent)


def test_metadata_mutation_detected_without_redecoding_raw_result(parent, tmp_path):
    (tmp_path / "source.py").write_text("# changed recorded source\n")
    with pytest.raises(ValueError, match="artifact hash changed"):
        importer.validate_saved(parent)


def test_other_parent_rejected_even_if_receipts_exist(parent, tmp_path):
    other = tmp_path / "another-parent"
    other.mkdir()
    with pytest.raises(ValueError, match="exact completed V18"):
        importer.validate_saved(other)


def test_alias_to_exact_parent_is_canonical_not_a_different_run(parent, tmp_path):
    alias = tmp_path / "parent-alias"
    alias.symlink_to(parent, target_is_directory=True)
    assert importer.validate_saved(alias)["gate"]["source_sha"] == importer.PARENT_SOURCE


def test_failed_current_cpu_suite_blocks_before_parent_reads(parent, tmp_path, monkeypatch):
    def failed(*args):
        raise ValueError("new CPU suite failed")

    monkeypatch.setattr(importer, "cpu_gate", failed)
    with pytest.raises(ValueError, match="new CPU suite failed"):
        importer.import_completed(tmp_path / "new", parent)


def test_write_once_import_never_overwrites_prior_output(parent, tmp_path):
    root = tmp_path / "import"
    importer.import_completed(root, parent)
    old = (root / "reuse.json").read_bytes()
    with pytest.raises(ValueError, match="write-once"):
        importer.import_completed(root, parent)
    assert (root / "reuse.json").read_bytes() == old
