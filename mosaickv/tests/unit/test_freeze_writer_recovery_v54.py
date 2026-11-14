"""Synthetic execution-link projection; no historic logs or artifact bodies."""

import hashlib

import evicastkv_freeze_writer_recovery_v54 as runner
import pytest

BASE = b"""classification: sealed_negative_label_blind_calibration_result
frozen_candidate_created: false
run_root: /scratch/djy8hg/runs/calibration
protocol:
  path: configs/experiments/example.yaml
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
slurm_jobs:
  freeze: 18572260
"""


def test_base_job_and_protocol_only_are_retained():
    result = runner.execution_projection(
        BASE + b"scores:\n  value: PRIVATE_OUTPUT\n  details: [a,b,c]\n", runner.prior.Budget()
    )
    assert result == dict(
        classification="sealed_negative_label_blind_calibration_result",
        frozen_candidate_created=False,
        run_root="/scratch/djy8hg/runs/calibration",
        protocol_path="configs/experiments/example.yaml",
        protocol_sha256="a" * 64,
        freeze_job="18572260",
    )
    assert "PRIVATE_OUTPUT" not in repr(result)


def test_robust_wrapper_success_preserves_negative_freeze_state():
    raw = BASE.replace(
        b"slurm_jobs:\n  freeze: 18572260",
        b"execution_jobs:\n  freeze_analysis: 18650057\n"
        b"  freeze_state: completed_failed_closed_no_eligible_candidate",
    )
    result = runner.execution_projection(raw, runner.prior.Budget())
    assert result["freeze_job"] == "18650057"
    assert result["freeze_state"] == "completed_failed_closed_no_eligible_candidate"
    assert result["frozen_candidate_created"] is False


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "two_job_fields",
        "missing_protocol",
        "positive_freeze",
        "alias",
        "tag",
        "merge",
        "two_documents",
        "array",
        "bad_syntax",
        "wrong_root",
        "missing_state",
    ],
)
def test_malformed_or_ambiguous_original_link_fails_closed(change):
    raw = BASE
    if change == "duplicate":
        raw += b"run_root: other\n"
    elif change == "two_job_fields":
        raw += b"execution_jobs:\n  freeze_analysis: 18572260\n"
    elif change == "missing_protocol":
        raw = raw.replace(b"  path: configs/experiments/example.yaml\n", b"")
    elif change == "positive_freeze":
        raw = raw.replace(b"created: false", b"created: true")
    elif change == "alias":
        raw += b"unknown: &anchor value\n"
    elif change == "tag":
        raw += b"unknown: !!str value\n"
    elif change == "merge":
        raw += b"unknown:\n  <<: {}\n"
    elif change == "two_documents":
        raw += b"---\nother: thing\n"
    elif change == "array":
        raw = raw.replace(b"  freeze: 18572260", b"  freeze: [18572260]")
    elif change == "bad_syntax":
        raw += b"unknown: [unterminated\n"
    elif change == "wrong_root":
        raw = b"[a,b,c]"
    elif change == "missing_state":
        raw = raw.replace(b"slurm_jobs:\n  freeze:", b"execution_jobs:\n  freeze_analysis:")
    with pytest.raises((ValueError, runner.yaml.YAMLError, StopIteration)):
        runner.execution_projection(raw, runner.prior.Budget())


@pytest.mark.parametrize("location", ["key", "value"])
def test_scalar_bound_covers_discarded_values_and_keys(location):
    large = b"x" * (2**20 + 1)
    raw = BASE + (b"? " + large + b"\n: value\n" if location == "key" else b"unknown: " + large)
    with pytest.raises(ValueError, match="1-MiB"):
        runner.execution_projection(raw, runner.prior.Budget())


def test_projection_keeps_unchanged_byte_depth_node_and_time_limits(monkeypatch):
    budget = runner.prior.Budget()
    budget.started -= 211
    with pytest.raises(TimeoutError):
        runner.execution_projection(BASE, budget)
    with pytest.raises(ValueError, match="64-level"):
        runner.execution_projection(
            BASE + b"unknown: " + b"[" * 66 + b"0" + b"]" * 66, runner.prior.Budget()
        )
    monkeypatch.setattr(runner.prior, "LIVE_NODES", 2)
    with pytest.raises(ValueError, match="node"):
        runner.execution_projection(BASE, runner.prior.Budget())
    monkeypatch.setattr(runner.prior, "LIVE_BYTES", 1)
    with pytest.raises(ValueError, match="8-MiB"):
        runner.execution_projection(BASE, runner.prior.Budget())


def test_original_source_hash_is_checked_not_replaced_with_current_code(monkeypatch):
    source = b"# synthetic source; never executed\n"
    outputs = iter([str(len(source)).encode(), source])
    monkeypatch.setattr(runner.seals, "git_output", lambda *a, **k: next(outputs))
    assert (
        runner.source_bytes(
            "a" * 40,
            "mosaickv/scripts/analyzer.py",
            hashlib.sha256(source).hexdigest(),
            runner.prior.Budget(),
        )
        == source
    )


@pytest.mark.parametrize(
    "path",
    [".codex/auth.json", "mosaickv/.codex/auth.json", "mosaickv/credentials.json", "mosaickv/.env"],
)
def test_changed_source_and_credential_paths_rejected(monkeypatch, path):
    monkeypatch.setattr(
        runner.seals, "git_output", lambda *a, **k: pytest.fail("unscoped source opened")
    )
    with pytest.raises(ValueError):
        runner.source_bytes("a" * 40, path, "b" * 64, runner.prior.Budget())


def test_full_coordinator_promotes_seven_and_preserves_other_127_without_body_io(
    monkeypatch, tmp_path
):
    import json
    from copy import deepcopy
    from pathlib import Path

    from test_parquet_binding_recovery_v49 import fixture

    from mosaickv.evaluation import freeze_writer_bindings_v54 as runtime
    from mosaickv.evaluation import negative_seal_bindings_v53 as sealed
    from mosaickv.evaluation.fox_campaign import digest
    from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding

    # Tracked source-registry metadata, not original results or artifact bodies.
    registry = json.loads(runner.REGISTRY.read_bytes())
    facts_by_version = {}
    parent_rows = []
    before_rows = []
    inventory_rows = []
    transactions = {}
    sealed_claims = {}
    for spec in registry["specs"]:
        fact = {
            key: spec[key]
            for key in (
                "version",
                "job_id",
                "source_sha",
                "repository_tree_oid",
                "frozen_output",
                "protocol_path",
                "protocol_sha256",
                "seal_path",
                "seal_sha256",
                "expected_state",
                "expected_exit_code",
                "accounting_sha256",
                "startup_sha256",
                "batch_record_sha256",
                "wrapper_path",
                "wrapper_sha256",
                "entrypoint_path",
            )
        }
        fact.update(
            schema="historical_freeze_runtime_facts_v54",
            run_root=runtime._path(spec["run_root"]),
            frozen_output=runtime._path(spec["frozen_output"]),
            source_blob_sha256s=deepcopy(spec["source_blob_sha256s"]),
            outputs=[dict(value, path=runtime._path(value["path"])) for value in spec["outputs"]],
            critical_source_closure_sha256=digest(spec["source_blob_sha256s"]),
            negative_result_preserved=True,
            quality_gate_pass=False,
            scientific_dependency_validation=False,
            canonical_eligible=False,
        )
        fact["runtime_evidence_sha256"] = digest(fact)
        runtime.validate_runtime_facts(fact)
        facts_by_version[spec["version"]] = fact
        for output in spec["outputs"]:
            row, transaction, _, _ = fixture()
            path = output["path"]
            row["path"] = transaction["path"] = path
            transaction["before_sha256"] = transaction["after_sha256"] = output["artifact_sha256"]
            row["original_registration_sha256"] = digest(
                {
                    name: row[name]
                    for name in sealed.contract.OCCURRENCE_FIELDS
                    | {"path", "signature", "operation"}
                }
            )
            before = classify_binding(row, transaction, [], reviewed_writers={})
            claim = sealed.artifact_claim(
                dict(
                    classification=sealed.CLASSIFICATION,
                    immutable_result=True,
                    frozen_candidate_created=False,
                    run_root=spec["run_root"],
                    artifacts={"ranking": dict(path=path, sha256=output["artifact_sha256"])},
                ),
                str(runner.SOURCE / spec["seal_path"]),
                spec["seal_sha256"],
                path,
            )
            parent_rows.append(
                sealed.augment_seal_binding(
                    before, row, transaction, [], [claim], reviewed_writers={}
                )
            )
            before_rows.append(before)
            inventory_rows.append(row)
            transactions[path] = transaction
            sealed_claims[path] = [claim]
    for index in range(127):
        # Include other previously qualified variants that must be copied, not
        # reclassified under the new runtime-writer registry.
        previous = deepcopy(parent_rows[0])
        path = f"/synthetic/unchanged-{index}.parquet"
        previous.update(
            path=path, schema="existing_snapshot_or_dirty_binding", status="preserved_partial"
        )
        parent_rows.append(previous)
        row = deepcopy(inventory_rows[0])
        row["path"] = path
        inventory_rows.append(row)
        before_rows.append(deepcopy(previous))
        transactions[path] = {}
    parent = dict(
        source_sha="f9fe413d85207606a21aaaecc8b7f977b9ab7f66",
        rows=parent_rows,
        supplemental_claims=sealed_claims,
        original_occurrence_counts={"net_unresolved_error_count": 135},
    )
    before = dict(rows=before_rows)
    original = dict(rows=deepcopy(before_rows), binding_claims={r["path"]: [] for r in parent_rows})
    saved = deepcopy(parent)
    records = [
        dict(
            job_id=s["job_id"],
            original_public_accounting_record="synthetic",
            original_clean_startup_prefix="synthetic",
            original_public_batch_script_record="synthetic",
        )
        for s in registry["specs"]
    ]

    def read(path, *_):
        if path == runner.REGISTRY:
            return registry
        if path.parent == runner.PARENT:
            return parent
        if path.parent == runner.seals.PARENT:
            return before
        if path.parent == runner.startup.PARENT:
            return original
        if path.parent == runner.prior.PARENT:
            return dict(rows=inventory_rows)
        return dict(records=records if path.name == next(iter(runner.CAPSULES)) else [])

    monkeypatch.setattr(runner, "saved_json", read)
    monkeypatch.setattr(runner, "FOLLOWUPS", {})
    monkeypatch.setattr(runner, "clean_source", lambda _: "f" * 40)
    monkeypatch.setattr(runner.prior.resume.reader.common, "cpu_gate", lambda *_: None)
    monkeypatch.setattr(runner, "source_bytes", lambda *a: b"synthetic source; never executed")
    monkeypatch.setattr(runner, "execution_projection", lambda *a: {})
    monkeypatch.setattr(
        runner, "validate_freeze_runtime", lambda *a: deepcopy(facts_by_version[a[4]["version"]])
    )
    trees = {s["source_sha"]: s["repository_tree_oid"] for s in registry["specs"]}
    monkeypatch.setattr(
        runner.seals, "git_output", lambda args, *a, **k: trees[args[-1].split("^")[0]].encode()
    )
    monkeypatch.setattr(runner.prior, "REVIEWED", {})
    signatures = {r["path"]: r["signature"] for r in parent_rows}
    monkeypatch.setattr(runner.prior, "allowed", lambda path: Path(path))
    monkeypatch.setattr(runner.prior, "geometry", lambda path: signatures[str(path)])
    reused = []

    def reuse(row, *_):
        reused.append(row["path"])
        return transactions[row["path"]]

    monkeypatch.setattr(runner.startup, "reused_transaction", reuse)

    def forbidden(*a, **k):
        pytest.fail("historical source/artifact proof replayed")

    monkeypatch.setattr(runner.prior, "hash_pass", forbidden)
    monkeypatch.setattr(runner.prior.resume.reader.common, "hash_file", forbidden)
    monkeypatch.setattr(runner.prior.resume, "require_artifact_capacity", lambda *_: None)
    monkeypatch.setattr(runner, "saved_sha", lambda *_: "a" * 64)
    outputs = {}
    monkeypatch.setattr(
        runner, "write_once", lambda path, value: outputs.update({path.name: value})
    )
    result = runner.recover(tmp_path)
    report = outputs["recovered_bindings.json"]
    assert result["promoted"] == 7
    assert len(reused) == len(set(reused)) == 134
    assert parent == saved
    assert all(a == b for a, b in zip(report["rows"][7:], parent_rows[7:], strict=True))
    assert all(
        row["status"] == "runtime_producer_and_artifact_binding_verified"
        for row in report["rows"][:7]
    )
    assert report["parquet_bodies_reopened"] == report["original_startup_logs_reopened"] == 0
    assert report["fresh_confirmation_authorized"] is False
