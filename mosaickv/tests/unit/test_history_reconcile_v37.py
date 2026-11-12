"""CPU-only metadata lineage tests; no historical file or dataset access."""

import json

import evicastkv_history_reconcile_v37 as cli
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return cli.sha(path)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SCRATCH", tmp_path)
    return tmp_path


def test_metadata_hash_is_required_when_bound(scratch):
    path = scratch / "proof.json"
    expected = put(path, {"sample_id": "x"})
    bindings = {}
    assert cli.metadata(path, bindings, expected) == {"sample_id": "x"}
    assert bindings == {str(path): expected}
    path.write_text('{"sample_id":"y"}')
    with pytest.raises(ValueError):
        cli.metadata(path, {}, expected)


def test_metadata_rejects_external_alias(scratch, tmp_path_factory):
    outside = tmp_path_factory.mktemp("external") / "proof.json"
    put(outside, {})
    alias = scratch / "proof.json"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="project-owned"):
        cli.metadata(alias, {})


def test_metadata_never_opens_credentials_or_aliases(scratch, monkeypatch):
    path = scratch / ".codex/auth.json"
    put(path, {"token": "NEVER_READ"})
    alias = scratch / "safe.json"
    alias.symlink_to(path)

    def fail(*args, **kwargs):
        raise AssertionError("credential body was opened")

    monkeypatch.setattr(cli, "read_bytes", fail)
    with pytest.raises(ValueError, match="credential metadata"):
        cli.metadata(path, {})
    with pytest.raises(ValueError, match="credential aliases"):
        cli.metadata(alias, {})


def test_import_yaml_exclusions_from_exact_bound_workers(scratch):
    bindings = {}
    for index in range(8):
        path = scratch / "yaml/workers" / f"shard-{index}.json"
        bindings[str(path)] = put(
            path,
            {"committed": [{"identities": {"sample_ids": [f"old-{index}"]}}]},
        )
    verified = {"raws": {"plan.json": json.dumps({"bindings": bindings}).encode()}}
    observed = {}
    proofs = cli.imported_identity_proofs(verified, observed)
    assert len(proofs) == 8
    assert observed == bindings
    assert proofs[-1]["identities"]["sample_ids"] == ["old-7"]


def test_missing_old_yaml_worker_is_not_silently_ignored(scratch):
    verified = {"raws": {"plan.json": b'{"bindings":{}}'}}
    with pytest.raises(ValueError, match="eight preserved"):
        cli.imported_identity_proofs(verified, {})


def test_import_retained_oversize_inventory_by_gate_hash(scratch):
    path = scratch / "oversize/combined_identity_inventory.json"
    expected = put(path, {"identities": {"sample_ids": ["retained-old-id"]}})
    gate = scratch / "oversize/GATE_pending_oversize_recovery.json"
    gate_sha = put(gate, {"identity_inventory_sha256": expected})
    verified = {"raws": {"plan.json": json.dumps({"bindings": {str(gate): gate_sha}}).encode()}}
    observed = {}
    inventories = cli.imported_identity_inventories(verified, observed)
    assert inventories[0]["inventory"]["identities"]["sample_ids"] == ["retained-old-id"]
    assert observed[str(path)] == expected
    path.write_text('{"identities":{"sample_ids":[]}}')
    with pytest.raises(ValueError):
        cli.imported_identity_inventories(verified, {})


def test_missing_oversize_gate_blocks_identity_import(scratch):
    verified = {"raws": {"plan.json": b'{"bindings":{}}'}}
    with pytest.raises(ValueError, match="one preserved"):
        cli.imported_identity_inventories(verified, {})


def test_missing_coverage_wave_preserves_failure(scratch):
    assert cli.checked_coverage(scratch, {}, {}) == (None, ["coverage_wave_missing_or_failed"])


def test_coverage_without_closeout_authority_not_used(scratch):
    put(scratch / "review/coverage/GATE_coverage_resolution_v36.json", {})
    overlay, blockers = cli.checked_coverage(scratch, {}, {})
    assert overlay is None
    assert blockers == ["coverage_wave_not_qualified_by_closeout"]


def test_source_bridge_rejects_frozen_source_edits(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: "M\tmosaickv/model.py\n")
    with pytest.raises(ValueError, match="may not modify"):
        cli.source_bridge()


def test_source_bridge_keeps_model_and_analyzer_identities_separate(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess, "check_output", lambda *a, **kw: "A\tmosaickv/scripts/new_analysis.py\n"
    )
    result = cli.source_bridge()
    assert result["model_source_sha"] == cli.BASE
    assert result["analyzer_additions"] == ["mosaickv/scripts/new_analysis.py"]


def test_complete_coverage_overlay_must_match_worker_payloads(scratch):
    root = scratch / "history"
    folder = root / "review/coverage"
    plan_hash = put(folder / "plan.json", {"source_sha": cli.BASE, "bindings": {}})
    worker = folder / "workers/shard-0.json"
    proof = {"identities": {"sample_ids": ["safe-id"]}}
    worker_hash = put(
        worker,
        {"source_sha": cli.BASE, "plan_sha256": plan_hash, "committed": [proof]},
    )
    overlay_hash = put(folder / "coverage_overlay.json", {"source_sha": cli.BASE, "new_proofs": []})
    report_hash = put(folder / "net_error_report.json", {})
    gate = {
        "source_sha": cli.BASE,
        "coverage_overlay_sha256": overlay_hash,
        "net_error_report_sha256": report_hash,
        "worker_bindings": {str(worker): worker_hash},
        "missing_files": 0,
        "blocked_files": 0,
        "pending_files": 0,
    }
    gate_hash = put(folder / "GATE_coverage_resolution_v36.json", gate)
    with pytest.raises(ValueError, match="actual committed worker"):
        cli.checked_coverage(root, {}, {"gate_sha256": gate_hash, "gate": gate})


def test_canonical_analyzer_rejects_reused_output_before_reading_inputs(scratch, monkeypatch):
    monkeypatch.setattr(cli, "clean_source", lambda *a: "b" * 40)
    monkeypatch.setattr(cli, "source_bridge", lambda: {})
    output = scratch / "old-derived-result"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new immutable"):
        cli.analyze(scratch, output, cpu_gate_root=scratch)


def test_current_cpu_failure_blocks_analysis_before_historical_reads(scratch, monkeypatch):
    monkeypatch.setattr(cli, "clean_source", lambda *a: "b" * 40)
    monkeypatch.setattr(cli, "source_bridge", lambda: {})

    def fail(*args):
        raise ValueError("CPU suite failed")

    monkeypatch.setattr(cli, "cpu_gate", fail)
    with pytest.raises(ValueError, match="CPU suite failed"):
        cli.analyze(scratch, scratch / "new-output", cpu_gate_root=scratch)
    assert not (scratch / "new-output").exists()
