"""Content-addressed configuration recovery, not benchmark or fresh data work."""

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import evicastkv_freshness_recipe_recovery as recovery
import pytest
import yaml


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return recovery.hash_bytes(raw)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    history, workspace = tmp_path / "history", tmp_path / "workspace"
    parent, snapshot = history / "campaign", tmp_path / "snapshot"
    rows, gaps, payloads, refs = [], [], {}, {}
    for index, task in enumerate(("a", "b")):
        ids = [f"{task}-0", f"{task}-1"]
        dataset = dict(id=f"synthetic/{task}", revision="fixed", split="test")
        scope = dict(dataset_id=dataset["id"], dataset_revision="fixed", split="test", task=task)
        ledger = history / "ledgers" / f"{task}.json"
        ledger_sha = put(ledger, dict(**scope, sample_count=2, sample_ids=ids))
        refs[task] = dict(path=str(ledger), sha256=ledger_sha)
        shard = parent / "sample_shards" / task / "00000.json"
        shard_sha = put(
            shard,
            dict(
                **scope, sample_ids=ids, parent_ledger_sha=ledger_sha, shard_index=0, shard_count=1
            ),
        )
        payload = dict(
            dataset=dataset, diagnostics=dict(sample_ids=ids[:1], sample_ledger_sha=ledger_sha)
        )
        key = recovery.hash_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        payloads[key] = payload
        config = parent / "configs" / f"{index:06d}-{key}-00000.yaml"
        put(config, yaml.safe_dump(payload, sort_keys=True).encode())
        rows.append(
            dict(
                array_index=index,
                config_path=str(config),
                config_sha=key,
                task_key=task,
                task=task,
                shard_index=0,
                shard_count=1,
                full_sample_ledger_path=str(ledger),
                full_sample_ledger_sha=ledger_sha,
                sample_id_file=str(shard),
                sample_id_file_sha=shard_sha,
                diagnostic_sample_count=1,
            )
        )
        gaps.append(dict(path=str(config), error=recovery.YAML_GAP))
    index_raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    index_sha = put(parent / "jobs.jsonl", index_raw)
    matrix = dict(
        source_git_sha=recovery.ORIGINAL_PRODUCER,
        index_sha256=index_sha,
        job_count=2,
        sample_ledgers=refs,
        sample_limit=None,
        shard_size=2,
        diagnostic_size=1,
    )
    put(parent / "matrix_manifest.json", matrix)
    inventory = tmp_path / "inventory.json"
    inv_sha = put(inventory, dict(errors=gaps))
    monkeypatch.setattr(recovery, "HISTORY", history)
    monkeypatch.setattr(recovery, "WORKSPACE", workspace)
    monkeypatch.setattr(recovery, "clean_source", lambda _path, expected=None: expected or "f" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda *_args: dict(status="passed"))
    return dict(
        parent=parent,
        root=workspace / "recovered",
        snapshot=snapshot,
        matrix=matrix,
        rows=rows,
        gaps={row["path"]: row for row in gaps},
        payloads=payloads,
        inventory=inventory,
        inventory_sha=inv_sha,
    )


def test_all_rows_require_exact_bytes_without_loading_yaml(fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("historical YAML loading forbidden")

    monkeypatch.setattr(yaml, "safe_load", forbidden)
    monkeypatch.setattr(yaml, "load", forbidden)
    proofs, ledgers, shards = recovery.verify_rows(
        fixture["parent"], fixture["matrix"], fixture["rows"], fixture["payloads"], fixture["gaps"]
    )
    assert len(proofs) == 2 and ledgers == shards == 2
    assert all(row["byte_identical"] and not row["error_waived"] for row in proofs)


def test_one_unmatched_config_blocks_bulk_fallback(fixture):
    fixture["payloads"].pop(next(iter(fixture["payloads"])))
    with pytest.raises(ValueError, match="config scope differs"):
        recovery.verify_rows(
            fixture["parent"],
            fixture["matrix"],
            fixture["rows"],
            fixture["payloads"],
            fixture["gaps"],
        )


def test_config_hash_does_not_substitute_for_actual_yaml_byte_match(fixture):
    Path(fixture["rows"][1]["config_path"]).write_bytes(b"arbitrary: untrusted YAML\n")
    with pytest.raises(ValueError, match="YAML byte mismatch"):
        recovery.verify_rows(
            fixture["parent"],
            fixture["matrix"],
            fixture["rows"],
            fixture["payloads"],
            fixture["gaps"],
        )


def test_changed_shard_is_not_covered_by_config_digest(fixture):
    Path(fixture["rows"][0]["sample_id_file"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="artifact hash changed"):
        recovery.verify_rows(
            fixture["parent"],
            fixture["matrix"],
            fixture["rows"],
            fixture["payloads"],
            fixture["gaps"],
        )


def test_inventory_hash_is_bound_before_any_export(fixture):
    with pytest.raises(ValueError, match="artifact hash changed"):
        recovery.inputs(fixture["parent"], fixture["inventory"], "0" * 64)


def test_absent_original_source_is_not_relabelled_as_recovery_snapshot(fixture):
    matrix, rows, gaps, provenance = recovery.inputs(
        fixture["parent"], fixture["inventory"], fixture["inventory_sha"]
    )
    assert matrix["source_git_sha"] == recovery.ORIGINAL_PRODUCER
    assert recovery.SNAPSHOT_SHA != recovery.ORIGINAL_PRODUCER
    assert len(rows) == len(gaps) == provenance["original_error_count"] == 2


def test_source_bundle_content_matches_snapshot_before_claim(tmp_path, monkeypatch):
    snapshot, bundle = tmp_path / "source", tmp_path / "source.tar"
    path = snapshot / recovery.MATRIX_RELATIVE
    put(path, b"trusted tracked source matrix\n")
    with tarfile.open(bundle, "w") as archive:
        archive.add(path, arcname="writer-4e5d3a/" + recovery.MATRIX_RELATIVE)
    monkeypatch.setattr(recovery, "BUNDLE_SHA", recovery.sha(bundle))
    result = recovery.verify_source_bundle(bundle, snapshot)
    assert result["source_files"][str(path)] == recovery.sha(path)
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="differs from immutable source bundle"):
        recovery.verify_source_bundle(bundle, snapshot)


def test_source_bundle_traversal_member_rejected(tmp_path, monkeypatch):
    bundle = tmp_path / "bad.tar"
    with tarfile.open(bundle, "w") as archive:
        member = tarfile.TarInfo("writer-4e5d3a/../outside")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(recovery, "BUNDLE_SHA", recovery.sha(bundle))
    with pytest.raises(ValueError, match="unsupported member"):
        recovery.verify_source_bundle(bundle, tmp_path)


def test_run_exports_configs_and_overlay_without_freshness(fixture, monkeypatch):
    monkeypatch.setattr(
        recovery, "verify_source_bundle", lambda *_args: dict(bundle_sha256="bound")
    )
    monkeypatch.setattr(recovery, "enumerate_payloads", lambda *_args: fixture["payloads"])
    result = recovery.run(
        fixture["root"],
        cpu_gate_root=fixture["root"].parent,
        parent=fixture["parent"],
        inventory_path=fixture["inventory"],
        inventory_sha256=fixture["inventory_sha"],
        snapshot=fixture["snapshot"],
    )
    assert result["status"] == "completed_exact_config_recovery_coverage_overlay"
    assert result["canonical_configs"] == result["indexed_configs"] == 2
    assert result["error_waivers"] == 0 and result["original_error_count"] == 2
    assert not result["fresh_confirmation_authorized"] and not result["historical_yaml_parsed"]
    assert (fixture["root"] / "canonical_configs.json").is_file()
    assert recovery.sha(fixture["inventory"]) == fixture["inventory_sha"]
    with pytest.raises(ValueError, match="immutable workspace"):
        recovery.run(fixture["root"], cpu_gate_root=fixture["root"].parent)


def test_failed_reconstruction_retains_structured_negative_gate(fixture, monkeypatch):
    monkeypatch.setattr(
        recovery, "verify_source_bundle", lambda *_args: dict(bundle_sha256="bound")
    )

    def fail(*args):
        raise ValueError("one canonical config digest failed")

    monkeypatch.setattr(recovery, "enumerate_payloads", fail)
    result = recovery.run(
        fixture["root"],
        cpu_gate_root=fixture["root"].parent,
        parent=fixture["parent"],
        inventory_path=fixture["inventory"],
        inventory_sha256=fixture["inventory_sha"],
        snapshot=fixture["snapshot"],
    )
    assert result["status"] == "blocked_exact_config_recovery"
    assert not (fixture["root"] / "canonical_configs.json").exists()


def test_child_guard_blocks_subprocess_before_config_enumeration(tmp_path):
    package = tmp_path / "mosaickv"
    package.mkdir()
    (package / "__init__.py").write_text("import subprocess\nsubprocess.run(['/bin/true'])\n")
    result = subprocess.run(
        [recovery.PYTHON, "-B", "-c", recovery.CHILD_CODE, "not_reached"],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(tmp_path),
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="",
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "cannot launch processes or network" in result.stderr
