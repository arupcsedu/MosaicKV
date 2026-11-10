"""Synthetic producer bindings only; no historical tree, YAML parser or GPU."""

import json
from pathlib import Path

import evicastkv_freshness_producer_canary as canary
import pytest
import yaml


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return canary.hash_bytes(raw)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    history, workspace = tmp_path / "history", tmp_path / "workspace"
    campaign, outputs = history / "campaign", history / "outputs"
    tasks = ("mmstar", "textvqa", "docvqa", "chartqa")
    rows, errors, manifests, ledger_bindings = [], [], [], {}
    for index, task in enumerate(tasks):
        dataset = dict(id=f"synthetic/{task}", revision="fixed", split="test")
        scope = dict(dataset_id=dataset["id"], dataset_revision="fixed", split="test", task=task)
        ids = [f"{task}-a", f"{task}-b", f"{task}-c"]
        ledger = history / "ledgers" / f"{task}.json"
        ledger_sha = put(ledger, dict(**scope, sample_count=3, sample_ids=ids))
        ledger_bindings[task] = dict(path=str(ledger), sha256=ledger_sha)
        shard = campaign / "sample_shards" / task / "00000.json"
        shard_sha = put(
            shard,
            dict(
                **scope,
                sample_ids=ids[:2],
                shard_index=0,
                shard_count=2,
                parent_ledger_sha=ledger_sha,
            ),
        )
        payload = dict(
            dataset=dataset, diagnostics=dict(sample_ids=ids[:1], sample_ledger_sha=ledger_sha)
        )
        config_sha = canary.hash_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        config = campaign / "configs" / f"{index:06d}-{config_sha}-00000.yaml"
        put(config, yaml.safe_dump(payload, sort_keys=True).encode())
        run_id = f"fixture-{task}"
        manifest = outputs / run_id / "manifest.json"
        put(
            manifest,
            dict(
                run_id=run_id,
                dataset=dataset,
                resolved_config=payload,
                source=dict(git_sha="a" * 40, git_dirty=False, config_sha=config_sha),
            ),
        )
        manifests.append(manifest)
        rows.append(
            dict(
                array_index=index,
                task_key=task,
                task=task,
                output_root=str(outputs),
                run_id=run_id,
                config_path=str(config),
                config_sha=config_sha,
                full_sample_ledger_path=str(ledger),
                full_sample_ledger_sha=ledger_sha,
                sample_id_file=str(shard),
                sample_id_file_sha=shard_sha,
                shard_index=0,
                shard_count=2,
                diagnostic_sample_count=1,
            )
        )
        errors.append(dict(path=str(config), error=canary.YAML_GAP))
    index = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    index_sha = put(campaign / "jobs.jsonl", index)
    matrix = dict(
        index_sha256=index_sha,
        job_count=4,
        output_root=str(outputs),
        source_git_sha="a" * 40,
        sample_ledgers=ledger_bindings,
        sample_limit=None,
        shard_size=2,
        diagnostic_size=1,
    )
    put(campaign / "matrix_manifest.json", matrix)
    inventory = tmp_path / "inventory.json"
    inventory_sha = put(inventory, dict(errors=[*errors, dict(path="unknown", error="unknown")]))
    monkeypatch.setattr(canary, "HISTORY", history)
    monkeypatch.setattr(canary, "WORKSPACE", workspace)
    monkeypatch.setattr(canary, "SLOTS", tuple(enumerate(tasks)))
    monkeypatch.setattr(canary, "clean_source", lambda _source: "b" * 40)
    monkeypatch.setattr(canary, "cpu_gate", lambda _root, _revision: dict(status="passed"))
    return dict(
        campaign=campaign,
        root=workspace / "canary",
        inventory=inventory,
        inventory_sha=inventory_sha,
        rows=rows,
        manifests=manifests,
        matrix=matrix,
    )


def execute(e):
    return canary.run(
        e["root"],
        e["campaign"],
        cpu_gate_root=e["root"].parent,
        inventory_path=e["inventory"],
        inventory_sha256=e["inventory_sha"],
    )


def test_four_producer_configs_verify_without_any_yaml_loading(evidence, monkeypatch):
    def no_load(*_args, **_kwargs):
        raise AssertionError("historical YAML must never be parsed")

    monkeypatch.setattr(yaml, "safe_load", no_load)
    monkeypatch.setattr(yaml, "load", no_load)
    result = execute(evidence)
    assert result["status"] == "completed_producer_bound_config_canary"
    assert result["verified"] == result["selected"] == 4
    assert not result["historical_yaml_parsed"]
    assert result["original_error_count"] == 5 and result["error_waivers"] == 0
    assert not result["fresh_confirmation_authorized"]
    assert all(
        row["shard_identity_count"] == 2 and not row["identities_exported"]
        for row in result["rows"]
    )
    assert (evidence["root"] / "GATE_freshness_producer_canary.json").is_file()
    assert canary.sha(evidence["inventory"]) == evidence["inventory_sha"]


def test_missing_producer_json_blocks_exact_slot_without_replacement(evidence):
    path = evidence["manifests"][1]
    path.rename(path.with_suffix(".saved"))
    result = execute(evidence)
    assert result["verified"] == 3 and result["selected"] == 4
    assert result["rows"][1]["status"] == "blocked_not_waived"
    assert result["rows"][1]["index"] == 1


def test_mismatched_historical_yaml_is_detected_as_bytes(evidence):
    put(Path(evidence["rows"][0]["config_path"]), b"!!untrusted historical YAML\n")
    result = execute(evidence)
    assert result["verified"] == 3
    assert "safe_dump bytes differ" in result["rows"][0]["error"]
    assert not result["historical_yaml_parsed"]


def test_producer_payload_digest_mismatch_fails_closed(evidence):
    path = evidence["manifests"][2]
    document = json.loads(path.read_text())
    document["resolved_config"]["unexpected"] = "changed"
    put(path, document)
    result = execute(evidence)
    assert result["verified"] == 3
    assert "canonical digest" in result["rows"][2]["error"]


def test_parent_ledger_mutation_is_not_a_coverage_pass(evidence):
    put(Path(evidence["rows"][0]["full_sample_ledger_path"]), b"{}")
    result = execute(evidence)
    assert "artifact hash changed" in result["rows"][0]["error"]
    assert result["error_waivers"] == 0


def test_valid_shard_hash_still_requires_exact_parent_slice(evidence):
    row = evidence["rows"][3]
    path = Path(row["sample_id_file"])
    document = json.loads(path.read_text())
    document["sample_ids"].reverse()
    row["sample_id_file_sha"] = put(path, document)
    index = b"".join(json.dumps(r).encode() + b"\n" for r in evidence["rows"])
    evidence["matrix"]["index_sha256"] = put(evidence["campaign"] / "jobs.jsonl", index)
    put(evidence["campaign"] / "matrix_manifest.json", evidence["matrix"])
    result = execute(evidence)
    assert "exact indexed parent-ledger slice" in result["rows"][3]["error"]


def test_changed_index_rejected_before_opening_producer_manifests(evidence):
    put(evidence["campaign"] / "jobs.jsonl", b"{}\n")
    with pytest.raises(ValueError, match="artifact hash changed"):
        execute(evidence)
    assert not evidence["root"].exists()


def test_explicit_inventory_and_cpu_gate_are_required(evidence):
    with pytest.raises(ValueError, match="verified CPU gate"):
        canary.run(evidence["root"], evidence["campaign"])
    evidence["inventory_sha"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hash changed"):
        execute(evidence)
    assert not evidence["root"].exists()


def test_completed_artifacts_are_write_once(evidence):
    execute(evidence)
    with pytest.raises(ValueError, match="immutable workspace output"):
        execute(evidence)
