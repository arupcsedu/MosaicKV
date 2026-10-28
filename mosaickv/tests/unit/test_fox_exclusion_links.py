"""Historical relative-ledger links are manifest-relative, never CWD-relative."""

import json

import pytest
from evicastkv_evidence import exclusion_inventory


def test_inventory_resolves_relative_links_and_retains_missing_links(tmp_path):
    runs = tmp_path / "runs"
    historical = runs / "coco-fixture"
    ledger = historical / "ledgers/inputs.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(dict(sample_ids=["ActionSequence-0019"])))
    (historical / "plan.json").write_text(
        json.dumps(
            dict(
                ledger_path="ledgers/inputs.json",
                nested=dict(ledger="ledgers/missing.json"),
            )
        )
    )
    root = tmp_path / "new"
    root.mkdir()
    (root / "protocol_lock.json").write_text(json.dumps(dict(cohorts={})))
    output = root / "inventory.json"
    exclusion_inventory(root, output, run_root=runs)
    data = json.loads(output.read_text())
    assert data["MileBench_consumed_ids"] == ["ActionSequence-0019"]
    assert str(ledger.resolve()) in data["bound_files"]
    assert len(data["errors"]) == 1
    assert data["errors"][0]["path"] == str(historical / "ledgers/missing.json")
    assert not data["complete_scan"] and not data["confirmation_authorized"]


def test_inventory_does_not_search_other_directories_for_missing_links(tmp_path):
    runs = tmp_path / "runs"
    historical = runs / "coco-fixture"
    historical.mkdir(parents=True)
    (historical / "plan.json").write_text(json.dumps(dict(ledger_path="inputs.json")))
    (runs / "inputs.json").write_text(json.dumps(dict(sample_ids=["ActionSequence-0001"])))
    root = tmp_path / "new"
    root.mkdir()
    (root / "protocol_lock.json").write_text(json.dumps(dict(cohorts={})))
    output = root / "inventory.json"
    exclusion_inventory(root, output, run_root=runs)
    data = json.loads(output.read_text())
    assert not data["MileBench_consumed_ids"] and len(data["errors"]) == 1


def test_explicit_seed_names_cover_protocol_locks_without_scanning_raw_logs(tmp_path):
    runs = tmp_path / "runs"
    historical = runs / "milebench-fixture"
    historical.mkdir(parents=True)
    (historical / "protocol_lock.json").write_text(
        json.dumps(dict(sample_ids=["IEdit-0008"], references=["ALFRED-0100"]))
    )
    (historical / "raw.jsonl").write_text("malformed scheduler output is not a seed")
    root = tmp_path / "new"
    root.mkdir()
    (root / "protocol_lock.json").write_text(json.dumps(dict(cohorts={})))
    data = exclusion_inventory(
        root,
        root / "inventory.json",
        run_root=runs,
        identity_tasks=("IEdit", "ALFRED"),
        extra_manifest_names=("protocol_lock.json",),
    )
    assert data["MileBench_consumed_ids"] == ["IEdit-0008"]
    assert not data["unclassified_workload_roots"] and not data["errors"]
    assert not data["complete_scan"] and not data["confirmation_authorized"]


@pytest.mark.parametrize("name", ["../raw.json", "nested/ledger.json", "raw.jsonl"])
def test_extra_manifest_names_cannot_widen_to_paths_or_raw_streams(tmp_path, name):
    with pytest.raises(ValueError, match="exact JSON basenames"):
        exclusion_inventory(tmp_path, tmp_path / "unused.json", extra_manifest_names=(name,))
