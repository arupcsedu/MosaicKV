"""Synthetic finite-frontier import and recovery, never fresh dataset tests."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import evicastkv_freshness_frontier_fork as fork
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return fork.sha(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    roots = (tmp_path / "history-a", tmp_path / "history-b")
    for root in roots:
        root.mkdir()
    parent, output, coverage = tmp_path / "parent", tmp_path / "fork", tmp_path / "coverage"
    source_sha = "f" * 40
    original_errors, sources = [], {}
    for index in range(8):
        path = roots[0] / f"consumed-{index}.json"
        put(path, dict(sample_id=f"consumed-{index}"))
        sources[index] = path
    queue_path = roots[1] / "queue.jsonl"
    raw = b'{"sample_id":"recovered-consumed","answer":"not-decoded"}\n'
    queue_path.write_bytes(raw)
    row = dict(path=str(queue_path), offset=0, error=fork.lines.PREFIX)
    original_errors.append(row)
    owner = fork.scanner.owner(queue_path, roots, 8)
    segment = [dict(start=0, end=len(raw), sha256=fork.lines.hashlib.sha256(raw).hexdigest())]
    binding = dict(
        signature=fork.scanner.signature(queue_path),
        segments=segment,
        segmented_content_root=fork.digest(segment),
    )
    inventory_bindings = {}
    originals = {}
    for index in range(8):
        state = fork.scanner.initial_state(roots)
        state["frontier"] = [
            dict(
                kind="file",
                path=str(sources[index]),
                signature=fork.scanner.signature(sources[index]),
            )
        ]
        if index == owner:
            state["bindings"][str(queue_path)] = binding
            state["errors"] = [row]
        registration = dict(
            source_sha=fork.OLD_SOURCE_SHA,
            shard_index=index,
            shard_count=8,
            source_sha256=fork.sha(Path(fork.scanner.__file__)),
            roots=list(map(str, roots)),
        )
        checkpoint_dir = parent / "shards" / f"shard-{index:03d}"
        registration_sha = put(checkpoint_dir / "registration.json", registration)
        checkpoint = dict(
            number=5,
            state=state,
            state_sha256=fork.digest(state),
            registration_sha256=registration_sha,
        )
        path = checkpoint_dir / "checkpoint-000005.json"
        inventory_bindings[str(path)] = put(path, checkpoint)
        originals[index] = fork.digest(state)
    inventory_path = parent / "aggregate/inventory.json"
    inventory_sha = put(
        inventory_path,
        dict(
            roots=list(map(str, roots)), errors=original_errors, shard_bindings=inventory_bindings
        ),
    )
    overlay = [
        dict(config_path=f"proven-{index}", byte_identical=True, error_waived=False)
        for index in range(9403)
    ]
    overlay_sha = put(coverage / "indexed_coverage_overlay.json", overlay)
    coverage_plan_sha = put(
        coverage / "plan.json", dict(bindings={str(inventory_path): inventory_sha})
    )
    put(
        coverage / "GATE_freshness_recipe_recovery.json",
        dict(
            status="completed_exact_config_recovery_coverage_overlay",
            indexed_configs=9403,
            source_sha=source_sha,
            fresh_confirmation_authorized=False,
            coverage_overlay_sha256=overlay_sha,
            plan_sha256=coverage_plan_sha,
        ),
    )
    monkeypatch.setattr(fork, "clean_source", lambda path, expected=None: expected or source_sha)
    monkeypatch.setattr(fork, "cpu_gate", lambda *args: dict(status="passed"))
    return dict(
        parent=parent,
        root=output,
        coverage=coverage,
        originals=originals,
        owner=owner,
        inventory=inventory_path,
        inventory_bindings=inventory_bindings,
    )


def prepare(fixture):
    return fork.prepare(
        fixture["root"],
        fixture["parent"],
        cpu_gate_root=fixture["root"].parent,
        coverage_root=fixture["coverage"],
    )


def test_import_is_exact_and_original_policy_is_separate(setup):
    result = prepare(setup)
    assert result["workers"] == 8 and result["oversized_offsets"] == 1
    plan = fork.load(setup["root"])
    assert plan["identity_projection_and_collection_unchanged"]
    assert plan["explicit_policy_difference"] == dict(old_MAX_LINE=2**20, new_MAX_LINE=8 * 2**20)
    assert plan["coverage_is_not_a_walk_skip_rule"]
    for index in range(8):
        saved = fork.read(setup["root"] / "imports" / f"shard-{index:03d}.json")
        assert fork.digest(saved["state"]) == setup["originals"][index]
        assert saved["old_source_sha"] == fork.OLD_SOURCE_SHA


def test_mutated_original_checkpoint_is_not_relabelled_as_fork_progress(setup):
    path = Path(next(iter(setup["inventory_bindings"])))
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact hash changed"):
        prepare(setup)


def test_worker_advances_frontier_and_recovers_registered_record(setup):
    prepare(setup)
    old_line = fork.scanner.MAX_LINE
    result = fork.worker(setup["root"], setup["owner"])
    assert result["files_added"] == result["records_added"] == 1
    assert result["frontier_complete"]
    assert result["recovered_records"] == 1 and result["pending_recovery_records"] == 0
    saved = fork.read(setup["root"] / "workers" / f"shard-{setup['owner']:03d}.json")
    assert saved["recovered_metadata"][0]["identities"]["sample_ids"] == ["recovered-consumed"]
    assert len(saved["state"]["errors"]) == 1  # Original error was never deleted.
    assert old_line == fork.scanner.MAX_LINE


def test_queue_mutation_fails_immutable_auxiliary_binding(setup):
    prepare(setup)
    put(setup["root"] / "oversize_queue.json", [])
    with pytest.raises(ValueError, match="immutable frontier-fork evidence changed"):
        fork.worker(setup["root"], 0)


def test_missing_workers_do_not_become_complete_scope(setup):
    prepare(setup)
    fork.worker(setup["root"], 0)
    report = fork.analyze(setup["root"])
    assert report["missing_shards"] == list(range(1, 8))
    assert not report["frontier_exhausted"]
    assert not report["fresh_confirmation_authorized"]


def test_all_finished_frontiers_still_require_coverage_and_media_certification(setup):
    prepare(setup)
    for index in range(8):
        fork.worker(setup["root"], index)
    report = fork.analyze(setup["root"])
    assert report["frontier_exhausted"] and report["newly_processed_files"] == 8
    assert report["recovered_oversize_records"] == 1
    assert report["original_and_new_coverage_errors"] == 1
    assert report["error_waivers"] == 0 and not report["fresh_confirmation_authorized"]
    inventory = fork.read(setup["root"] / "combined_identity_inventory.json")
    assert "recovered-consumed" in inventory["identities"]["sample_ids"]
    assert not inventory["complete_historical_id_inventory"]


def test_dry_launcher_is_bounded_to_eight_single_cpu_workers(setup, monkeypatch):
    prepare(setup)

    def no_submit(*args, **kwargs):
        raise AssertionError("dry launch must never submit")

    monkeypatch.setattr(fork.subprocess, "run", no_submit)
    result = fork.launch(setup["root"])
    assert result["jobs_submitted"] == 0
    assert "--array=0-7%8" in result["commands"][0]
    assert "--cpus-per-task=1" in result["commands"][0]
    assert "--cpus-per-task=4" in result["commands"][1]
    assert "--time=00:05:00" in result["commands"][0]
    assert all(
        not any("--gpus" in arg or "--gres" in arg for arg in args) for args in result["commands"]
    )


def test_source_semantic_drift_requires_explicit_review(setup):
    path = Path(next(iter(setup["inventory_bindings"])))
    saved = fork.read(path)
    registration_path = path.parent / "registration.json"
    registration = fork.read(registration_path)
    registration["source_sha256"] = "0" * 64
    saved["registration_sha256"] = put(registration_path, registration)
    updated = put(path, saved)
    inventory = fork.read(setup["inventory"])
    inventory["shard_bindings"][str(path)] = updated
    inventory_sha = put(setup["inventory"], inventory)
    coverage_plan_sha = put(
        setup["coverage"] / "plan.json", dict(bindings={str(setup["inventory"]): inventory_sha})
    )
    gate_path = setup["coverage"] / "GATE_freshness_recipe_recovery.json"
    gate = fork.read(gate_path)
    gate["plan_sha256"] = coverage_plan_sha
    put(gate_path, gate)
    with pytest.raises(ValueError, match="explicit review required"):
        prepare(setup)


def rebind_coverage_plan(fixture, bindings):
    coverage_plan_sha = put(fixture["coverage"] / "plan.json", dict(bindings=bindings))
    path = fixture["coverage"] / "GATE_freshness_recipe_recovery.json"
    gate = fork.read(path)
    gate["plan_sha256"] = coverage_plan_sha
    put(path, gate)


def test_inventory_alias_is_resolved_before_exact_binding_comparison(setup):
    alias = setup["root"].parent / "inventory-alias.json"
    alias.symlink_to(setup["inventory"])
    rebind_coverage_plan(setup, {str(alias): fork.sha(setup["inventory"])})
    assert prepare(setup)["workers"] == 8


def test_conflicting_alias_bindings_are_not_silently_normalized(setup):
    alias = setup["root"].parent / "inventory-alias.json"
    alias.symlink_to(setup["inventory"])
    rebind_coverage_plan(
        setup,
        {str(alias): "0" * 64, str(setup["inventory"]): fork.sha(setup["inventory"])},
    )
    with pytest.raises(ValueError, match="conflicting aliases"):
        prepare(setup)


def test_broken_binding_alias_fails_closed(setup):
    alias = setup["root"].parent / "inventory-alias.json"
    alias.symlink_to(setup["root"].parent / "missing.json")
    rebind_coverage_plan(setup, {str(alias): fork.sha(setup["inventory"])})
    with pytest.raises(FileNotFoundError):
        prepare(setup)


@pytest.mark.parametrize("uncertain", [False, True])
def test_analyzer_submission_failure_preserves_workers_without_retry(setup, monkeypatch, uncertain):
    prepare(setup)
    calls = []

    def submit(command, **kwargs):
        label = "workers" if not calls else "analysis"
        assert (setup["root"] / f"{label}-request.json").is_file()
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(stdout="12345;cluster\n")
        if uncertain:
            raise subprocess.TimeoutExpired(command, 30)
        raise subprocess.CalledProcessError(
            1, command, stderr="sbatch: error: Batch job submission failed: unavailable"
        )

    monkeypatch.setattr(fork.subprocess, "run", submit)
    result = fork.launch(setup["root"], execute=True)
    assert len(calls) == 2 and result["workers"] == "12345"
    assert "--dependency=afterany:12345" in calls[1]
    assert result["analysis"] is None and result["status"] == "blocked_submission"
    assert result["analysis_submission_uncertain"] is uncertain
    assert result["analysis_definitely_not_submitted"] is not uncertain
    assert not result["automatic_retry"]
    with pytest.raises(FileExistsError):
        fork.launch(setup["root"], execute=True)
    assert len(calls) == 2


def test_ambiguous_worker_submission_does_not_submit_analysis(setup, monkeypatch):
    prepare(setup)
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="unexpected scheduler response")

    monkeypatch.setattr(fork.subprocess, "run", submit)
    result = fork.launch(setup["root"], execute=True)
    assert len(calls) == 1 and result["workers"] is None
    assert result["workers_submission_uncertain"]
    assert result["analysis"] is None and result["analysis_definitely_not_submitted"]
