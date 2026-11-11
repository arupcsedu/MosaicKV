"""Synthetic immutable states and mocked Slurm only; no actual traversal."""

import json
import subprocess
from pathlib import Path

import evicastkv_history_closure_v16 as history
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return history.sha(path)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, recovery, old_root = (
        tmp_path / "closure",
        tmp_path / "recovered",
        tmp_path / "V13-history",
    )
    roots = [tmp_path / "original-runs-one", tmp_path / "original-runs-two"]
    for path in roots:
        path.mkdir()
    original_plan = old_root / "segment-9/plan.json"
    original_sha = put(
        original_plan,
        dict(
            source_sha=history.OLD_SHA,
            roots=list(map(str, roots)),
            same_scanner_file_sha256=history.sha(Path(history.scanner.__file__)),
        ),
    )
    rows, receipts, bindings = [], {}, {}
    for index in range(8):
        state = history.scanner.initial_state(tuple(roots))
        state["traversal_complete"] = False
        state["frontier"] = [
            dict(kind="directory", path=str(roots[0] / f"saved-{index}"), position=10)
        ]
        state["sample_ids"] = [f"consumed-{index}"]
        state["errors"] = [dict(path=f"unknown-{index}.yaml", error="old unsupported metadata")]
        path = old_root / "segment-9/workers" / f"shard-{index}.json"
        expected = put(
            path,
            dict(
                source_sha=history.OLD_SHA,
                index=index,
                state=state,
                state_sha256=history.digest(state),
            ),
        )
        stdout = old_root / "slurm" / f"worker-{index}.out"
        stdout.parent.mkdir(exist_ok=True)
        stdout.write_text(f"clean_git_sha={history.OLD_SHA}\n")
        stdout_sha = history.sha(stdout)
        receipts[str(stdout)] = stdout_sha
        bindings[str(path)] = expected
        rows.append(
            dict(
                index=index,
                path=str(path),
                sha256=expected,
                stdout=str(stdout),
                stdout_sha256=stdout_sha,
            )
        )
    inventory = recovery / "combined_identity_inventory.json"
    identities = {name: [] for name in history.IDENTITIES}
    identities["sample_ids"] = ["earlier-overlay-ID"]
    inventory_sha = put(inventory, dict(identities=identities))
    put(
        recovery / "plan.json",
        dict(
            source_sha=history.PARENT_SOURCE,
            parent=str(old_root),
            workers=rows,
            bindings={str(original_plan): original_sha},
        ),
    )
    put(
        recovery / "GATE_history_segment_recovery.json",
        dict(
            source_sha=history.PARENT_SOURCE,
            historical_worker_source_sha=history.OLD_SHA,
            missing_shards=[],
            status="recovered_completed_segment_nine_metadata_not_freshness_certified",
            fresh_confirmation_authorized=False,
            identity_inventory_sha256=inventory_sha,
            receipt_bindings=receipts,
            worker_bindings=bindings,
        ),
    )
    monkeypatch.setattr(history, "WORKDIR", tmp_path)
    monkeypatch.setattr(history.scanner, "ROOTS", tuple(roots))
    monkeypatch.setattr(history, "clean_source", lambda *a: "e" * 40)
    monkeypatch.setattr(history, "cpu_gate", lambda path, sha: dict(status="passed"))
    monkeypatch.setattr(
        history, "bridge", lambda: dict(origin_source_sha=history.OLD_SHA, same=True)
    )
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return str(9000 + len(calls)) + "\n"

    monkeypatch.setattr(subprocess, "check_output", submit)
    return dict(root=root, recovery=recovery, old_root=old_root, cpu=tmp_path, calls=calls)


def register(fixture, **kwargs):
    return history.register(
        fixture["root"], cpu_gate_root=fixture["cpu"], parent_recovery=fixture["recovery"], **kwargs
    )


def meaningful_advance(state, roots, *, shard_index, shard_count, budget):
    assert shard_count == 8 and budget.maximum_bytes == 2 * 2**30
    assert state["frontier"][0]["position"] >= 10  # Never restart at roots.
    state["frontier"][0]["position"] += 1
    state["processed_files"] += 1
    return state


def finish_workers(fixture, number, monkeypatch, update=meaningful_advance):
    monkeypatch.setattr(history.scanner, "advance", update)
    for index in range(8):
        history.worker(fixture["root"], number, index)


def test_dry_default_checks_real_gate_without_submissions(fixture):
    result = register(fixture)
    assert result["status"] == "dry_registered_no_submissions"
    assert fixture["calls"] == []
    assert not (fixture["root"] / "segment-1").exists()
    with pytest.raises(ValueError, match="dry registration cannot scan"):
        history.worker(fixture["root"], 1, 0)
    with pytest.raises(ValueError, match="dry registration cannot advance"):
        history.advance(fixture["root"], 1)


def test_four_finite_segments_preserve_origin_errors_and_all_overlay_IDs(fixture, monkeypatch):
    register(fixture, execute=True)
    original = history.sha(fixture["old_root"] / "segment-9/workers/shard-0.json")
    for number in range(1, 5):
        finish_workers(fixture, number, monkeypatch)
        history.advance(fixture["root"], number)
    gate = history.read(fixture["root"] / "GATE_history_closure_v16.json")
    assert gate["stop_reason"] == "maximum_four_segments" and gate["last_segment"] == 4
    assert gate["pending_frontier_items"] == 8 and gate["original_and_new_coverage_errors"] == 8
    assert not gate["old_producer_relabelled"] and not gate["fresh_confirmation_authorized"]
    assert len(fixture["calls"]) == 8  # One worker array + afterany per finite segment.
    assert history.sha(fixture["old_root"] / "segment-9/workers/shard-0.json") == original
    inventory = history.read(fixture["root"] / "segment-4/combined_identity_inventory.json")
    assert "earlier-overlay-ID" in inventory["identities"]["sample_ids"]
    row = history.read(fixture["root"] / "segment-1/workers/shard-0.json")
    assert row["source_sha"] == "e" * 40 and row["parent_producer_source_sha"] == history.OLD_SHA
    assert row["origin_source_sha"] == history.OLD_SHA


def test_no_progress_stops_after_one_segment(fixture, monkeypatch):
    register(fixture, execute=True)
    finish_workers(fixture, 1, monkeypatch, update=lambda state, *a, **k: state)
    gate = history.advance(fixture["root"], 1)
    assert gate["stop_reason"] == "no_state_progress"
    assert len(fixture["calls"]) == 2


def test_exhausted_frontier_still_retains_unknown_coverage_and_external_video(fixture, monkeypatch):
    register(fixture, execute=True)

    def exhausted(state, *args, **kwargs):
        state.update(frontier=[], traversal_complete=True)
        return state

    finish_workers(fixture, 1, monkeypatch, exhausted)
    gate = history.advance(fixture["root"], 1)
    assert gate["frontier_exhausted"] and gate["original_and_new_coverage_errors"] == 8
    assert gate["external_video_journals_unresolved"] and not gate["fresh_confirmation_authorized"]
    assert len(fixture["calls"]) == 2


def test_missing_worker_cannot_launch_next_segment(fixture, monkeypatch):
    register(fixture, execute=True)
    monkeypatch.setattr(history.scanner, "advance", meaningful_advance)
    history.worker(fixture["root"], 1, 0)
    gate = history.advance(fixture["root"], 1)
    assert gate["stop_reason"] == "missing_or_failed_workers" and gate["missing_shards"] == list(
        range(1, 8)
    )
    assert len(fixture["calls"]) == 2


@pytest.mark.parametrize("kind", ["errors", "identity", "bindings", "restart"])
def test_dropped_history_proofs_or_IDs_cannot_be_promoted(fixture, monkeypatch, kind):
    register(fixture, execute=True)
    finish_workers(fixture, 1, monkeypatch)
    path = fixture["root"] / "segment-1/workers/shard-0.json"
    value = history.read(path)
    if kind == "errors":
        value["state"]["errors"] = []
    elif kind == "identity":
        value["state"]["sample_ids"] = []
    elif kind == "restart":
        value["state"]["seen_directories"] = []
    else:
        # Mismatched lineage is a binding failure even if the state hash is recomputed.
        value["parent_worker_sha256"] = "a" * 64
    value["state_sha256"] = history.digest(value["state"])
    put(path, value)
    with pytest.raises(ValueError):
        history.advance(fixture["root"], 1)
    assert not (fixture["root"] / "segment-2").exists()


def test_changed_parent_receipt_or_semantics_blocks_before_state_reuse(fixture, monkeypatch):
    register(fixture, execute=True)
    monkeypatch.setattr(history, "bridge", lambda: dict(changed=True))
    with pytest.raises(ValueError, match="policy changed"):
        history.worker(fixture["root"], 1, 0)


def test_missing_actual_recovery_gate_does_not_create_new_root(fixture):
    (fixture["recovery"] / "GATE_history_segment_recovery.json").unlink()
    with pytest.raises(FileNotFoundError):
        register(fixture, execute=True)
    assert fixture["calls"] == []


def test_parent_state_mutation_is_not_restarted_or_waived(fixture):
    register(fixture, execute=True)
    path = fixture["old_root"] / "segment-9/workers/shard-0.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        history.worker(fixture["root"], 1, 0)


def test_atomic_compact_checkpoint_is_equivalent_and_size_bounded(tmp_path):
    value = dict(state=dict(frontier=[dict(offset=123)], sample_ids=["consumed"]), source="old")
    target = tmp_path / "state.json"
    history.bounded_write(target, value, 1024, compact=True)
    assert history.read(target) == value
    assert history.digest(history.read(target)) == history.digest(value)
    with pytest.raises(FileExistsError):
        history.bounded_write(target, value, 1024, compact=True)
    with pytest.raises(ValueError, match="size exceeded"):
        history.bounded_write(tmp_path / "too-big.json", value, 1, compact=True)
    assert not (tmp_path / "too-big.json").exists()


@pytest.mark.parametrize(
    "option",
    ["--gres=gpu:1", "--gpus=1", "--partition=bii-gpu", "--array=0-7", "--dependency=afterany:1"],
)
def test_completion_hook_cannot_request_GPUs_or_override_dependencies(fixture, option):
    with pytest.raises(ValueError, match="CPU-only"):
        register(fixture, execute=True, completion_argv=["sbatch", option, "finish.sbatch"])


def test_terminal_completion_is_queued_once_after_current_callback(fixture, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    register(fixture, execute=True, completion_argv=["sbatch", "--partition=bii", "finish.sbatch"])
    finish_workers(fixture, 1, monkeypatch, update=lambda state, *a, **k: state)
    result = history.advance(fixture["root"], 1)
    assert result["completion"] and len(fixture["calls"]) == 3
    assert "--dependency=afterany:12345" in fixture["calls"][-1]
    with pytest.raises(FileExistsError):
        history.finish(fixture["root"], "repeat")


def test_uncertain_submission_preserves_known_worker_and_blocks_completion(fixture, monkeypatch):
    calls = []

    def unreliable(command, **kwargs):
        calls.append(command)
        return "4321\n" if len(calls) == 1 else "unacknowledged-response\n"

    monkeypatch.setattr(subprocess, "check_output", unreliable)
    gate = register(
        fixture, execute=True, completion_argv=["sbatch", "--partition=bii", "finish.sbatch"]
    )
    assert gate["scheduler_submission_uncertain"]
    assert gate["known_outstanding_worker_ids"] == ["4321"] and len(calls) == 2
    assert not gate["fresh_confirmation_authorized"]


def test_definite_analyzer_failure_completion_waits_known_worker_without_retry(
    fixture, monkeypatch
):
    calls = []

    def failed_analyzer(command, **kwargs):
        calls.append(command)
        # The request must be durable before a scheduler side effect.
        if len(calls) == 1:
            assert (fixture["root"] / "jobs/segment-1-request.json").exists()
            return "4321\n"
        if len(calls) == 2:
            assert (fixture["root"] / "jobs/after-segment-1-request.json").exists()
            raise subprocess.CalledProcessError(1, command, stderr="synthetic allocation refusal")
        return "4322\n"

    monkeypatch.setattr(subprocess, "check_output", failed_analyzer)
    result = register(
        fixture, execute=True, completion_argv=["sbatch", "--partition=bii", "finish.sbatch"]
    )
    assert result["gate"]["known_outstanding_worker_ids"] == ["4321"]
    assert not result["gate"]["scheduler_submission_uncertain"]
    assert result["completion"] == "4322" and len(calls) == 3
    assert "--dependency=afterany:4321" in calls[-1]


def test_string_false_is_not_execution_authorization(fixture):
    with pytest.raises(ValueError, match="boolean execution"):
        register(fixture, execute="False")
    assert fixture["calls"] == []
