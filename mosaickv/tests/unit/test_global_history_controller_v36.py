"""Synthetic V36 controller checks; no historical traversal or Slurm."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import evicastkv_global_history_v36 as driver
import pytest

from mosaickv.evaluation import coco_history_resume as prior
from mosaickv.evaluation.fox_campaign import digest, sha

REVISION = "b" * 40


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True)


def read(path, expected=None):
    path = Path(path)
    if expected is not None and sha(path) != expected:
        raise ValueError("synthetic immutable binding differs")
    return json.loads(path.read_bytes())


def owner_state(owner, count=0):
    directory = f"/synthetic/owner-{owner}"
    state = dict(
        frontier=[
            dict(kind="file", path=f"{directory}/pending-{i}.jsonl", offset=0) for i in range(count)
        ],
        errors=[dict(path=f"{directory}/old.yml", error="old preserved coverage error")],
        sample_ids=[],
        coco_ids=[],
        media_hashes=[],
        source_group_ids=[],
        scopes=[],
        scoped_identities=[],
        seen_files=[f"{directory}/old.json"],
        seen_directories=[directory],
        aliases=[],
        scheduler_logs_excluded=[],
        projection_pauses=[],
        bindings={f"{directory}/old.json": dict(signature=[])},
        directories={directory: []},
        processed_files=1,
        projected_records=1,
        binary_or_unstructured_files=0,
        verification_paths=None,
        verification_index=0,
        traversal_complete=False,
    )
    state = driver.core.normalize_seed(state)[0]
    if not count:
        state["verification_paths"] = sorted(state["directories"]) + sorted(state["bindings"])
        state["verification_index"] = len(state["verification_paths"])
        state["traversal_complete"] = True
    return state


def test_allocator_prioritizes_busy_owners_without_changing_ownership():
    counts = {0: 77, 1: 1685, 2: 0, 3: 0, 4: 0, 5: 1781, 6: 0, 7: 4938}
    allocation = driver.allocate(counts, slots=8)
    assert set(allocation) == {0, 1, 5, 7}
    assert sum(allocation.values()) == 8
    assert allocation[7] > allocation[0]
    assert all(value >= 1 for value in allocation.values())
    assert allocation == driver.allocate(dict(reversed(list(counts.items()))), slots=8)
    assert driver.allocate({0: 0, 1: 0}, slots=8) == {}


@pytest.mark.parametrize("counts", [{0: True}, {0: -1}, {0: "3"}])
def test_allocator_rejects_nonstrict_or_negative_counts(counts):
    with pytest.raises(ValueError):
        driver.allocate(counts)


def test_allocator_rejects_too_few_slots_for_original_owners():
    with pytest.raises(ValueError):
        driver.allocate({0: 1, 1: 1}, slots=1)


def test_import_preserves_all_old_exclusions_errors_maps_and_verification_state():
    base = owner_state(1, 1)
    after = deepcopy(base)
    after["sample_ids"].append("new-id")
    after["bindings"]["/synthetic/owner-1/new.json"] = dict(signature=[])
    after["processed_files"] += 1
    driver.preservation(base, after)
    for field, value in (
        ("errors", []),
        ("seen_files", []),
        ("bindings", {}),
        ("processed_files", 0),
        ("verification_index", 1),
    ):
        bad = deepcopy(after)
        bad[field] = value
        with pytest.raises(ValueError):
            driver.preservation(base, bad)


def setup_controller(monkeypatch, tmp_path, *, counts=None):
    root = tmp_path / "run"
    root.mkdir()
    counts = [1, 0, 0, 0, 0, 0, 0, 0] if counts is None else counts
    seeds = {}
    for owner in range(8):
        value = owner_state(owner, counts[owner])
        path = tmp_path / "seeds" / f"owner-{owner}.json"
        save(path, value)
        seeds[str(owner)] = dict(
            path=str(path), sha256=sha(path), container=False, state_sha256=digest(value)
        )
    plan = dict(
        source_sha=REVISION,
        seeds=seeds,
        roots=["/synthetic"],
        seed_error_counts={str(owner): 1 for owner in range(8)},
        preserved_resolved_error_sha256s=[],
    )
    monkeypatch.setattr(driver, "load", lambda _root: deepcopy(plan))
    monkeypatch.setattr(driver, "read", read)
    monkeypatch.setattr(driver, "put", save)
    monkeypatch.setattr(driver, "SLOTS", 2)
    monkeypatch.setattr(driver, "MAX_ROUNDS", 4)
    allocate = driver.allocate
    monkeypatch.setattr(driver, "allocate", lambda values: allocate(values, slots=2))
    return root, plan


def publish(root, number, index, *, failed=False, new_error=False, unchanged=False):
    folder = Path(root) / f"round-{number:03d}"
    spec = read(folder / "plan.json")
    task = spec["tasks"][index]
    if failed:
        return dict(index=index, returncode=1, timeout=False)
    base, _ = driver.state(spec["heads"][str(task["owner"])])
    if spec["phase"] == "traverse":
        queue = task["queue"]
        state = deepcopy(base)
        state["frontier"] = deepcopy(queue) if unchanged else []
        if not unchanged:
            for row in queue:
                state["bindings"][row["path"]] = dict(signature=row.get("signature"))
                state["seen_files"].append(row["path"])
                state["processed_files"] += 1
                state["projected_records"] += 1
        state["seen_files"].sort()
        if new_error:
            state["errors"].append(
                deepcopy(base["errors"][0])
                if new_error == "same-hash"
                else dict(path="/synthetic/new-unresolved.json", error="new error")
            )
        with driver.core._composition(task["owner"]):
            summary = prior._summary(base, queue, state)
        full = dict(
            owner_index=task["owner"],
            owner_count=8,
            base_state_sha256=digest(base),
            queue_sha256=digest(queue),
            all_scope_roots_sha256=spec["roots_sha256"],
            state=state,
            state_sha256=digest(state),
            summary=summary,
        )
        result = driver.core.compact_result(base, queue, full, owner_index=task["owner"])
    else:
        state = deepcopy(base)
        paths = sorted(base["directories"]) + sorted(base["bindings"])
        state.update(
            verification_paths=paths, verification_index=len(paths), traversal_complete=True
        )
        summary = driver.core._verification_summary(
            base, state, owner_index=task["owner"], calls=len(paths) - base["verification_index"]
        )
        result = dict(
            owner_index=task["owner"],
            owner_count=8,
            base_state_sha256=digest(base),
            state_sha256=digest(state),
            state=state,
            summary=summary,
        )
    save(folder / "workers" / f"task-{index}.json", dict(source_sha=REVISION, **result))
    return dict(index=index, returncode=0, timeout=False)


def assert_blocked(gate):
    for key in (
        "global_historical_inventory_complete",
        "complete_historical_id_inventory",
        "global_media_source_group_inventory_complete",
        "fresh_confirmation_authorized",
        "materialization_authorized",
    ):
        assert gate[key] is False
    assert gate["GPU_jobs_submitted"] == 0


def test_controller_uses_deltas_then_explicit_new_verification_for_every_owner(
    monkeypatch, tmp_path
):
    root, plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(driver, "run_task", publish)
    gate = driver.controller(root)
    assert gate["complete"] is True
    assert gate["remaining_pending_items"] == 0
    assert gate["current_time_verification_epoch_performed"] is True
    assert gate["historical_coverage_errors"] == 8
    assert gate["complete_declared_identity_field_scope"] is False
    assert_blocked(gate)
    for owner in range(8):
        before, _ = driver.state(plan["seeds"][str(owner)])
        epoch = read(root / "verification-epochs" / f"owner-{owner}.json")
        receipt = read(root / "verification-epochs" / f"owner-{owner}.receipt.json")
        assert epoch["verification_index"] == 0 and epoch["verification_paths"] is None
        assert receipt["original_verification_index"] == (0 if owner == 0 else 2)
        assert epoch["errors"] == before["errors"]
    verification_tasks = read(root / "round-001/plan.json")["tasks"]
    assert {task["owner"] for task in verification_tasks} == set(range(8))
    inventory = read(root / "combined_identity_inventory.json")
    assert len(inventory["errors"]) == 8


def test_failed_task_stops_preserving_seed_heads_without_verification_or_retry(
    monkeypatch, tmp_path
):
    root, plan = setup_controller(monkeypatch, tmp_path)
    calls = []

    def failed(path, number, index):
        calls.append((number, index))
        return publish(path, number, index, failed=True)

    monkeypatch.setattr(driver, "run_task", failed)
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "failed_execution_partition"
    assert gate["heads"] == plan["seeds"]
    assert not (root / "verification-epochs").exists()
    assert not (root / "round-001").exists()
    assert all(number == 0 for number, _index in calls)
    assert_blocked(gate)


def test_real_no_progress_is_detected_by_state_not_new_checkpoint_path(monkeypatch, tmp_path):
    root, _plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        driver, "run_task", lambda path, number, index: publish(path, number, index, unchanged=True)
    )
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "no_checkpoint_progress"
    assert not (root / "round-001").exists()
    assert_blocked(gate)


def test_new_errors_are_retained_and_do_not_become_a_freshness_pass(monkeypatch, tmp_path):
    root, _plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        driver,
        "run_task",
        lambda path, number, index: publish(
            path, number, index, new_error=number == 0 and index == 0
        ),
    )
    gate = driver.controller(root)
    assert gate["historical_coverage_errors"] == 9
    assert gate["complete_declared_identity_field_scope"] is False
    assert_blocked(gate)


def test_old_resolution_proof_never_waives_a_new_same_hash_error_occurrence(monkeypatch, tmp_path):
    root, plan = setup_controller(monkeypatch, tmp_path)
    plan["preserved_resolved_error_sha256s"] = [
        digest(driver.state(head)[0]["errors"][0]) for head in plan["seeds"].values()
    ]
    monkeypatch.setattr(
        driver,
        "run_task",
        lambda path, number, index: publish(
            path, number, index, new_error="same-hash" if number == 0 and index == 0 else False
        ),
    )
    gate = driver.controller(root)
    assert gate["historical_coverage_errors"] == 9
    assert gate["preserved_prior_resolutions"] == 8
    assert gate["net_unresolved_coverage_errors"] == 1
    assert gate["complete_declared_identity_field_scope"] is False
    assert_blocked(gate)


def test_resolution_classifier_uses_original_occurrence_not_owner_added_hash():
    error = dict(path="/synthetic/old.yml", error="original")
    covered = {digest(error)}
    rows = [
        dict(
            owner_index=0,
            historical_error_ordinal=index,
            original_error_sha256=digest(error),
            **error,
        )
        for index in range(2)
    ]
    assert driver.unresolved_errors(rows, covered, {"0": 1}) == rows[1:]
    assert driver.unresolved_errors(rows, set(), {"0": 1}) == rows


def test_wall_bound_stops_without_launching_a_worker(monkeypatch, tmp_path):
    root, _plan = setup_controller(monkeypatch, tmp_path)
    times = iter([0.0, driver.MAX_WALL, driver.MAX_WALL + 1])
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setattr(driver, "run_task", lambda *_: pytest.fail("no launch after wall bound"))
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["rounds_completed"] == 0
    assert gate["remaining_pending_items"] == 1
    assert_blocked(gate)


def test_owner_state_rejects_changed_hash_and_preserves_normalization(monkeypatch, tmp_path):
    _root, plan = setup_controller(monkeypatch, tmp_path)
    head = deepcopy(plan["seeds"]["0"])
    value, normalization = driver.state(head)
    assert normalization["added_fields"] == {}
    assert len(value["frontier"]) == 1
    head["state_sha256"] = "forged"
    with pytest.raises(ValueError, match="state digest"):
        driver.state(head)


def test_registration_imports_completed_owners_without_repeating_the_506_roots(
    monkeypatch, tmp_path
):
    import evicastkv_review_followups as followups

    from mosaickv.evaluation import coco_history_subset as subset

    class ScratchPath(type(Path())):
        def is_relative_to(self, *others):
            if others == (Path("/scratch/djy8hg/workdir").resolve(),):
                return super().is_relative_to(tmp_path)
            return super().is_relative_to(*others)

    parent, original = tmp_path / "parent", {}
    bindings = {}
    old_counts = [77, 1694, 0, 0, 0, 1781, 0, 5450]
    for owner, count in enumerate(old_counts):
        value = owner_state(owner, count)
        path = parent / "workers" / f"shard-{owner}.json"
        save(path, dict(index=owner, state=value, state_sha256=digest(value)))
        original[owner] = value
        bindings[str(path.resolve())] = sha(path)
    save(parent / "decision.json", dict(worker_bindings=bindings))
    one = tmp_path / "one" / "workers/shard-1.json"
    imported1 = deepcopy(original[1])
    imported1["frontier"] = imported1["frontier"][9:]
    save(one, dict(state=imported1, summary=dict(synthetic_valid=True)))
    seven = tmp_path / "v35/round-04/merged-state.json"
    imported7 = deepcopy(original[7])
    imported7["frontier"] = imported7["frontier"][512:]
    save(seven, imported7)
    save(seven.parent / "decision.json", dict(state_file_sha256=sha(seven)))
    gate = seven.parents[1] / "GATE_history_resume_v35.json"
    save(gate, dict(reviewed_pending_items=0, complete=True))
    cpu = tmp_path / "cpu"
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        save(cpu / "full-cpu" / name, dict(synthetic_fixture=True))
    proof = tmp_path / "resolution-proof.json"
    save(proof, dict(original_resolutions_preserved=True))
    historical = tmp_path / "historical"
    historical.mkdir()
    monkeypatch.setattr(driver, "Path", ScratchPath)
    monkeypatch.setattr(driver, "PARENT", parent)
    monkeypatch.setattr(driver, "DECISION_SHA", sha(parent / "decision.json"))
    monkeypatch.setattr(driver, "ONE", one)
    monkeypatch.setattr(driver, "ONE_SHA", sha(one))
    monkeypatch.setattr(driver, "SEVEN", seven)
    monkeypatch.setattr(driver, "V35", seven.parents[1])
    monkeypatch.setattr(driver, "V35_GATE_SHA", sha(gate))
    monkeypatch.setattr(driver, "clean_source", lambda *_: REVISION)
    monkeypatch.setattr(driver, "read", read)
    monkeypatch.setattr(driver, "source_bridge", lambda: dict(unchanged=True))
    monkeypatch.setattr(
        driver, "modules", lambda: (SimpleNamespace(ROOTS=(historical,)), None, None)
    )
    monkeypatch.setattr(followups, "cpu_gate", lambda *_: dict(passed=True))
    monkeypatch.setattr(
        driver,
        "saved_resolution",
        lambda: dict(
            bindings={str(proof): sha(proof)},
            covered_error_hashes=["synthetic-proof-hash"],
            original_source_sha="c" * 40,
        ),
    )

    def split(_record, owner):
        return dict(deferred=[dict(item=item) for item in original[owner]["frontier"][512:]])

    def verify(record, _split, imported, _summary):
        assert imported["frontier"] == record["state"]["frontier"][9:]

    monkeypatch.setattr(subset, "split_frontier", split)
    monkeypatch.setattr(subset, "verify_continuation", verify)
    destination = tmp_path / "registered"
    result = driver.register(destination, cpu)
    plan = read(destination / "registration.json")
    assert result["pending_items"] == 8481 and result["GPU_jobs"] == 0
    assert plan["pending_by_owner"] == driver.EXPECTED_PENDING
    assert plan["seeds"]["1"]["path"] == str(one)
    assert plan["seeds"]["7"]["path"] == str(seven)
    assert plan["seeds"]["7"]["container"] is False
    assert plan["bindings"][str(proof)] == sha(proof)
    assert plan["seed_error_counts"] == {str(owner): 1 for owner in range(8)}
    assert plan["preserved_resolved_error_sha256s"] == ["synthetic-proof-hash"]
    assert plan["current_time_verification_not_yet_performed"] is True
    for owner, head in plan["seeds"].items():
        value, _ = driver.state(head)
        assert value["errors"] == original[int(owner)]["errors"]


@pytest.mark.parametrize("prepare_failure", [True, False])
def test_closeout_preserves_prepare_failure_and_unresolved_parquet_after_cpu_wave(
    monkeypatch, tmp_path, prepare_failure
):
    root, plan = setup_controller(monkeypatch, tmp_path)
    plan["cpu_gate_root"] = str(tmp_path / "cpu")
    commands = []
    gate_name = "GATE_synthetic_coverage.json"

    def prepare(output, **_kwargs):
        if prepare_failure:
            raise ValueError("synthetic missing preparation evidence")
        Path(output).mkdir()
        return dict(registered_files=8)

    def analyze(output):
        gate = dict(
            missing_shards=[],
            pending_files=0,
            blocked_files=3,
            net_unresolved_after=3,
            net_unresolved_categories={"Parquet has no safe identity columns": 3},
            fresh_confirmation_authorized=False,
        )
        save(Path(output) / gate_name, gate)
        return gate

    coverage = SimpleNamespace(prepare=prepare, analyze=analyze, GATE=gate_name)
    media = SimpleNamespace(
        recover=lambda **_: dict(media_source_groups_certified=False), publish=save
    )

    def import_module(name):
        if name == "evicastkv_coverage_resolution_v36":
            return coverage
        if name == "evicastkv_media_recovery_v36":
            return media
        pytest.fail("missing qualified proofs must not invoke confirmation admission")

    def run(command, **kwargs):
        if prepare_failure:
            pytest.fail("preparation failure must not run workers")
        commands.append((command, kwargs))
        assert kwargs["timeout"] == 420 and kwargs["check"] is False
        assert command[2] == "worker"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver, "importlib", SimpleNamespace(import_module=import_module))
    monkeypatch.setattr(driver.subprocess, "run", run)
    result = driver.closeout(root)
    assert result["GPU_confirmation_jobs_submitted"] == 0
    assert result["fresh_confirmation_authorized"] is False
    assert result["materialization_authorized"] is False
    assert result["media_source_groups_certified"] is False
    assert result["status"] == "blocked_missing_or_negative_confirmation_prerequisites"
    assert set(result["missing_qualified_proofs"]) == {
        "correctness",
        "metric",
        "byte_matching",
        "baseline_fidelity",
        "history",
        "media",
    }
    recovery = result["additive_coverage_recovery"]
    if prepare_failure:
        assert recovery["complete"] is False and recovery["phase"] == "prepare"
        assert recovery["error_waivers"] == 0
        assert (root / "review/coverage-infrastructure-error.json").is_file()
        assert commands == []
    else:
        assert len(commands) == 8
        assert {int(command[-1]) for command, _kwargs in commands} == set(range(8))
        assert recovery["gate"]["net_unresolved_after"] == 3
        assert recovery["complete"] is False
        assert recovery["fresh_confirmation_authorized"] is False
        assert (root / "review/coverage-wave.json").is_file()
    assert (root / "review/closeout.json").is_file()
