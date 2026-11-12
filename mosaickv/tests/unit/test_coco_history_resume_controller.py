"""Synthetic controller/provenance tests; no scheduler or historical scan."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import evicastkv_coco_history_resume as driver
import pytest

from mosaickv.evaluation.fox_campaign import digest, sha

REVISION = "a" * 40


def state_fixture():
    state = dict(
        frontier=[
            dict(kind="directory", path="/deferred/keep"),
            dict(kind="file", path="/synthetic/a.jsonl", offset=10),
            dict(kind="directory", path="/synthetic/b"),
        ],
        sample_ids=[],
        coco_ids=[],
        media_hashes=[],
        source_group_ids=[],
        scopes=[],
        scoped_identities=[],
        seen_files=[],
        seen_directories=[],
        errors=[],
        aliases=[],
        projection_pauses=[],
        scheduler_logs_excluded=[],
        bindings={},
        directories={},
        processed_files=0,
        projected_records=0,
        binary_or_unstructured_files=0,
        verification_paths=None,
        verification_index=0,
        traversal_complete=False,
    )
    return state


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True)


def read(path, expected=None):
    path = Path(path)
    if expected is not None and sha(path) != expected:
        raise ValueError("synthetic immutable input changed")
    return json.loads(path.read_bytes())


def setup_controller(monkeypatch, tmp_path):
    root = tmp_path / "execution"
    root.mkdir()
    base = state_fixture()
    seed = tmp_path / "seed.json"
    save(seed, dict(state=base))
    scopes = [row["path"] for row in base["frontier"][1:]]
    plan = dict(
        source_sha=REVISION,
        original_scope_roots=scopes,
        original_scope_sha256=digest(scopes),
        roots=["/synthetic"],
    )
    monkeypatch.setattr(driver, "load", lambda _root: deepcopy(plan))
    monkeypatch.setattr(driver, "SEED", seed)
    monkeypatch.setattr(driver, "SEED_SHA", sha(seed))
    monkeypatch.setattr(driver, "DEFERRED", 1)
    monkeypatch.setattr(driver, "INITIAL_PENDING", 2)
    monkeypatch.setattr(driver, "PARTS", 2)
    monkeypatch.setattr(driver, "MAX_ROUNDS", 2)
    monkeypatch.setattr(driver, "read", read)
    monkeypatch.setattr(driver, "put", save)
    return root, base, plan


def publish_result(root, number, index, *, action="consume", mutate=None):
    round_root = Path(root) / f"round-{number:02d}"
    plan = read(round_root / "plan.json")
    base = read(plan["base_path"], plan["base_file_sha256"])
    queue = plan["queues"][index]
    state = deepcopy(base)
    state["frontier"] = deepcopy(queue)
    if action == "consume":
        for item in queue:
            name = item["path"]
            if item["kind"] == "file":
                state["bindings"][name] = {"sha256": "synthetic"}
                state["seen_files"].append(name)
                state["processed_files"] += 1
                state["projected_records"] += 1
            else:
                state["directories"][name] = item.get("signature")
                state["seen_directories"].append(name)
        state["frontier"] = []
    elif action == "progress":
        for item in state["frontier"]:
            cursor = "offset" if item["kind"] == "file" else "position"
            item[cursor] = item.get(cursor, 0) + 1
            if item["kind"] == "directory":
                item["prefix_sha256"] = f"prefix-{number}"
    result = dict(
        source_sha=REVISION,
        original_owner=7,
        execution_partition=index,
        all_scope_roots_sha256=plan["all_scope_roots_sha256"],
        base_state_sha256=digest(base),
        queue_sha256=digest(queue),
        state=state,
        state_sha256=digest(state),
        summary=driver.core._summary(base, queue, state),
    )
    if mutate is not None:
        mutate(result)
    save(round_root / "workers" / f"part-{index}.json", result)
    return dict(index=index, returncode=0, timeout=False, wall_seconds=0)


def assert_no_freshness(gate):
    assert gate["scope_traversal_only"] is True
    for key in (
        "global_verification_performed",
        "historical_identity_media_coverage_complete",
        "globally_fresh",
        "fresh_confirmation_authorized",
        "materialization_authorized",
    ):
        assert gate[key] is False


def test_controller_exhausts_only_reviewed_scope_preserving_deferred_and_global_blocks(
    monkeypatch, tmp_path
):
    root, base, _plan = setup_controller(monkeypatch, tmp_path)
    calls = []

    def run(path, number, index):
        calls.append((number, index))
        return publish_result(path, number, index)

    monkeypatch.setattr(driver, "run_process", run)
    gate = driver.controller(root)
    assert sorted(calls) == [(0, 0), (0, 1)]
    assert gate["complete"] is True and gate["reviewed_pending_items"] == 0
    merged = read(root / "round-00/merged-state.json")
    assert merged["frontier"] == base["frontier"][:1]
    assert merged["verification_paths"] is None and merged["verification_index"] == 0
    assert merged["traversal_complete"] is False
    assert_no_freshness(gate)


@pytest.mark.parametrize(
    "status", [dict(returncode=1, timeout=False), dict(returncode=None, timeout=True)]
)
def test_failed_partition_stops_without_attempting_to_read_missing_results(
    monkeypatch, tmp_path, status
):
    root, base, _plan = setup_controller(monkeypatch, tmp_path)
    calls = []

    def failed(_root, number, index):
        calls.append((number, index))
        return dict(index=index, wall_seconds=1, **status)

    monkeypatch.setattr(driver, "run_process", failed)
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "failed_execution_partition"
    assert gate["reviewed_pending_items"] == len(base["frontier"]) - 1
    assert sorted(calls) == [(0, 0), (0, 1)]
    assert not (root / "round-01").exists()
    assert_no_freshness(gate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "b" * 40),
        ("original_owner", 0),
        ("execution_partition", 9),
        ("all_scope_roots_sha256", "forged"),
    ],
)
def test_invalid_result_producer_scope_or_partition_fails_closed(
    monkeypatch, tmp_path, field, value
):
    root, _base, _plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        driver,
        "run_process",
        lambda path, number, index: publish_result(
            path, number, index, mutate=lambda result: result.update({field: value})
        ),
    )
    with pytest.raises(ValueError, match="binding"):
        driver.controller(root)
    assert not (root / "GATE_history_resume_v35.json").exists()
    assert not (root / "round-01").exists()


def test_controller_stops_at_no_progress_without_rescanning_the_same_checkpoint(
    monkeypatch, tmp_path
):
    root, _base, _plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        driver,
        "run_process",
        lambda path, number, index: publish_result(path, number, index, action="unchanged"),
    )
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "no_checkpoint_progress"
    assert not (root / "round-01").exists()
    assert_no_freshness(gate)


def test_controller_resumes_saved_cursors_and_stops_at_registered_round_bound(
    monkeypatch, tmp_path
):
    root, base, plan = setup_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        driver,
        "run_process",
        lambda path, number, index: publish_result(path, number, index, action="progress"),
    )
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "registered_round_or_wall_bound"
    first, second = read(root / "round-00/merged-state.json"), read(root / "round-01/base.json")
    assert first == second
    pending = read(root / "round-01/merged-state.json")["frontier"][1:]
    assert pending[0]["offset"] == base["frontier"][1]["offset"] + 2
    assert (
        read(root / "round-01/plan.json")["all_scope_roots_sha256"] == plan["original_scope_sha256"]
    )
    assert not (root / "round-02").exists()
    assert_no_freshness(gate)


def test_elapsed_wall_bound_stops_before_launching_another_round(monkeypatch, tmp_path):
    root, _base, _plan = setup_controller(monkeypatch, tmp_path)
    times = iter([0.0, driver.MAX_WALL, driver.MAX_WALL + 1])
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setattr(
        driver, "run_process", lambda *_: pytest.fail("wall bound must stop launch")
    )
    gate = driver.controller(root)
    assert gate["complete"] is False and gate["reason"] == "registered_round_or_wall_bound"
    assert not (root / "round-00").exists()
    assert_no_freshness(gate)


@pytest.mark.parametrize("timeout", [False, True])
def test_subprocess_wrapper_has_finite_420_second_envelope_and_write_once_status(
    monkeypatch, tmp_path, timeout
):
    root = tmp_path / "execution"
    path = root / "round-00"
    for name in ("logs", "jobs"):
        (path / name).mkdir(parents=True, exist_ok=False)
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if timeout:
            raise driver.subprocess.TimeoutExpired(command, timeout=420)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    result = driver.run_process(root, 0, 1)
    assert result["timeout"] is timeout
    assert commands[0][1]["timeout"] == 420
    assert commands[0][0][2] == "worker"
    assert "--index" in commands[0][0] and commands[0][0][-1] == "1"
    assert read(path / "jobs/worker-1.json")["timeout"] is timeout
    with pytest.raises(FileExistsError):
        driver.run_process(root, 0, 1)


def load_fixture(monkeypatch, tmp_path):
    plan = dict(
        source_sha=REVISION,
        policy=deepcopy(driver.POLICY),
        additive_bridge={"additions": []},
        source_bridge={"bridge": "original"},
        original_scope_roots=["/synthetic"],
        original_scope_sha256=digest(["/synthetic"]),
        bindings={},
    )
    monkeypatch.setattr(driver, "read", lambda _path: deepcopy(plan))
    monkeypatch.setattr(driver, "clean_source", lambda *_: REVISION)
    monkeypatch.setattr(driver, "additive_bridge", lambda: {"additions": []})
    monkeypatch.setattr(
        driver,
        "dependencies",
        lambda: (None, None, None, SimpleNamespace(bridge=lambda: {"bridge": "original"})),
    )
    return plan, tmp_path


def test_loader_accepts_only_hash_bound_scope(monkeypatch, tmp_path):
    plan, root = load_fixture(monkeypatch, tmp_path)
    assert driver.load(root)["original_scope_roots"] == ["/synthetic"]
    plan["original_scope_roots"].append("/expanded")
    with pytest.raises(ValueError, match="scope"):
        driver.load(root)


def test_loader_rejects_mutated_prior_control_binding(monkeypatch, tmp_path):
    plan, root = load_fixture(monkeypatch, tmp_path)
    bound = tmp_path / "prior-plan.json"
    save(bound, dict(unchanged=True))
    plan["bindings"][str(bound)] = sha(bound)
    assert driver.load(root)["bindings"]
    bound.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="binding"):
        driver.load(root)


@pytest.mark.parametrize("timeout_first", [False, True])
def test_afterany_closeout_is_bounded_metadata_only_and_never_authorizes_confirmation(
    monkeypatch, tmp_path, timeout_first
):
    root, _base, _plan = setup_controller(monkeypatch, tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if timeout_first and len(commands) == 1:
            raise driver.subprocess.TimeoutExpired(command, kwargs["timeout"])
        output = Path(command[command.index("--output") + 1])
        save(output, dict(derived_metadata=True, fresh_confirmation_authorized=False))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    report = driver.closeout(root)
    assert len(commands) == 2
    assert [kwargs["timeout"] for _command, kwargs in commands] == [240, 60]
    assert all(kwargs["check"] is False for _command, kwargs in commands)
    assert "--history-report" in commands[1][0]
    assert report["complete"] is (not timeout_first)
    assert report["no_GPU_work"] is True
    assert report["no_dataset_or_media_materialization"] is True
    assert report["fresh_confirmation_authorized"] is False
    assert report["materialization_authorized"] is False
    assert len(report["reports"]) == (1 if timeout_first else 2)
    assert (root / "review/closeout.json").is_file()
    with pytest.raises(FileExistsError):
        driver.closeout(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_GPU_workers", 1),
        ("globally_fresh", True),
        ("fresh_confirmation_authorized", True),
        ("materialization_authorized", True),
        ("globally_fresh", 0),
    ],
)
def test_loader_rejects_permission_or_policy_type_substitution(monkeypatch, tmp_path, field, value):
    plan, root = load_fixture(monkeypatch, tmp_path)
    plan["policy"][field] = value
    with pytest.raises(ValueError, match="policy"):
        driver.load(root)


def test_registration_binds_actual_prior_plan_worker1_and_original_controls(monkeypatch, tmp_path):
    # Only the synthetic temporary root is accepted as a stand-in for scratch.
    class ScratchPath(type(Path())):
        def is_relative_to(self, *others):
            if others == (Path("/scratch/djy8hg/workdir").resolve(),):
                return super().is_relative_to(tmp_path)
            return super().is_relative_to(*others)

    monkeypatch.setattr(driver, "Path", ScratchPath)
    base = state_fixture()
    prior_root = tmp_path / "prior"
    seed = prior_root / "workers/shard-7.json"
    save(seed, dict(state=base, state_sha256=digest(base)))
    gate = prior_root / "GATE_COCO_history_subset.json"
    save(gate, dict(scope_traversal_only=True, globally_fresh=False))
    worker1 = prior_root / "workers/shard-1.json"
    save(worker1, dict(synthetic_preserved_worker=True))
    historical = tmp_path / "historical"
    historical.mkdir()
    bound = tmp_path / "unchanged-control.json"
    save(bound, dict(immutable=True))
    prior = dict(
        roots=[str(historical.resolve())],
        source_sha=driver.BASE,
        original_owner_indices=[1, 7],
        original_shard_count=8,
        execute=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        empirical_admission=dict(
            whole_request_engineering_gate_passed=True, COCO_boundary_gate_passed=True
        ),
        bindings={str(bound): sha(bound)},
    )
    save(prior_root / "plan.json", prior)
    cpu = tmp_path / "cpu"
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        save(cpu / "full-cpu" / name, dict(synthetic_pass=True))
    monkeypatch.setattr(driver, "SEED", seed)
    monkeypatch.setattr(driver, "SEED_SHA", sha(seed))
    monkeypatch.setattr(driver, "SEED_STATE_SHA", digest(base))
    monkeypatch.setattr(driver, "PRIOR_GATE", gate)
    monkeypatch.setattr(driver, "DEFERRED", 1)
    monkeypatch.setattr(driver, "INITIAL_PENDING", 2)
    monkeypatch.setattr(driver, "PARTS", 2)
    monkeypatch.setattr(driver, "clean_source", lambda *_: REVISION)
    monkeypatch.setattr(driver, "read", read)
    monkeypatch.setattr(driver, "additive_bridge", lambda: dict(additions=[]))
    monkeypatch.setattr(
        driver,
        "dependencies",
        lambda: (
            None,
            SimpleNamespace(ROOTS=(historical,)),
            None,
            SimpleNamespace(bridge=lambda: dict(unchanged=True)),
        ),
    )
    monkeypatch.setattr(
        driver,
        "importlib",
        SimpleNamespace(
            import_module=lambda _name: SimpleNamespace(cpu_gate=lambda *_: dict(passed=True))
        ),
    )
    root = tmp_path / "registered"
    result = driver.register(root, cpu)
    plan = read(root / "registration.json")
    assert result["GPU_jobs"] == 0
    assert plan["bindings"][str(prior_root / "plan.json")] == sha(prior_root / "plan.json")
    assert plan["bindings"][str(worker1)] == sha(worker1)
    assert plan["bindings"][str(seed)] == sha(seed)
    assert plan["successful_worker1_preserved"] == str(worker1)
    assert plan["roots"] == prior["roots"]
    assert plan["original_scope_sha256"] == digest(plan["original_scope_roots"])
    assert plan["policy"]["globally_fresh"] is False
    with pytest.raises(ValueError, match="new immutable"):
        driver.register(root, cpu)
