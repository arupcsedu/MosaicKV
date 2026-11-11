"""Empty Arrow projection equivalence and exact finished-V11-state continuation."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_freshness_empty_parquet_continue as continuation
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return continuation.sha(path)


@pytest.fixture
def packet(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    path = root / "edges.parquet"
    pq.write_table(
        pa.table(
            dict(
                source=list(range(1025)),
                target=list(range(1025)),
                weight=[0.5] * 1025,
                edge_type=["graph"] * 1025,
            )
        ),
        path,
        row_group_size=1025,
    )
    state = continuation.scanner.initial_state((root,))
    state["frontier"] = []
    item = dict(kind="file", path=str(path), signature=continuation.scanner.signature(path))
    return path, state, item


def budget():
    return continuation.scanner.Budget(seconds=60, maximum_bytes=32 * 2**20)


def test_empty_schema_identity_and_binding_equivalence_without_empty_dictionary_rows(packet):
    path, state, item = packet
    old, new = copy.deepcopy(state), copy.deepcopy(state)
    assert continuation.scanner._process_file(copy.deepcopy(item), old, budget())
    with patch.object(continuation.scanner, "_parquet", continuation.empty_projection_parquet):
        assert continuation.scanner._process_file(copy.deepcopy(item), new, budget())
    for key in continuation.IDENTITIES:
        assert old[key] == new[key]
    assert old["projected_records"] == 1025 and new["projected_records"] == 0
    assert old["errors"] == new["errors"]
    left, right = (
        copy.deepcopy(old["bindings"][str(path)]),
        copy.deepcopy(new["bindings"][str(path)]),
    )
    left.pop("projected_batches")
    right.pop("projected_batches")
    assert left == right
    assert new["empty_parquet_projection"][str(path)]["rows_not_enumerated"] == 1025


def test_partial_legacy_row_cursor_preserves_old_counter_errors_and_batch_digests(packet):
    path, state, item = packet
    item.update(columns=[], row_group=0, batch=1, projected_batches=["legacy-bound-batch"])
    state["projected_records"] = 512
    continuation.scanner.error(
        state, path, "Parquet has no safely projectable identity metadata columns"
    )
    assert continuation.empty_projection_parquet(path, item, state, budget())
    assert state["projected_records"] == 512 and len(state["errors"]) == 1
    assert item["projected_batches"] == ["legacy-bound-batch"]
    assert state["empty_parquet_projection"][str(path)]["rows_not_enumerated"] == 513


def test_partial_file_hash_segments_resume_without_restarting_rows_or_source(packet, monkeypatch):
    path, state, item = packet
    monkeypatch.setattr(continuation.scanner, "CHUNK", 64)
    finite = SimpleNamespace(bytes=0)
    finite.available = lambda: finite.bytes < 64
    assert not continuation.empty_projection_parquet(path, item, state, finite)
    assert item["hash_offset"] == 64
    prefix = copy.deepcopy(item["segments"])
    assert continuation.empty_projection_parquet(path, item, state, budget())
    assert item["segments"][0] == prefix[0]
    raw = path.read_bytes()
    assert item["segments"][-1]["end"] == len(raw)
    assert all(
        hashlib.sha256(raw[s["start"] : s["end"]]).hexdigest() == s["sha256"]
        for s in item["segments"]
    )
    assert state["projected_records"] == 0


def test_saved_hash_gap_is_not_tolerated(packet):
    path, state, item = packet
    item.update(hash_offset=10, segments=[])
    with pytest.raises(ValueError, match="hash cursor"):
        continuation.empty_projection_parquet(path, item, state, budget())


def test_source_change_prevents_completed_binding(packet, monkeypatch):
    path, state, item = packet
    original = continuation.scanner._stat_guard
    calls = []

    def guard(*args):
        calls.append(1)
        if len(calls) == 3:  # process-file entry, Parquet entry, final check
            with path.open("ab") as stream:
                stream.write(b"mutated")
        return original(*args)

    monkeypatch.setattr(continuation.scanner, "_stat_guard", guard)
    with (
        patch.object(continuation.scanner, "_parquet", continuation.empty_projection_parquet),
        pytest.raises(ValueError, match="source changed"),
    ):
        continuation.scanner._process_file(item, state, budget())
    assert str(path) not in state["bindings"]


def test_nonempty_safe_projection_delegates_to_exact_original_path(tmp_path, monkeypatch):
    path = tmp_path / "ids.parquet"
    pq.write_table(pa.table(dict(sample_id=["known-ID"], response=["FORBIDDEN"])), path)
    state = continuation.scanner.initial_state((tmp_path,))
    item = dict(signature=continuation.scanner.signature(path))
    calls = []

    def original(*args):
        calls.append(args)
        return "unchanged-path"

    monkeypatch.setattr(continuation, "ORIGINAL_PARQUET", original)
    assert continuation.empty_projection_parquet(path, item, state, budget()) == "unchanged-path"
    assert len(calls) == 1 and "columns" not in item


@pytest.fixture
def prepared(tmp_path, packet, monkeypatch):
    path, base, item = packet
    parent, root = tmp_path / "V11", tmp_path / "V12"
    put(
        parent / "plan.json",
        dict(
            source_sha="a" * 40,
            roots=[str(path.parent)],
            same_scanner_file_sha256=continuation.sha(Path(continuation.scanner.__file__)),
        ),
    )
    workers = {}
    for index in range(8):
        state = copy.deepcopy(base)
        state["sample_ids"] = [f"old-shard-{index}"]
        if index == 0:
            state["frontier"] = [copy.deepcopy(item)]
        target = parent / "workers" / f"shard-{index:03d}.json"
        workers[str(target)] = put(
            target, dict(source_sha="a" * 40, state=state, state_sha256=continuation.digest(state))
        )
    identities = {k: [] for k in continuation.IDENTITIES}
    identities["sample_ids"] = ["previous-oversize-recovery-must-not-be-lost"]
    inventory_sha = put(parent / "combined_identity_inventory.json", dict(identities=identities))
    put(
        parent / "GATE_freshness_frontier_fork.json",
        dict(
            source_sha="a" * 40,
            missing_shards=[],
            identity_inventory_sha256=inventory_sha,
            worker_bindings=workers,
        ),
    )
    monkeypatch.setattr(continuation, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(continuation, "cpu_gate", lambda *args: dict(status="passed"))
    continuation.prepare(root, cpu_gate_root=tmp_path, parent=parent)
    return root, parent


def test_prepare_copies_exact_V11_states_not_CP005(prepared):
    root, parent = prepared
    plan = continuation.load(root)
    assert plan["parent_state_generation"] == "V11_finished_worker_states_not_CP005"
    for index in range(8):
        old = continuation.read(parent / "workers" / f"shard-{index:03d}.json")
        new = continuation.read(root / "imports" / f"shard-{index}.json")
        assert old["state_sha256"] == new["state_sha256"]


def test_continuation_preserves_all_prior_recovered_identity_and_negative_errors(prepared):
    root, parent = prepared
    old_sha = continuation.sha(parent / "workers/shard-000.json")
    for index in range(8):
        continuation.worker(root, index)
    gate = continuation.analyze(root)
    assert gate["frontier_exhausted"] and gate["empty_projection_rows_not_enumerated"] == 1025
    assert gate["newly_projected_records"] == 0
    inventory = continuation.read(root / "combined_identity_inventory.json")
    assert "previous-oversize-recovery-must-not-be-lost" in inventory["identities"]["sample_ids"]
    assert (
        not gate["fresh_confirmation_authorized"] and gate["original_and_new_coverage_errors"] == 1
    )
    assert continuation.sha(parent / "workers/shard-000.json") == old_sha


def test_missing_continuation_workers_are_explicit(prepared):
    root, _parent = prepared
    gate = continuation.analyze(root)
    assert gate["missing_shards"] == list(range(8)) and not gate["frontier_exhausted"]


def test_grouped_branch_union_keeps_old_and_new_recovered_ids(prepared):
    root, parent = prepared
    recovery = root.parent / "grouped"
    put(recovery / "plan.json", dict(parent=str(parent)))
    identities = {key: [] for key in continuation.IDENTITIES}
    identities["sample_ids"] = ["new-grouped-offset-ID"]
    inventory_sha = put(recovery / "combined_identity_inventory.json", dict(identities=identities))
    put(
        recovery / "GATE_grouped_offset_recovery.json",
        dict(
            source_sha="b" * 40,
            identity_inventory_sha256=inventory_sha,
            status="bounded_grouped_recovery_incomplete",
        ),
    )
    continuation.analyze(root, recovery_root=recovery)
    merged = continuation.read(root / "combined_identity_inventory.json")["identities"][
        "sample_ids"
    ]
    assert "previous-oversize-recovery-must-not-be-lost" in merged
    assert "new-grouped-offset-ID" in merged


def test_YAML_and_grouped_union_keep_worker_bindings_separate_from_branch_proofs(prepared):
    root, parent = prepared
    branches = []
    for name, gate_name in (
        ("yaml", "GATE_freshness_yaml_events.json"),
        ("offsets", "GATE_grouped_offset_recovery.json"),
    ):
        branch = root.parent / name
        put(branch / "plan.json", dict(parent=str(parent)))
        identities = {key: [] for key in continuation.IDENTITIES}
        identities["sample_ids"] = [f"recovered-{name}"]
        actual = put(branch / "combined_identity_inventory.json", dict(identities=identities))
        put(
            branch / gate_name,
            dict(
                source_sha="b" * 40,
                identity_inventory_sha256=actual,
                status="bounded-branch-evidence",
            ),
        )
        branches.append(branch)
    result = continuation.analyze(root, recovery_root=branches[1], yaml_root=branches[0])
    observed = continuation.read(root / "combined_identity_inventory.json")
    assert set(observed["identities"]["sample_ids"]) == {
        "recovered-yaml",
        "recovered-offsets",
        "previous-oversize-recovery-must-not-be-lost",
    }
    assert result["worker_bindings"] == {}  # All workers remain missing in this fixture.
    assert len(result["recovery_bindings"]) == 6
    assert result["YAML_event_recovery_status"] == "bounded-branch-evidence"


def test_V11_to_V12_to_V12_imports_latest_exact_states_without_restart(prepared):
    first, _original = prepared
    for index in range(8):
        continuation.worker(first, index)
    continuation.analyze(first)
    second = first.parent / "second-segment"
    continuation.prepare(second, cpu_gate_root=first.parent, parent=first)
    plan = continuation.load(second)
    assert plan["parent_state_generation"] == "continued_finished_worker_states_no_restart"
    assert plan["same_scanner_file_sha256"] == continuation.sha(Path(continuation.scanner.__file__))
    for index in range(8):
        before = continuation.read(first / "workers" / f"shard-{index}.json")
        imported = continuation.read(second / "imports" / f"shard-{index}.json")
        assert before["state_sha256"] == imported["state_sha256"]
        continuation.worker(second, index)
    gate = continuation.analyze(second)
    assert gate["newly_processed_files"] == 0 and gate["frontier_exhausted"]
    ids = continuation.read(second / "combined_identity_inventory.json")["identities"]["sample_ids"]
    assert "previous-oversize-recovery-must-not-be-lost" in ids
    assert not plan["automatic_resubmission"] and plan["segment_count"] == 1


@pytest.mark.parametrize("kind", ["source", "scanner", "index", "gate"])
def test_invalid_parent_source_index_or_gate_fails_closed(prepared, kind):
    first, parent = prepared
    gate_path = parent / "GATE_freshness_frontier_fork.json"
    if kind == "source":
        gate = continuation.read(gate_path)
        gate["source_sha"] = "0" * 40
        put(gate_path, gate)
    elif kind == "scanner":
        path = parent / "plan.json"
        plan = continuation.read(path)
        plan["same_scanner_file_sha256"] = "0" * 64
        put(path, plan)
    elif kind == "index":
        gate = continuation.read(gate_path)
        old = parent / "workers/shard-007.json"
        new = parent / "workers/shard-008.json"
        actual = put(new, continuation.read(old))
        del gate["worker_bindings"][str(old)]
        gate["worker_bindings"][str(new)] = actual
        put(gate_path, gate)
    with pytest.raises(ValueError):
        continuation.prepare(
            first.parent / f"bad-{kind}",
            cpu_gate_root=first.parent,
            parent=parent,
            parent_gate_filename="../unexpected.json" if kind == "gate" else None,
        )
