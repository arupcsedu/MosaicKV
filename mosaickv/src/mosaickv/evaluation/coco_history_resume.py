"""Resumable, partitioned CPU continuation of an already reviewed COCO queue.

Execution partitions never change the historical scanner's owner (7 of 8).
They cannot certify global freshness.  Completed work is represented by the
immutable base state; only its pending tail is eligible for another round.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import digest

SECONDS = 210
MAX_BYTES = 2 * 2**30
LINE_BYTES = 8 * 2**20
OWNER_INDEX = 7
OWNER_COUNT = 8

SET_FIELDS = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
    "seen_files",
    "seen_directories",
)
NESTED_FIELDS = {"scopes", "scoped_identities"}
APPEND_FIELDS = ("errors", "aliases", "projection_pauses", "scheduler_logs_excluded")
DICT_FIELDS = ("bindings", "directories")
COUNTER_FIELDS = ("processed_files", "projected_records", "binary_or_unstructured_files")
CONTROL_FIELDS = {"frontier", "verification_paths", "verification_index", "traversal_complete"}
KNOWN_FIELDS = set(SET_FIELDS + APPEND_FIELDS + DICT_FIELDS + COUNTER_FIELDS) | CONTROL_FIELDS


class ExecutionPartitionFenceError(BaseException):
    """A valid global target belongs to another CPU execution partition.

    This deliberately is not an Exception: the historical scanner catches
    ordinary errors into its coverage ledger.  A scheduling-fence violation
    must instead abort publication, not manufacture a historical input error.
    """


def _path(value):
    if not isinstance(value, str):
        raise ValueError("absolute canonical frontier path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("absolute normalized frontier path required")
    return path


class _Fence:
    """Constant-size ancestor walk, not a linear scan of every scope root."""

    def __init__(self, directories=(), files=()):
        self.directories = {str(_path(str(value))) for value in directories}
        self.files = {str(_path(str(value))) for value in files}

    def contains(self, value):
        path = _path(str(value))
        if str(path) in self.files or str(path) in self.directories:
            return True
        return any(str(parent) in self.directories for parent in path.parents)


def _queue(queue):
    if not isinstance(queue, list):
        raise ValueError("an explicit exact pending queue is required")
    paths = set()
    for item in queue:
        if not isinstance(item, dict) or item.get("kind") not in {"file", "directory"}:
            raise ValueError("pending entries must preserve file/directory kinds")
        path = str(_path(item.get("path")))
        if path in paths:
            raise ValueError("duplicate pending path")
        paths.add(path)
    return queue


def _queue_fence(queue):
    return _Fence(
        (item["path"] for item in queue if item["kind"] == "directory"),
        (item["path"] for item in queue if item["kind"] == "file"),
    )


def _values(state, name):
    values = state[name]
    if not isinstance(values, list):
        raise ValueError(f"{name} must remain an explicit list")
    if name in NESTED_FIELDS:
        if any(not isinstance(value, (tuple, list)) for value in values):
            raise ValueError(f"{name} requires complete metadata tuples")
        result = {tuple(value) for value in values}
    else:
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"{name} requires metadata strings")
        result = set(values)
    if len(result) != len(values):
        raise ValueError(f"{name} contains duplicate exclusions")
    return result


def _base(base):
    if not isinstance(base, dict) or not isinstance(base.get("frontier"), list):
        raise ValueError("a complete immutable scanner base is required")
    for name in SET_FIELDS:
        _values(base, name)
    for name in APPEND_FIELDS:
        if not isinstance(base.get(name), list):
            raise ValueError(f"complete original {name} list required")
    for name in DICT_FIELDS:
        if not isinstance(base.get(name), dict):
            raise ValueError(f"complete original {name} mapping required")
    for name in COUNTER_FIELDS:
        if type(base.get(name)) is not int or base[name] < 0:
            raise ValueError(f"nonnegative strict integer {name} required")
    if type(base.get("verification_index")) is not int or base["verification_index"] < 0:
        raise ValueError("original verification cursor required")
    if base.get("verification_paths") is not None and not isinstance(
        base["verification_paths"], list
    ):
        raise ValueError("original verification paths required")
    if base.get("traversal_complete") is not False:
        raise ValueError("a restricted queue cannot start from a global completion claim")


def partition(base_state, queues, deferred_count=4938, parts=8):
    """Partition exactly the saved pending tail; never repartition ownership.

    ``queues`` is the pending entry list, not a list of worker lists.  Related
    ancestor/descendant entries stay together, which also makes later rounds
    safe if a paused directory and already-enumerated children coexist.
    """
    _base(base_state)
    _queue(queues)
    if type(parts) is not int or not 1 <= parts <= 64:
        raise ValueError("one through 64 bounded execution partitions required")
    if type(deferred_count) is not int or not 0 <= deferred_count <= len(base_state["frontier"]):
        raise ValueError("strict original deferred count required")
    deferred = base_state["frontier"][:deferred_count]
    if base_state["frontier"][deferred_count:] != queues:
        raise ValueError("pending input must exactly equal the saved frontier tail")
    paths = {item["path"] for item in queues}
    file_paths = {item["path"] for item in queues if item["kind"] == "file"}
    if any(str(parent) in file_paths for path in paths for parent in _path(path).parents):
        raise ValueError("a pending file cannot contain another pending entry")
    minimal = {
        name for name in paths if not any(str(parent) in paths for parent in _path(name).parents)
    }
    roots_in_order, root_for, observed_roots = [], {}, set()
    for item in queues:
        path = _path(item["path"])
        root = next(
            str(candidate) for candidate in (path, *path.parents) if str(candidate) in minimal
        )
        root_for[item["path"]] = root
        if root not in observed_roots:
            roots_in_order.append(root)
            observed_roots.add(root)
    assignments = {root: index % parts for index, root in enumerate(roots_in_order)}
    output = [[] for _ in range(parts)]
    for item in queues:
        output[assignments[root_for[item["path"]]]].append(copy.deepcopy(item))
    return dict(
        queues=output,
        deferred=copy.deepcopy(deferred),
        all_scope_roots=roots_in_order,
        base_state_sha256=digest(base_state),
        pending_frontier_sha256=digest(queues),
        deferred_frontier_sha256=digest(deferred),
        original_shard_index=OWNER_INDEX,
        original_shard_count=OWNER_COUNT,
        execution_partitions=parts,
    )


def _summary(base, queue, state):
    return dict(
        original_shard_index=OWNER_INDEX,
        original_shard_count=OWNER_COUNT,
        initial_pending_items=len(queue),
        remaining_pending_items=len(state["frontier"]),
        partition_frontier_exhausted=not state["frontier"],
        newly_processed_files=state["processed_files"] - base["processed_files"],
        newly_projected_records=state["projected_records"] - base["projected_records"],
        new_errors=len(state["errors"]) - len(base["errors"]),
        maximum_seconds=SECONDS,
        maximum_source_bytes=MAX_BYTES,
        global_verification_performed=False,
        global_historical_inventory_complete=False,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )


def advance_partition(base, queue, all_scope_roots, roots, scanner, empty_projection_parquet):
    """Advance only one execution queue with unchanged historical semantics."""
    _base(base)
    _queue(queue)
    if not isinstance(all_scope_roots, (list, tuple)) or not all_scope_roots:
        raise ValueError("bound full reviewed scope required")
    names = [str(_path(str(value))) for value in all_scope_roots]
    if len(names) != len(set(names)):
        raise ValueError("duplicate full scope roots")
    global_fence, local_fence = _Fence(names), _queue_fence(queue)
    if any(not global_fence.contains(item["path"]) for item in queue):
        raise ValueError("execution queue escapes the reviewed full scope")
    state = copy.deepcopy(base)
    state["frontier"] = copy.deepcopy(queue)
    # An exhausted restricted queue must not burn its budget verifying every
    # historical file.  These fields are restored before result publication.
    state["verification_paths"], state["verification_index"] = [], 0
    old_within = scanner.within
    already_seen = set(base["seen_files"]) | set(base["seen_directories"])

    def within(path, actual_roots):
        if not old_within(path, actual_roots) or not global_fence.contains(path):
            return False
        if not local_fence.contains(path):
            # The unchanged directory walker checks seen paths before enqueue.
            # A previously recorded canonical alias adds no foreign coverage.
            if str(path) in already_seen:
                return True
            raise ExecutionPartitionFenceError(
                "canonical target belongs outside this execution partition"
            )
        return True

    with (
        patch.object(scanner, "within", within),
        patch.object(scanner, "_parquet", empty_projection_parquet),
        patch.object(scanner, "MAX_LINE", LINE_BYTES),
    ):
        scanner.advance(
            state,
            tuple(roots),
            shard_index=OWNER_INDEX,
            shard_count=OWNER_COUNT,
            budget=scanner.Budget(seconds=SECONDS, maximum_bytes=MAX_BYTES),
        )
    state["verification_paths"] = copy.deepcopy(base["verification_paths"])
    state["verification_index"] = base["verification_index"]
    state["traversal_complete"] = False
    result = dict(
        base_state_sha256=digest(base),
        queue_sha256=digest(queue),
        all_scope_roots_sha256=digest(names),
        state=state,
        state_sha256=digest(state),
        summary=_summary(base, queue, state),
    )
    verify_result(base, queue, result)
    return result


def verify_result(base, queue, result):
    """Independently reject state loss, rewriting, escapes and invented passes."""
    _base(base)
    _queue(queue)
    if not isinstance(result, dict) or not isinstance(result.get("state"), dict):
        raise ValueError("complete partition state required")
    state = result["state"]
    if (
        result.get("base_state_sha256") != digest(base)
        or result.get("queue_sha256") != digest(queue)
        or result.get("state_sha256") != digest(state)
        or set(state) != set(base)
    ):
        raise ValueError("bound base, queue, state digest or complete field set changed")
    _queue(state["frontier"])
    fence = _queue_fence(queue)
    if any(not fence.contains(item["path"]) for item in state["frontier"]):
        raise ValueError("resumed frontier escaped its execution partition")
    original_by_path = {item["path"]: item for item in base["frontier"]}
    if any(original_by_path.get(item["path"]) != item for item in queue):
        raise ValueError("execution queue rewrote an original pending item")
    pending_by_path = {item["path"]: item for item in state["frontier"]}
    for item in queue:
        pending = pending_by_path.get(item["path"])
        if pending is None:
            receipts = state["bindings"] if item["kind"] == "file" else state["directories"]
            failed_paths = {
                row.get("path")
                for row in state["errors"][len(base["errors"]) :]
                if isinstance(row, dict)
            }
            if item["path"] not in receipts and item["path"] not in failed_paths:
                raise ValueError("pending item disappeared without a processing receipt")
        else:
            if any(pending.get(key) != item.get(key) for key in ("kind", "path", "signature")):
                raise ValueError("pending path/kind/signature changed during continuation")
            cursor = "offset" if item["kind"] == "file" else "position"
            if type(pending.get(cursor, 0)) is not int or pending.get(cursor, 0) < item.get(
                cursor, 0
            ):
                raise ValueError("pending file/directory cursor moved backwards")
            if pending.get("segments", [])[: len(item.get("segments", []))] != item.get(
                "segments", []
            ):
                raise ValueError("pending file segment prefix was rewritten")
            if (
                item["kind"] == "directory"
                and pending.get("position", 0) == item.get("position", 0)
                and pending.get("prefix_sha256") != item.get("prefix_sha256")
            ):
                raise ValueError("unchanged directory cursor lost its prefix hash")
    if (
        state["verification_paths"] != base["verification_paths"]
        or type(state["verification_index"]) is not int
        or state["verification_index"] != base["verification_index"]
        or state["traversal_complete"] is not False
    ):
        raise ValueError("global verification cursor/completion changed")
    for name in SET_FIELDS:
        old, new = _values(base, name), _values(state, name)
        if not old <= new:
            raise ValueError(f"original {name} exclusions were removed")
        if name in {"seen_files", "seen_directories"} and any(
            not fence.contains(value) for value in new - old
        ):
            raise ValueError("new seen path escaped its execution partition")
    for name in APPEND_FIELDS:
        new, old = state[name], base[name]
        if not isinstance(new, list) or new[: len(old)] != old:
            raise ValueError(f"original {name} prefix was rewritten")
    for name in DICT_FIELDS:
        old, new = base[name], state[name]
        if not isinstance(new, dict) or any(
            key not in new or new[key] != value for key, value in old.items()
        ):
            raise ValueError(f"original {name} mapping was removed or rewritten")
        if any(not fence.contains(key) for key in new.keys() - old.keys()):
            raise ValueError(f"new {name} path escaped its execution partition")
    for name in COUNTER_FIELDS:
        if type(state[name]) is not int or state[name] < base[name]:
            raise ValueError(f"original {name} counter decreased or changed type")
    for name in set(base) - KNOWN_FIELDS:
        if state[name] != base[name]:
            raise ValueError(f"unknown scanner state field mutated: {name}")
    expected = _summary(base, queue, state)
    if result.get("summary") != expected or any(
        type(result["summary"].get(key)) is not type(value) for key, value in expected.items()
    ):
        raise ValueError("partition summary differs from independent reduction")
    return True


def merge(base, partition_queues, results, deferred):
    """Merge one complete round; preserve all old/deferred state exactly once."""
    _base(base)
    if (
        not isinstance(partition_queues, list)
        or not isinstance(results, list)
        or len(partition_queues) != len(results)
        or not partition_queues
        or not isinstance(deferred, list)
        or base["frontier"][: len(deferred)] != deferred
    ):
        raise ValueError("complete indexed execution round and exact deferred prefix required")
    expected = base["frontier"][len(deferred) :]
    flattened = [item for queue in partition_queues for item in _queue(queue)]
    if len(flattened) != len(expected):
        raise ValueError("execution partition omitted or duplicated a pending item")
    expected_by_path = {item["path"]: item for item in _queue(expected)}
    if len({item["path"] for item in flattened}) != len(flattened) or any(
        item != expected_by_path.get(item["path"]) for item in flattened
    ):
        raise ValueError("execution queues do not exactly partition the saved pending tail")
    for queue, result in zip(partition_queues, results, strict=True):
        verify_result(base, queue, result)
    fences = [_queue_fence(queue) for queue in partition_queues]
    for index, queue in enumerate(partition_queues):
        if any(
            fence.contains(item["path"])
            for other, fence in enumerate(fences)
            if other != index
            for item in queue
        ):
            raise ValueError("ancestor-related queue entries cross execution partitions")
    output = copy.deepcopy(base)
    for name in SET_FIELDS:
        values = _values(base, name)
        owned_new = set()
        old = set(values)
        for result in results:
            additions = _values(result["state"], name) - old
            if name in {"seen_files", "seen_directories"} and owned_new & additions:
                raise ValueError("new seen paths were processed by multiple partitions")
            owned_new.update(additions)
            values.update(additions)
        output[name] = (
            [list(value) for value in sorted(values)] if name in NESTED_FIELDS else sorted(values)
        )
    for name in APPEND_FIELDS:
        for result in results:
            output[name].extend(copy.deepcopy(result["state"][name][len(base[name]) :]))
    for name in DICT_FIELDS:
        new_owners = set()
        for result in results:
            additions = result["state"][name].keys() - base[name].keys()
            if new_owners & additions:
                raise ValueError("mapping path was constructed by multiple partitions")
            new_owners.update(additions)
            for key in additions:
                output[name][key] = copy.deepcopy(result["state"][name][key])
    for name in COUNTER_FIELDS:
        output[name] = base[name] + sum(result["state"][name] - base[name] for result in results)
    output["frontier"] = copy.deepcopy(deferred) + [
        copy.deepcopy(item) for result in results for item in result["state"]["frontier"]
    ]
    _queue(output["frontier"])
    output["verification_paths"] = copy.deepcopy(base["verification_paths"])
    output["verification_index"] = base["verification_index"]
    output["traversal_complete"] = False
    return output
