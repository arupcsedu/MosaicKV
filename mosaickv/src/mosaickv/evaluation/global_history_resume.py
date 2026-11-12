"""Owner-preserving global historical continuation and explicit verification.

This composes the V35 checkpoint validator without modifying historical code.
Execution partitions do not change the original eight-way source ownership.
Traversal/verification completion is never a global freshness certificate.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation import coco_history_resume as prior
from mosaickv.evaluation.fox_campaign import digest

SECONDS, MAX_BYTES, LINE_BYTES = prior.SECONDS, prior.MAX_BYTES, prior.LINE_BYTES
OWNER_COUNT = 8
EXTRA_MAP = "empty_parquet_projection"
EXCLUDED_CREDENTIAL_ERROR = "credential body excluded pending closed producer proof"
_ORIGINAL_BASE = prior._base
_OWNER_LOCK = threading.Lock()


def _credential_body(path):
    """Classify only named Codex credential bodies, without opening them."""
    lexical = Path(path)

    def protected(value):
        return value.name in {"auth.json", "credentials.json"} and ".codex" in value.parts[:-1]

    if protected(lexical):
        return True
    try:
        canonical = lexical.resolve(strict=True)
    except (OSError, RuntimeError):
        # Ordinary source/signature failure remains the scanner's own error.
        return False
    return protected(canonical)


def _owner(index):
    if type(index) is not int or not 0 <= index < OWNER_COUNT:
        raise ValueError("original strict owner index 0 through 7 required")
    return index


def normalize_seed(state):
    """Record additive optional-field normalization; never reset old cursors."""
    if not isinstance(state, dict):
        raise ValueError("complete scanner state required")
    normalized = copy.deepcopy(state)
    added = {}
    for name, value in (("projection_pauses", []), (EXTRA_MAP, {}), ("traversal_complete", False)):
        if name not in normalized:
            normalized[name] = copy.deepcopy(value)
            added[name] = copy.deepcopy(value)
    _validate_base(normalized)
    return normalized, dict(
        original_state_sha256=digest(state),
        normalized_state_sha256=digest(normalized),
        added_fields=added,
        original_fields_unchanged=True,
        original_verification_cursor_unchanged=True,
        globally_fresh=False,
    )


def _validate_base(base):
    if not isinstance(base, dict) or not isinstance(base.get(EXTRA_MAP), dict):
        raise ValueError("explicit normalized empty-Parquet metadata map required")
    if base.get("traversal_complete") is True:
        if base.get("frontier"):
            raise ValueError("completed original owner cannot have pending traversal")
        shallow = dict(base)
        shallow["traversal_complete"] = False
        _ORIGINAL_BASE(shallow)
    else:
        _ORIGINAL_BASE(base)


@contextmanager
def _composition(owner_index):
    """Guard process-local overrides; concurrent threads must use processes."""
    owner_index = _owner(owner_index)
    if not _OWNER_LOCK.acquire(blocking=False):
        raise RuntimeError("owner composition is process-isolated, not concurrently threaded")
    try:
        with (
            patch.object(prior, "OWNER_INDEX", owner_index),
            patch.object(prior, "_base", _validate_base),
            patch.object(prior, "DICT_FIELDS", (*prior.DICT_FIELDS, EXTRA_MAP)),
            patch.object(prior, "KNOWN_FIELDS", prior.KNOWN_FIELDS | {EXTRA_MAP}),
        ):
            yield
    finally:
        _OWNER_LOCK.release()


def partition(base_state, queues, deferred_count=0, parts=8, *, owner_index):
    with _composition(owner_index):
        return prior.partition(base_state, queues, deferred_count, parts)


def advance_partition(
    base, queue, all_scope_roots, roots, scanner, empty_projection_parquet, *, owner_index
):
    """Use the unchanged scanner, retaining its original owner-skip semantics."""
    with _composition(owner_index):
        _validate_base(base)
        prior._queue(queue)
        names = [str(prior._path(str(value))) for value in all_scope_roots]
        if not names or len(names) != len(set(names)):
            raise ValueError("distinct unchanged global roots required")
        actual_roots = tuple(Path(value) for value in roots)
        if names != [str(path) for path in actual_roots]:
            raise ValueError("global scope must equal unchanged historical scanner roots")
        global_fence, local_fence = prior._Fence(names), prior._queue_fence(queue)
        if any(not global_fence.contains(item["path"]) for item in queue):
            raise ValueError("pending queue escapes unchanged historical roots")
        state = copy.deepcopy(base)
        state["frontier"] = copy.deepcopy(queue)
        state["verification_paths"], state["verification_index"] = [], 0
        already_seen = set(base["seen_files"]) | set(base["seen_directories"])
        old_within = scanner.within
        original_process = getattr(scanner, "_process_file", None)

        def privacy_process(item, actual_state, budget):
            if _credential_body(item["path"]):
                scanner.error(actual_state, Path(item["path"]), EXCLUDED_CREDENTIAL_ERROR)
                # This is a negative coverage receipt, not a processed file,
                # identity projection, content hash, or closed producer proof.
                return True
            if original_process is None:
                raise RuntimeError("unchanged historical file processor is unavailable")
            return original_process(item, actual_state, budget)

        def within(path, supplied_roots):
            if not old_within(path, supplied_roots) or not global_fence.contains(path):
                return False
            # The unchanged directory walker next checks source ownership and
            # skips this target.  Do not turn that skip into a coverage error.
            if scanner.owner(path, supplied_roots, OWNER_COUNT) != owner_index:
                return True
            if local_fence.contains(path) or str(path) in already_seen:
                return True
            raise prior.ExecutionPartitionFenceError(
                "unseen same-owner canonical target belongs to another execution partition"
            )

        with (
            patch.object(scanner, "within", within),
            patch.object(scanner, "_parquet", empty_projection_parquet),
            patch.object(scanner, "MAX_LINE", LINE_BYTES),
            patch.object(scanner, "_process_file", privacy_process, create=True),
        ):
            scanner.advance(
                state,
                actual_roots,
                shard_index=owner_index,
                shard_count=OWNER_COUNT,
                budget=scanner.Budget(seconds=SECONDS, maximum_bytes=MAX_BYTES),
            )
        state["verification_paths"] = copy.deepcopy(base["verification_paths"])
        state["verification_index"] = base["verification_index"]
        state["traversal_complete"] = False
        result = dict(
            owner_index=owner_index,
            owner_count=OWNER_COUNT,
            base_state_sha256=digest(base),
            queue_sha256=digest(queue),
            all_scope_roots_sha256=digest(names),
            state=state,
            state_sha256=digest(state),
            summary=prior._summary(base, queue, state),
        )
        prior.verify_result(base, queue, result)
        return result


def verify_result(base, queue, result, *, owner_index):
    with _composition(owner_index):
        if (
            type(result.get("owner_index")) is not int
            or result["owner_index"] != owner_index
            or type(result.get("owner_count")) is not int
            or result["owner_count"] != OWNER_COUNT
        ):
            raise ValueError("result changed original source ownership")
        return prior.verify_result(base, queue, result)


def merge(base, partition_queues, results, deferred, *, owner_index):
    for queue, result in zip(partition_queues, results, strict=True):
        verify_result(base, queue, result, owner_index=owner_index)
    with _composition(owner_index):
        return prior.merge(base, partition_queues, results, deferred)


def compact_result(base, queue, result, *, owner_index):
    """Transmit only audited additions; retain the reconstructed full hash."""
    verify_result(base, queue, result, owner_index=owner_index)
    state = result["state"]
    sets = {}
    for name in prior.SET_FIELDS:
        additions = prior._values(state, name) - prior._values(base, name)
        sets[name] = (
            [list(value) for value in sorted(additions)]
            if name in prior.NESTED_FIELDS
            else sorted(additions)
        )
    body = dict(
        schema_version=1,
        owner_index=owner_index,
        owner_count=OWNER_COUNT,
        base_state_sha256=result["base_state_sha256"],
        queue_sha256=result["queue_sha256"],
        all_scope_roots_sha256=result["all_scope_roots_sha256"],
        state_sha256=result["state_sha256"],
        summary=copy.deepcopy(result["summary"]),
        frontier=copy.deepcopy(state["frontier"]),
        set_additions=sets,
        map_additions={
            name: {
                key: copy.deepcopy(value)
                for key, value in state[name].items()
                if key not in base[name]
            }
            for name in (*prior.DICT_FIELDS, EXTRA_MAP)
        },
        append_tails={
            name: copy.deepcopy(state[name][len(base[name]) :]) for name in prior.APPEND_FIELDS
        },
        counter_deltas={name: state[name] - base[name] for name in prior.COUNTER_FIELDS},
    )
    return dict(**body, delta_sha256=digest(body))


def expand_result(base, queue, delta, *, owner_index):
    """Reconstruct and revalidate a worker delta against its exact base."""
    _owner(owner_index)
    expected = {
        "schema_version",
        "owner_index",
        "owner_count",
        "base_state_sha256",
        "queue_sha256",
        "all_scope_roots_sha256",
        "state_sha256",
        "summary",
        "frontier",
        "set_additions",
        "map_additions",
        "append_tails",
        "counter_deltas",
        "delta_sha256",
    }
    if not isinstance(delta, dict) or set(delta) != expected:
        raise ValueError("strict complete delta schema required")
    body = {key: value for key, value in delta.items() if key != "delta_sha256"}
    if (
        type(delta["schema_version"]) is not int
        or delta["schema_version"] != 1
        or delta["delta_sha256"] != digest(body)
        or delta["base_state_sha256"] != digest(base)
        or delta["queue_sha256"] != digest(queue)
        or type(delta["owner_index"]) is not int
        or delta["owner_index"] != owner_index
        or type(delta["owner_count"]) is not int
        or delta["owner_count"] != OWNER_COUNT
    ):
        raise ValueError("delta source owner/base/queue/hash changed")
    state = copy.deepcopy(base)
    for name, fields in (
        ("set_additions", prior.SET_FIELDS),
        ("map_additions", (*prior.DICT_FIELDS, EXTRA_MAP)),
        ("append_tails", prior.APPEND_FIELDS),
        ("counter_deltas", prior.COUNTER_FIELDS),
    ):
        if not isinstance(delta[name], dict) or set(delta[name]) != set(fields):
            raise ValueError("delta omitted or added a scanner field")
    for name in prior.SET_FIELDS:
        additions = prior._values({name: delta["set_additions"][name]}, name)
        old = prior._values(base, name)
        if old & additions:
            raise ValueError("delta repeats a previously recorded exclusion")
        values = old | additions
        state[name] = (
            [list(value) for value in sorted(values)]
            if name in prior.NESTED_FIELDS
            else sorted(values)
        )
    for name in (*prior.DICT_FIELDS, EXTRA_MAP):
        additions = delta["map_additions"][name]
        if not isinstance(additions, dict) or base[name].keys() & additions.keys():
            raise ValueError("delta rewrites an existing immutable mapping")
        state[name].update(copy.deepcopy(additions))
    for name in prior.APPEND_FIELDS:
        if not isinstance(delta["append_tails"][name], list):
            raise ValueError("append-only metadata tail required")
        state[name].extend(copy.deepcopy(delta["append_tails"][name]))
    for name in prior.COUNTER_FIELDS:
        change = delta["counter_deltas"][name]
        if type(change) is not int or change < 0:
            raise ValueError("strict nonnegative counter delta required")
        state[name] += change
    state["frontier"] = copy.deepcopy(delta["frontier"])
    state["traversal_complete"] = False
    result = {
        key: copy.deepcopy(delta[key])
        for key in (
            "owner_index",
            "owner_count",
            "base_state_sha256",
            "queue_sha256",
            "all_scope_roots_sha256",
            "state_sha256",
            "summary",
        )
    }
    result["state"] = state
    verify_result(base, queue, result, owner_index=owner_index)
    return result


def begin_verification_epoch(base):
    """Explicit immutable transition, preserving old verification in a receipt.

    The caller must seal both states and this receipt.  This is the only API
    that resets a historical completed cursor for a NEW current-time pass.
    """
    _validate_base(base)
    if base["frontier"]:
        raise ValueError("current verification requires exhausted traversal")
    state = copy.deepcopy(base)
    state.update(verification_paths=None, verification_index=0, traversal_complete=False)
    receipt = dict(
        phase="new_current_time_verification_epoch",
        previous_state_sha256=digest(base),
        epoch_state_sha256=digest(state),
        original_verification_paths_sha256=digest(base["verification_paths"]),
        original_verification_paths_count=len(base["verification_paths"] or []),
        original_verification_paths_were_none=base["verification_paths"] is None,
        original_verification_index=base["verification_index"],
        original_traversal_complete=base["traversal_complete"],
        old_state_must_remain_immutable=True,
        globally_fresh=False,
    )
    verify_epoch(base, state, receipt)
    return state, receipt


def verify_epoch(base, state, receipt):
    expected = dict(
        phase="new_current_time_verification_epoch",
        previous_state_sha256=digest(base),
        epoch_state_sha256=digest(state),
        original_verification_paths_sha256=digest(base["verification_paths"]),
        original_verification_paths_count=len(base["verification_paths"] or []),
        original_verification_paths_were_none=base["verification_paths"] is None,
        original_verification_index=base["verification_index"],
        original_traversal_complete=base["traversal_complete"],
        old_state_must_remain_immutable=True,
        globally_fresh=False,
    )
    if (
        base["frontier"]
        or set(base) != set(state)
        or digest(receipt) != digest(expected)
        or state["verification_paths"] is not None
        or type(state["verification_index"]) is not int
        or state["verification_index"] != 0
        or state["traversal_complete"] is not False
        or any(
            state[key] != value
            for key, value in base.items()
            if key not in {"verification_paths", "verification_index", "traversal_complete"}
        )
    ):
        raise ValueError("verification epoch rewrote original evidence or lost its prior cursor")
    return True


def _verification_summary(base, state, *, owner_index, calls):
    start, stop = base["verification_index"], state["verification_index"]
    return dict(
        owner_index=owner_index,
        owner_count=OWNER_COUNT,
        actual_stat_guard_calls=calls,
        newly_verified_paths=stop - start,
        verified_sequence_sha256=digest(state["verification_paths"][start:stop]),
        verification_paths=len(state["verification_paths"]),
        verification_index=stop,
        verification_exhausted=state["traversal_complete"],
        new_errors=len(state["errors"]) - len(base["errors"]),
        maximum_seconds=SECONDS,
        maximum_source_bytes=MAX_BYTES,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )


def advance_verification(base, roots, scanner, *, owner_index):
    """Stat the saved owned paths, never rediscover files or reset the cursor."""
    _owner(owner_index)
    _validate_base(base)
    if base["frontier"] or base["traversal_complete"] is not False:
        raise ValueError("an explicitly opened, traversal-exhausted verification epoch is required")
    expected_paths = sorted(base["directories"]) + sorted(base["bindings"])
    if base["verification_paths"] is not None and base["verification_paths"] != expected_paths:
        raise ValueError("saved verification paths differ from exact owned mappings")
    state = copy.deepcopy(base)
    start, calls = base["verification_index"], 0
    original_guard = scanner._stat_guard

    def observed_guard(path, expected):
        nonlocal calls
        if start + calls >= len(expected_paths) or str(path) != expected_paths[start + calls]:
            raise prior.ExecutionPartitionFenceError("verification changed exact saved path order")
        calls += 1
        return original_guard(path, expected)

    with patch.object(scanner, "_stat_guard", observed_guard):
        scanner.advance(
            state,
            tuple(Path(path) for path in roots),
            shard_index=owner_index,
            shard_count=OWNER_COUNT,
            budget=scanner.Budget(seconds=SECONDS, maximum_bytes=MAX_BYTES),
        )
    result = dict(
        owner_index=owner_index,
        owner_count=OWNER_COUNT,
        base_state_sha256=digest(base),
        state=state,
        state_sha256=digest(state),
        summary=_verification_summary(base, state, owner_index=owner_index, calls=calls),
    )
    verify_verification(base, result, owner_index=owner_index)
    return result


def verify_verification(base, result, *, owner_index):
    _owner(owner_index)
    _validate_base(base)
    state = result.get("state")
    if (
        not isinstance(state, dict)
        or base["frontier"]
        or state.get("frontier")
        or result.get("base_state_sha256") != digest(base)
        or result.get("state_sha256") != digest(state)
        or set(state) != set(base)
        or type(result.get("owner_index")) is not int
        or result["owner_index"] != owner_index
        or type(result.get("owner_count")) is not int
        or result["owner_count"] != OWNER_COUNT
    ):
        raise ValueError("verification source/base/state ownership changed")
    paths = sorted(base["directories"]) + sorted(base["bindings"])
    start, stop = base["verification_index"], state["verification_index"]
    if (
        base["verification_paths"] not in (None, paths)
        or state["verification_paths"] != paths
        or type(stop) is not int
        or not 0 <= start <= stop <= len(paths)
        or (base["verification_paths"] is None and start != 0)
        or state["traversal_complete"] is not (stop == len(paths))
    ):
        raise ValueError("verification cursor/path/completion is not genuine")
    for key, value in base.items():
        if (
            key not in {"errors", "verification_paths", "verification_index", "traversal_complete"}
            and state[key] != value
        ):
            raise ValueError("verification changed traversal or identity evidence")
    if state["errors"][: len(base["errors"])] != base["errors"]:
        raise ValueError("verification removed a historical coverage error")
    checked = set(paths[start:stop])
    if any(
        not isinstance(row, dict) or row.get("path") not in checked
        for row in state["errors"][len(base["errors"]) :]
    ):
        raise ValueError("verification error was not produced on a checked saved path")
    expected = _verification_summary(base, state, owner_index=owner_index, calls=stop - start)
    if result.get("summary") != expected or any(
        type(result["summary"].get(key)) is not type(value) for key, value in expected.items()
    ):
        raise ValueError("actual verification calls do not match cursor advancement")
    return True
