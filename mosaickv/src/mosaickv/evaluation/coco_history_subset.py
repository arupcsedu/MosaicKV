"""Finite historical COCO-subset continuation, never a freshness certificate.

The original global owner indices and every deferred frontier/error are kept.
Only the two reviewed pending-directory subsets may be parsed.  Existing
scanner identity projection and byte/clock budgets remain authoritative.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import digest

PARENT_SOURCE = "84900a6a402e97c7d58499a722c159cfc86e46c7"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history/segment-4")
SPECS = {
    1: dict(
        worker_sha256="3b72165f4e5a7c4c4c4fc893bc201a9083d9bd118d58be13804c2749dde36768",
        state_sha256="ae1202f35b979c1bf38b9b920e5439d8fe444f4af661f37e1d499edfc095314c",
        selected_count=9,
        selected_sha256="c22d1f3c408a661a842ca5ff0388f2917b28bee7ac2975f2d782a29d687c35af",
    ),
    7: dict(
        worker_sha256="a3d6c115a9cbc9858ed247858adf5d7cac23a6bbfdb8c9236d91da2134cc34a9",
        state_sha256="419495d67305ce69dc5ee6d25c5d224bb1ac600efbd77b6f0c91ba20172b8d9a",
        selected_count=512,
        selected_sha256="7a3b1e18af3facce1749e81e64f67901ed0b7b9ba8b2286efa5297f689ab1e7f",
    ),
}
SECONDS = 210
MAX_BYTES = 2 * 2**30
LINE_BYTES = 8 * 2**20
ORIGINAL_SHARD_COUNT = 8
GLOBAL_PARENT_FRONTIER = 9002


def reviewed_path(path, index):
    """The exact registered prefix is additionally protected by its queue hash."""
    if type(index) is not int or index not in SPECS or not isinstance(path, str):
        return False
    if index == 1:
        root = (
            "/sfs/weka/scratch/djy8hg/runs/evicastkv/paper-quality-coco-20261106-guard-recovery1/"
        )
        return path in {
            root + suffix
            for suffix in (
                "canary/workers/0",
                "quality/workers/0",
                "quality/workers/1",
                *(f"timing/workers/{i}" for i in range(6)),
            )
        }
    root = (
        "/sfs/weka/scratch/djy8hg/runs/mosaickv/paper-native-7day-v1-61e5a68/vl_cache/coco_caption/"
    )
    return path.startswith(root)


def split_frontier(parent, index, *, spec=None, predicate=reviewed_path):
    """Validate the full immutable parent; keep exact deferred queue entries."""
    if type(index) is not int or index not in SPECS:
        raise ValueError("original shard index must remain 1 or 7, never renumbered")
    spec = SPECS[index] if spec is None else spec
    if (
        parent.get("source_sha") != PARENT_SOURCE
        or type(parent.get("index")) is not int
        or parent["index"] != index
        or not isinstance(parent.get("state"), dict)
        or parent.get("state_sha256") != digest(parent["state"])
        or parent["state_sha256"] != spec["state_sha256"]
    ):
        raise ValueError("original complete state/source/index digest differs")
    frontier = parent["state"].get("frontier")
    if not isinstance(frontier, list) or not frontier:
        raise ValueError("an explicit unfinished original frontier is required")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in frontier
    ):
        raise ValueError("malformed original frontier entry")
    selected, deferred = [], []
    for offset, item in enumerate(frontier):
        row = dict(original_queue_offset=offset, item=copy.deepcopy(item))
        if predicate(item["path"], index):
            if item.get("kind") != "directory":
                raise ValueError("the reviewed COCO subset contains directories only")
            selected.append(row)
        else:
            deferred.append(row)
    queue = [row["item"] for row in selected]
    if len(queue) != spec["selected_count"] or digest(queue) != spec["selected_sha256"]:
        raise ValueError("exact reviewed COCO pending queue count/digest differs")
    if len({row["path"] for row in queue}) != len(queue):
        raise ValueError("duplicate reviewed COCO pending directory")
    return dict(
        index=index,
        original_frontier_count=len(frontier),
        original_frontier_sha256=digest(frontier),
        selected=selected,
        deferred=deferred,
        selected_frontier_sha256=digest(queue),
        deferred_frontier_sha256=digest([row["item"] for row in deferred]),
        parent_state_sha256=parent["state_sha256"],
    )


def advance_subset(parent, subset, roots, *, scanner, empty_projection_parquet):
    """One finite invocation; no caller can change the registered time/byte cap.

    The scanner's ``within`` predicate is only tightened to reject a symlink
    escaping the selected directory trees.  Original ``roots``, ``owner`` and
    shard count are retained, avoiding changes to deterministic ownership.
    Deferred entries are restored in their original order after the invocation.
    """
    index = subset["index"]
    if type(index) is not int or index not in SPECS:
        raise ValueError("original shard index must remain 1 or 7")
    original = parent["state"]
    if digest(original) != subset["parent_state_sha256"]:
        raise ValueError("bound parent state changed before continuation")
    if digest(original["frontier"]) != subset["original_frontier_sha256"]:
        raise ValueError("original full frontier changed before continuation")
    if digest([row["item"] for row in subset["selected"]]) != subset["selected_frontier_sha256"]:
        raise ValueError("selected queue changed before continuation")
    if digest([row["item"] for row in subset["deferred"]]) != subset["deferred_frontier_sha256"]:
        raise ValueError("deferred queue changed before continuation")
    if any(
        type(row.get("original_queue_offset")) is not int
        for row in [*subset["selected"], *subset["deferred"]]
    ):
        raise ValueError("original queue positions must be strict integers")
    combined = sorted(
        [*subset["selected"], *subset["deferred"]], key=lambda row: row["original_queue_offset"]
    )
    if [row["original_queue_offset"] for row in combined] != list(
        range(len(original["frontier"]))
    ) or ([row["item"] for row in combined] != original["frontier"]):
        raise ValueError("selected/deferred partition does not exactly preserve the original queue")
    state = copy.deepcopy(original)
    state["frontier"] = [copy.deepcopy(row["item"]) for row in subset["selected"]]
    initial_directories = tuple(Path(row["item"]["path"]) for row in subset["selected"])
    original_within = scanner.within

    def scoped_within(path, actual_roots):
        return original_within(path, actual_roots) and any(
            path.is_relative_to(directory) for directory in initial_directories
        )

    with (
        patch.object(scanner, "_parquet", empty_projection_parquet),
        patch.object(scanner, "MAX_LINE", LINE_BYTES),
        patch.object(scanner, "within", scoped_within),
    ):
        scanner.advance(
            state,
            tuple(roots),
            shard_index=index,
            shard_count=ORIGINAL_SHARD_COUNT,
            budget=scanner.Budget(seconds=SECONDS, maximum_bytes=MAX_BYTES),
        )
    pending = copy.deepcopy(state["frontier"])
    if any(
        not any(Path(row["path"]).is_relative_to(directory) for directory in initial_directories)
        for row in pending
    ):
        raise ValueError("continued pending work escaped the explicitly reviewed COCO trees")
    if state["errors"][: len(original["errors"])] != original["errors"]:
        raise ValueError("original historical coverage errors were removed or rewritten")
    for field in (
        "sample_ids",
        "coco_ids",
        "media_hashes",
        "source_group_ids",
        "scopes",
        "scoped_identities",
    ):
        if field in original:
            nested = field in {"scopes", "scoped_identities"}
            old_values = {tuple(value) if nested else value for value in original[field]}
            new_values = {tuple(value) if nested else value for value in state[field]}
            if not old_values <= new_values:
                raise ValueError("original historical identity exclusions were removed")
    if any(
        type(state.get(field)) is not int or state[field] < original[field]
        for field in ("processed_files", "projected_records")
    ):
        raise ValueError("historical progress counters cannot decrease or change type")
    deferred = [copy.deepcopy(row["item"]) for row in subset["deferred"]]
    state["frontier"] = [*deferred, *pending]
    # The restricted queue cannot complete global verification or exclude
    # future deferred files. Keep its original global verification cursor.
    state["verification_paths"] = copy.deepcopy(original.get("verification_paths"))
    state["verification_index"] = original.get("verification_index", 0)
    state["traversal_complete"] = False
    summary = dict(
        original_shard_index=index,
        original_shard_count=ORIGINAL_SHARD_COUNT,
        selected_initial_directories=len(initial_directories),
        subset_pending_items=len(pending),
        subset_frontier_exhausted=not pending,
        original_deferred_frontier_items_preserved=len(deferred),
        original_errors_preserved=True,
        new_errors=state["errors"][len(original["errors"]) :],
        newly_processed_files=state["processed_files"] - original["processed_files"],
        newly_projected_records=state["projected_records"] - original["projected_records"],
        global_historical_inventory_complete=False,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        scoped_alias_escape_rejected=True,
        original_projection_and_owner_functions_unchanged=True,
        maximum_seconds=SECONDS,
        maximum_source_bytes=MAX_BYTES,
    )
    return state, summary


def verify_continuation(parent, selected, state, summary):
    """Recheck preserved state independently; a self-consistent hash is insufficient."""
    original = parent["state"]
    if digest(original) != selected["parent_state_sha256"]:
        raise ValueError("immutable original state required for independent verification")
    deferred = [row["item"] for row in selected["deferred"]]
    if state["frontier"][: len(deferred)] != deferred:
        raise ValueError("continued state lost or rewrote an original deferred entry")
    pending = state["frontier"][len(deferred) :]
    roots = tuple(Path(row["item"]["path"]) for row in selected["selected"])
    if any(not any(Path(row["path"]).is_relative_to(root) for root in roots) for row in pending):
        raise ValueError("continued work escaped its exact COCO scope")
    if (
        state["traversal_complete"] is not False
        or state.get("verification_paths") != original.get("verification_paths")
        or state.get("verification_index", 0) != original.get("verification_index", 0)
    ):
        raise ValueError("subset cannot complete or change original global verification")
    if state["errors"][: len(original["errors"])] != original["errors"]:
        raise ValueError("continued state lost an original coverage error")
    for field in (
        "sample_ids",
        "coco_ids",
        "media_hashes",
        "source_group_ids",
        "scopes",
        "scoped_identities",
    ):
        if field in original:
            nested = field in {"scopes", "scoped_identities"}
            old = {tuple(x) if nested else x for x in original[field]}
            new = {tuple(x) if nested else x for x in state[field]}
            if not old <= new:
                raise ValueError("continued state lost an original exclusion identity")
    expected = dict(
        original_shard_index=selected["index"],
        original_shard_count=8,
        selected_initial_directories=len(selected["selected"]),
        subset_pending_items=len(pending),
        subset_frontier_exhausted=not pending,
        original_deferred_frontier_items_preserved=len(deferred),
        original_errors_preserved=True,
        new_errors=state["errors"][len(original["errors"]) :],
        newly_processed_files=state["processed_files"] - original["processed_files"],
        newly_projected_records=state["projected_records"] - original["projected_records"],
        global_historical_inventory_complete=False,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        scoped_alias_escape_rejected=True,
        original_projection_and_owner_functions_unchanged=True,
        maximum_seconds=SECONDS,
        maximum_source_bytes=MAX_BYTES,
    )
    if summary != expected or any(
        type(summary[key]) is not type(value) for key, value in expected.items()
    ):
        raise ValueError("summary differs from independent original/result reduction")
    if expected["newly_processed_files"] < 0 or expected["newly_projected_records"] < 0:
        raise ValueError("historical progress decreased")
    return True
