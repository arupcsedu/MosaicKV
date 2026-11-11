"""Synthetic bounded queues only; no real historical frontier is advanced."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import evicastkv_coco_history_subset as driver
import pytest

from mosaickv.evaluation import coco_history_subset as study


def fixture(index=1):
    state = dict(
        frontier=[
            dict(kind="directory", path="/synthetic/deferred-first"),
            dict(kind="directory", path="/synthetic/chosen"),
            dict(kind="file", path="/synthetic/deferred-last.json", offset=17),
        ],
        errors=[dict(path="old.yml", error="unresolved original metadata")],
        sample_ids=["old-id"],
        coco_ids=[],
        media_hashes=[],
        source_group_ids=[],
        scopes=[],
        scoped_identities=[],
        processed_files=5,
        projected_records=9,
        verification_paths=None,
        verification_index=0,
        traversal_complete=False,
    )
    parent = dict(
        source_sha=study.PARENT_SOURCE, index=index, state=state, state_sha256=study.digest(state)
    )
    spec = dict(
        state_sha256=parent["state_sha256"],
        selected_count=1,
        selected_sha256=study.digest([state["frontier"][1]]),
    )

    def predicate(path, _index):
        return path == "/synthetic/chosen"

    selected = study.split_frontier(parent, index, spec=spec, predicate=predicate)
    return parent, selected, spec, predicate


def scanner_fixture(action=None):
    observations = {}

    class Budget:
        def __init__(self, **kwargs):
            observations["budget"] = kwargs

    scanner = SimpleNamespace(
        within=lambda path, roots: any(path.is_relative_to(root) for root in roots),
        _parquet=object(),
        MAX_LINE=1024,
        Budget=Budget,
    )

    def advance(state, roots, **kwargs):
        observations.update(roots=roots, kwargs=kwargs, current_line_bound=scanner.MAX_LINE)
        observations["inside"] = scanner.within(Path("/synthetic/chosen/file.json"), roots)
        observations["escaped_alias"] = scanner.within(
            Path("/synthetic/elsewhere/file.json"), roots
        )
        state["frontier"] = []
        state["traversal_complete"] = True
        state["verification_paths"] = ["internal-global-verification-not-reusable"]
        state["verification_index"] = 1
        state["processed_files"] += 1
        state["projected_records"] += 2
        state["sample_ids"].append("new-id")
        if action is not None:
            action(state)

    scanner.advance = advance
    return scanner, observations


def run(parent, selected, action=None):
    scanner, observations = scanner_fixture(action)
    original_within, original_parquet, original_line = (
        scanner.within,
        scanner._parquet,
        scanner.MAX_LINE,
    )
    state, summary = study.advance_subset(
        parent, selected, (Path("/synthetic"),), scanner=scanner, empty_projection_parquet=object()
    )
    assert scanner.within is original_within and scanner._parquet is original_parquet
    assert original_line == scanner.MAX_LINE
    return state, summary, observations


def test_exact_partition_preserves_each_original_queue_position_and_parent():
    parent, selected, _spec, _predicate = fixture()
    assert [r["original_queue_offset"] for r in selected["selected"]] == [1]
    assert [r["original_queue_offset"] for r in selected["deferred"]] == [0, 2]
    assert len(parent["state"]["frontier"]) == 3


def test_continuation_preserves_deferred_frontier_errors_and_verification_cursor():
    parent, selected, _spec, _predicate = fixture()
    before = deepcopy(parent)
    state, report, _obs = run(parent, selected)
    assert parent == before
    assert state["frontier"] == [before["state"]["frontier"][0], before["state"]["frontier"][2]]
    assert state["errors"] == before["state"]["errors"]
    assert state["verification_paths"] is None and state["verification_index"] == 0
    assert state["traversal_complete"] is False
    assert report["subset_frontier_exhausted"] is True
    assert report["globally_fresh"] is False
    assert report["fresh_confirmation_authorized"] is False


def test_original_owner_index_and_global_shard_count_are_not_renumbered():
    parent, selected, _spec, _predicate = fixture(index=7)
    _state, _report, observed = run(parent, selected)
    assert observed["kwargs"]["shard_index"] == 7
    assert observed["kwargs"]["shard_count"] == 8
    assert observed["roots"] == (Path("/synthetic"),)


def test_budget_and_unchanged_line_policy_are_explicitly_bounded():
    parent, selected, _spec, _predicate = fixture()
    _state, _report, observed = run(parent, selected)
    assert observed["budget"] == dict(seconds=210, maximum_bytes=2 * 2**30)
    assert observed["current_line_bound"] == 8 * 2**20


def test_alias_inside_global_root_but_outside_selected_tree_is_rejected():
    parent, selected, _spec, _predicate = fixture()
    _state, _report, observed = run(parent, selected)
    assert observed["inside"] is True
    assert observed["escaped_alias"] is False


def test_remaining_selected_work_is_kept_after_unchanged_deferred_entries():
    parent, selected, _spec, _predicate = fixture()
    state, report, _obs = run(
        parent,
        selected,
        lambda state: state.update(
            frontier=[dict(kind="file", path="/synthetic/chosen/next.json", offset=4)]
        ),
    )
    assert state["frontier"][-1]["offset"] == 4
    assert len(state["frontier"]) == 3
    assert report["subset_frontier_exhausted"] is False


@pytest.mark.parametrize("mutation", ("source", "state", "index", "count", "digest", "kind"))
def test_parent_or_exact_reviewed_queue_mismatch_fails_closed(mutation):
    parent, _selected, spec, predicate = fixture()
    if mutation == "source":
        parent["source_sha"] = "0" * 40
    elif mutation == "state":
        parent["state"]["errors"] = []
    elif mutation == "index":
        parent["index"] = True
    elif mutation == "count":
        spec["selected_count"] = 2
    elif mutation == "digest":
        spec["selected_sha256"] = "0" * 64
    else:
        parent["state"]["frontier"][1]["kind"] = "file"
        parent["state_sha256"] = spec["state_sha256"] = study.digest(parent["state"])
    with pytest.raises(ValueError):
        study.split_frontier(parent, 1, spec=spec, predicate=predicate)


@pytest.mark.parametrize(
    "mutation", ("drop_deferred", "edit_deferred", "bool_offset", "edit_selected")
)
def test_deferred_queue_cannot_be_deleted_renumbered_or_rewritten(mutation):
    parent, selected, _spec, _predicate = fixture()
    if mutation == "drop_deferred":
        selected["deferred"].pop()
    elif mutation == "edit_deferred":
        selected["deferred"][0]["item"]["path"] = "/synthetic/other"
    elif mutation == "bool_offset":
        selected["deferred"][0]["original_queue_offset"] = False
    else:
        selected["selected"][0]["item"]["path"] = "/synthetic/other"
    with pytest.raises(ValueError):
        run(parent, selected)


@pytest.mark.parametrize("action", ("drop_error", "drop_old_id", "counter_decrease", "escape"))
def test_scanner_result_cannot_lose_old_exclusions_or_escape_scope(action):
    parent, selected, _spec, _predicate = fixture()

    def mutate(state):
        if action == "drop_error":
            state["errors"] = []
        elif action == "drop_old_id":
            state["sample_ids"] = ["new-id"]
        elif action == "counter_decrease":
            state["processed_files"] = 0
        else:
            state["frontier"] = [dict(kind="directory", path="/synthetic/escape")]

    with pytest.raises(ValueError):
        run(parent, selected, mutate)


def controls():
    whole = dict(
        source_sha="a" * 40,
        complete=True,
        source_fidelity_128_pass=True,
        whole_request_KV_byte_match_pass=True,
        whole_request_KV_peak_certified=True,
        failures=[],
    )
    coco = dict(
        source_sha="b" * 40,
        complete=True,
        corrected_boundary_128_pass=True,
        failures=[],
        source_scope="FP16_GQA_COCO_FullKV_evaluation_only_boundary_control",
    )
    return whole, coco


def test_engineering_admission_does_not_authorize_confirmation():
    result = driver.empirical_controls(
        *controls(), whole_producer_sha="a" * 40, coco_producer_sha="b" * 40
    )
    assert result["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize(
    "field",
    (
        "complete",
        "source_fidelity_128_pass",
        "whole_request_KV_byte_match_pass",
        "whole_request_KV_peak_certified",
    ),
)
def test_whole_request_gate_must_pass_every_registered_field(field):
    whole, coco = controls()
    whole[field] = False
    with pytest.raises(ValueError, match="whole-request"):
        driver.empirical_controls(
            whole, coco, whole_producer_sha="a" * 40, coco_producer_sha="b" * 40
        )


@pytest.mark.parametrize(
    "field", ("complete", "corrected_boundary_128_pass", "source_scope", "source_sha", "failures")
)
def test_coco_boundary_gate_is_distinct_and_fail_closed(field):
    whole, coco = controls()
    coco[field] = "not-passed"
    with pytest.raises(ValueError, match="COCO numerical"):
        driver.empirical_controls(
            whole, coco, whole_producer_sha="a" * 40, coco_producer_sha="b" * 40
        )


def test_registered_real_path_policy_does_not_select_sibling_workloads():
    root = "/sfs/weka/scratch/djy8hg/runs/evicastkv/paper-quality-coco-20261106-guard-recovery1/"
    assert study.reviewed_path(root + "quality/workers/0", 1)
    assert not study.reviewed_path(root + "quality/workers/2", 1)
    assert not study.reviewed_path(root + "timing/workers/6", 1)
    assert not study.reviewed_path(root + "quality/workers/0/child", 1)
    assert not study.reviewed_path(root + "quality/workers/0", 7)


def test_independent_reducer_rechecks_preserved_queue_not_only_state_hash():
    parent, selected, _spec, _predicate = fixture()
    state, summary, _observed = run(parent, selected)
    assert study.verify_continuation(parent, selected, state, summary)
    state["frontier"][0]["path"] = "/synthetic/forged-deferred"
    with pytest.raises(ValueError, match="deferred"):
        study.verify_continuation(parent, selected, state, summary)


def test_independent_reducer_rejects_boolean_or_forged_progress_summary():
    parent, selected, _spec, _predicate = fixture()
    state, summary, _observed = run(parent, selected)
    summary["newly_processed_files"] = True
    with pytest.raises(ValueError, match="independent"):
        study.verify_continuation(parent, selected, state, summary)
