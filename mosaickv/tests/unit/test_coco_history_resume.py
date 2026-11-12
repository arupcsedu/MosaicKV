"""Synthetic metadata only: never advance any real historical run."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import coco_history_resume as study


def fixture():
    pending = [
        dict(
            kind="file",
            path="/synthetic/a.jsonl",
            signature=[1, 2, 30, 4],
            offset=17,
            segments=[dict(offset=0, length=17, sha256="old-segment")],
        ),
        dict(kind="directory", path="/synthetic/b", signature=[1, 3, None, 4]),
        dict(
            kind="directory",
            path="/synthetic/c",
            signature=[1, 4, None, 4],
            position=2,
            prefix_sha256="original-prefix",
        ),
    ]
    deferred = [dict(kind="file", path="/deferred/old.jsonl", offset=23)]
    base = dict(
        frontier=deepcopy(deferred + pending),
        sample_ids=["old-id"],
        coco_ids=["1"],
        media_hashes=["old-hash"],
        source_group_ids=["old-group"],
        scopes=[["old", "split"]],
        scoped_identities=[["old", "split", "old-id"]],
        seen_files=["/old/kept.json"],
        seen_directories=["/old"],
        errors=[dict(path="/old/unresolved.yml", error="preserve me")],
        aliases=[],
        projection_pauses=[],
        scheduler_logs_excluded=[],
        bindings={"/old/kept.json": dict(sha256="original-binding")},
        directories={"/old": [1, 5, None, 4]},
        processed_files=4,
        projected_records=10,
        binary_or_unstructured_files=2,
        verification_paths=["/old/kept.json"],
        verification_index=0,
        traversal_complete=False,
        future_immutable_field={"must": "remain"},
    )
    return base, pending, deferred


def scanner_fixture(action=None):
    seen = {}

    class Budget:
        def __init__(self, **kwargs):
            seen["budget"] = kwargs

    scanner = SimpleNamespace(
        within=lambda path, roots: any(path.is_relative_to(root) for root in roots),
        _parquet=object(),
        MAX_LINE=100,
        Budget=Budget,
    )

    def advance(state, roots, **kwargs):
        seen.update(
            kwargs=kwargs,
            roots=roots,
            line=scanner.MAX_LINE,
            verification_paths=deepcopy(state["verification_paths"]),
            verification_index=state["verification_index"],
        )
        if action is not None:
            action(state, scanner)
        # A restricted exhaustion may be locally complete, never globally so.
        state["traversal_complete"] = not state["frontier"]

    scanner.advance = advance
    return scanner, seen


def run(base, queue, *, action=None, scope=None):
    scanner, seen = scanner_fixture(action)
    originals = (scanner.within, scanner._parquet, scanner.MAX_LINE)
    result = study.advance_partition(
        base,
        queue,
        scope if scope is not None else [item["path"] for item in base["frontier"][1:]],
        (Path("/synthetic"),),
        scanner,
        object(),
    )
    assert (scanner.within, scanner._parquet, scanner.MAX_LINE) == originals
    return result, seen


def consume(state, _scanner):
    for item in state["frontier"]:
        name = item["path"]
        if item["kind"] == "file":
            state["bindings"][name] = dict(signature=item.get("signature"), sha256="new")
            state["seen_files"].append(name)
            state["processed_files"] += 1
            state["projected_records"] += 2
        else:
            state["directories"][name] = item.get("signature")
            state["seen_directories"].append(name)
    state["frontier"] = []


def resign(result):
    result["state_sha256"] = study.digest(result["state"])


def test_partition_keeps_exact_entries_cursors_and_base_unchanged():
    base, pending, deferred = fixture()
    before = deepcopy(base)
    plan = study.partition(base, pending, deferred_count=1, parts=8)
    assert plan["queues"][:3] == [[item] for item in pending]
    assert plan["queues"][3:] == [[]] * 5
    assert plan["deferred"] == deferred
    assert base == before
    assert plan["original_shard_index"] == 7 and plan["original_shard_count"] == 8
    assert plan["base_state_sha256"] == study.digest(base)


def test_related_pending_directory_and_children_stay_in_one_partition():
    base, pending, _ = fixture()
    child = dict(kind="file", path="/synthetic/b/resume.jsonl", offset=91)
    pending = [child, *pending]
    base["frontier"] = [base["frontier"][0], *pending]
    plan = study.partition(base, pending, deferred_count=1, parts=3)
    indexes = {row["path"]: index for index, queue in enumerate(plan["queues"]) for row in queue}
    assert indexes[child["path"]] == indexes["/synthetic/b"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(parts=True),
        dict(parts=0),
        dict(parts=65),
        dict(deferred_count=True),
        dict(deferred_count=99),
    ],
)
def test_partition_rejects_invalid_integer_controls(kwargs):
    base, pending, _ = fixture()
    controls = dict(deferred_count=1, parts=8) | kwargs
    with pytest.raises(ValueError):
        study.partition(base, pending, **controls)


def test_partition_rejects_changed_tail_or_duplicate_paths():
    base, pending, _ = fixture()
    changed = deepcopy(pending)
    changed[0]["offset"] = 19
    with pytest.raises(ValueError, match="exactly"):
        study.partition(base, changed, deferred_count=1)
    base["frontier"].append(deepcopy(pending[0]))
    with pytest.raises(ValueError, match="duplicate"):
        study.partition(base, base["frontier"][1:], deferred_count=1)


def test_file_cannot_be_an_ancestor_of_another_pending_item():
    base, pending, _ = fixture()
    pending.append(dict(kind="file", path="/synthetic/a.jsonl/child"))
    base["frontier"] = [base["frontier"][0], *pending]
    with pytest.raises(ValueError, match="file cannot contain"):
        study.partition(base, pending, deferred_count=1)


@pytest.mark.parametrize("path", ["relative/path", "/synthetic/../escape", "/synthetic//a"])
def test_noncanonical_paths_rejected(path):
    base, pending, _ = fixture()
    pending[0]["path"] = path
    base["frontier"] = [base["frontier"][0], *pending]
    with pytest.raises(ValueError):
        study.partition(base, pending, deferred_count=1)


def test_unchanged_budget_owner_projection_and_global_cursor_restoration():
    base, pending, _ = fixture()
    result, seen = run(base, pending[:1])
    assert seen["kwargs"]["shard_index"] == 7 and seen["kwargs"]["shard_count"] == 8
    assert seen["budget"] == dict(seconds=210, maximum_bytes=2 * 2**30)
    assert seen["line"] == 8 * 2**20
    assert seen["verification_paths"] == [] and seen["verification_index"] == 0
    assert result["state"]["verification_paths"] == base["verification_paths"]
    assert result["state"]["verification_index"] == 0
    assert result["state"]["traversal_complete"] is False
    assert result["state"]["frontier"] == pending[:1]
    assert study.verify_result(base, pending[:1], result)


def test_consumed_files_and_directories_have_receipts_and_no_global_pass():
    base, pending, _ = fixture()
    result, _ = run(base, pending, action=consume)
    assert result["state"]["frontier"] == []
    assert result["summary"]["partition_frontier_exhausted"] is True
    for name in (
        "globally_fresh",
        "global_historical_inventory_complete",
        "fresh_confirmation_authorized",
        "materialization_authorized",
    ):
        assert result["summary"][name] is False
    assert result["summary"]["newly_processed_files"] == 1


def test_paused_file_cursor_and_segments_are_preserved_and_may_advance():
    base, pending, _ = fixture()

    def pause(state, _scanner):
        item = state["frontier"][0]
        item["offset"] = 22
        item["segments"].append(dict(offset=17, length=5, sha256="new-segment"))
        state["projection_pauses"].append(dict(path=item["path"], offset=22))

    result, _ = run(base, pending[:1], action=pause)
    assert result["state"]["frontier"][0]["offset"] == 22
    assert result["state"]["frontier"][0]["segments"][:1] == pending[0]["segments"]


def test_cross_partition_target_aborts_not_historical_error_and_restores_patch():
    base, pending, _ = fixture()
    scanner, _ = scanner_fixture()
    old = scanner.within

    def action(state, _roots, **_kwargs):
        try:
            scanner.within(Path("/synthetic/b/other.json"), (Path("/synthetic"),))
        except Exception:
            state["errors"].append(dict(error="MUST NOT HAPPEN"))

    scanner.advance = action
    before = deepcopy(base)
    with pytest.raises(study.ExecutionPartitionFenceError):
        study.advance_partition(
            base,
            pending[:1],
            [row["path"] for row in pending],
            (Path("/synthetic"),),
            scanner,
            object(),
        )
    assert scanner.within is old and base == before


def test_outside_registered_scope_preserves_false_failclosed_semantics():
    base, pending, _ = fixture()

    def outside(state, scanner):
        assert scanner.within(Path("/synthetic/not-reviewed.json"), (Path("/synthetic"),)) is False
        state["errors"].append(dict(path=pending[0]["path"], error="outside reviewed root"))
        state["frontier"] = []

    result, _ = run(base, pending[:1], action=outside)
    assert result["summary"]["new_errors"] == 1
    assert result["state"]["errors"][:1] == base["errors"]


def test_original_allowed_roots_check_runs_before_partition_fence():
    base, pending, _ = fixture()
    scanner, _ = scanner_fixture()
    scanner.within = lambda *_: False

    def action(_state, _roots, **_kwargs):
        assert scanner.within(Path("/synthetic/b"), (Path("/synthetic"),)) is False

    scanner.advance = action
    study.advance_partition(
        base,
        pending[:1],
        [row["path"] for row in pending],
        (Path("/synthetic"),),
        scanner,
        object(),
    )


@pytest.mark.parametrize(
    "field,path",
    [
        ("seen_files", "/synthetic/b/already.json"),
        ("seen_directories", "/synthetic/b/already"),
    ],
)
def test_previously_seen_canonical_foreign_alias_is_allowed_without_new_coverage(field, path):
    base, pending, _ = fixture()
    base[field].append(path)

    def alias(state, scanner):
        assert scanner.within(Path(path), (Path("/synthetic"),)) is True
        state["aliases"].append(dict(alias="/synthetic/a.jsonl", canonical_target=path))

    result, _ = run(base, pending[:1], action=alias)
    assert result["state"][field] == base[field]
    assert result["state"]["errors"] == base["errors"]


def test_empty_execution_partition_is_an_explicit_unchanged_state():
    base, _pending, _ = fixture()
    result, _ = run(base, [])
    assert result["state"]["frontier"] == []
    assert result["summary"]["newly_processed_files"] == 0
    assert result["summary"]["new_errors"] == 0


def test_new_identity_and_nested_scope_exclusions_survive_merge():
    base, pending, deferred = fixture()
    queues = study.partition(base, pending, deferred_count=1, parts=2)["queues"]
    results = []
    for index, queue in enumerate(queues):

        def add(state, _scanner, index=index):
            state["scopes"].append(["new", str(index)])
            state["scoped_identities"].append(["new", str(index), "id"])
            state["coco_ids"].append(str(index + 2))

        results.append(run(base, queue, action=add)[0])
    output = study.merge(base, queues, results, deferred)
    assert output["coco_ids"] == ["1", "2", "3"]
    assert output["scopes"] == [["new", "0"], ["new", "1"], ["old", "split"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("errors", []),
        ("sample_ids", []),
        ("seen_files", []),
        ("bindings", {}),
        ("directories", {}),
        ("processed_files", 3),
        ("processed_files", True),
        ("verification_index", 1),
        ("traversal_complete", True),
        ("future_immutable_field", {"changed": True}),
    ],
)
def test_resigned_state_tampering_is_rejected(field, value):
    base, pending, _ = fixture()
    result, _ = run(base, pending[:1])
    result["state"][field] = value
    resign(result)
    with pytest.raises(ValueError):
        study.verify_result(base, pending[:1], result)


@pytest.mark.parametrize("change", ["backwards", "signature", "segment", "disappear", "escape"])
def test_frontier_receipts_and_cursors_cannot_be_forged(change):
    base, pending, _ = fixture()
    result, _ = run(base, pending[:1])
    row = result["state"]["frontier"][0]
    if change == "backwards":
        row["offset"] = 1
    elif change == "signature":
        row["signature"] = [9, 9, 9, 9]
    elif change == "segment":
        row["segments"] = []
    elif change == "disappear":
        result["state"]["frontier"] = []
    else:
        row["path"] = "/synthetic/c/escape.json"
    resign(result)
    with pytest.raises(ValueError):
        study.verify_result(base, pending[:1], result)


def test_unknown_new_field_and_changed_digest_are_rejected():
    base, pending, _ = fixture()
    result, _ = run(base, pending[:1])
    result["queue_sha256"] = "forged"
    with pytest.raises(ValueError):
        study.verify_result(base, pending[:1], result)
    result, _ = run(base, pending[:1])
    result["state"]["new_unregistered_field"] = 1
    resign(result)
    with pytest.raises(ValueError):
        study.verify_result(base, pending[:1], result)


def test_summary_boolean_integer_substitution_rejected():
    base, pending, _ = fixture()
    result, _ = run(base, pending[:1])
    result["summary"]["globally_fresh"] = 0
    with pytest.raises(ValueError, match="summary"):
        study.verify_result(base, pending[:1], result)


def test_merge_preserves_deferred_old_errors_unknown_fields_and_unions_deltas_once():
    base, pending, deferred = fixture()
    plan = study.partition(base, pending, deferred_count=1, parts=2)
    results = []
    for index, queue in enumerate(plan["queues"]):

        def action(state, scanner, index=index):
            consume(state, scanner)
            state["sample_ids"].append(f"new-{index}")
            state["errors"].append(dict(path=f"/synthetic/new-{index}", error="new unresolved"))

        result, _ = run(base, queue, action=action)
        results.append(result)
    output = study.merge(base, plan["queues"], results, deferred)
    assert output["frontier"] == deferred
    assert output["errors"] == base["errors"] + [r["state"]["errors"][-1] for r in results]
    assert output["processed_files"] == base["processed_files"] + 1
    assert output["projected_records"] == base["projected_records"] + 2
    assert set(output["sample_ids"]) == {"old-id", "new-0", "new-1"}
    assert output["future_immutable_field"] == base["future_immutable_field"]
    assert output["verification_paths"] == base["verification_paths"]
    assert output["traversal_complete"] is False
    assert base["frontier"] == deferred + pending


def test_next_round_contains_only_saved_residuals_not_completed_files():
    base, pending, deferred = fixture()
    plan = study.partition(base, pending, deferred_count=1, parts=2)
    result0, _ = run(base, plan["queues"][0], action=consume)
    result1, _ = run(base, plan["queues"][1])
    merged = study.merge(base, plan["queues"], [result0, result1], deferred)
    next_plan = study.partition(merged, merged["frontier"][1:], deferred_count=1, parts=2)
    assert [item for queue in next_plan["queues"] for item in queue] == plan["queues"][1]
    assert pending[0] not in merged["frontier"]


@pytest.mark.parametrize(
    "change", ["missing-result", "duplicate-queue", "changed-deferred", "changed-entry"]
)
def test_merge_rejects_incomplete_duplicated_or_rewritten_round(change):
    base, pending, deferred = fixture()
    plan = study.partition(base, pending, deferred_count=1, parts=2)
    queues = plan["queues"]
    results = [run(base, queue)[0] for queue in queues]
    if change == "missing-result":
        results.pop()
    elif change == "duplicate-queue":
        queues[1] = deepcopy(queues[0])
    elif change == "changed-deferred":
        deferred[0]["offset"] = 99
    else:
        queues[0][0]["offset"] = 19
    with pytest.raises(ValueError):
        study.merge(base, queues, results, deferred)


def test_ancestor_related_entries_cannot_be_split_between_execution_workers():
    base, pending, deferred = fixture()
    child = dict(kind="file", path="/synthetic/b/child.jsonl", offset=0)
    pending.append(child)
    base["frontier"] = deferred + pending
    queues = [[pending[1]], [pending[0], pending[2], child]]
    results = [run(base, queue)[0] for queue in queues]
    with pytest.raises(ValueError, match="ancestor-related"):
        study.merge(base, queues, results, deferred)


def test_real_projector_synthetic_file_resumes_with_original_owner_and_no_labels(tmp_path):
    import evicastkv_review_freshness_stream as scanner

    # Find a synthetic canonical filename assigned to the unchanged owner.
    roots = (tmp_path.resolve(),)
    path = next(
        tmp_path / f"case-{index}.jsonl"
        for index in range(200)
        if scanner.owner(tmp_path / f"case-{index}.jsonl", roots, 8) == 7
    )
    path.write_text('{"sample_id":"synthetic-id","answer":"DO NOT RETAIN"}\n')
    queue = [dict(kind="file", path=str(path), signature=scanner.signature(path))]
    base = scanner.initial_state(roots)
    base.update(frontier=deepcopy(queue), traversal_complete=False, projection_pauses=[])
    result = study.advance_partition(base, queue, [str(path)], roots, scanner, object())
    assert result["state"]["frontier"] == []
    assert "synthetic-id" in result["state"]["sample_ids"]
    assert "DO NOT RETAIN" not in str(result)
    assert result["state"]["processed_files"] == 1
