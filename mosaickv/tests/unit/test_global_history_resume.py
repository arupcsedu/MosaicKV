"""Synthetic global metadata continuation; no historical scan or scheduler."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import coco_history_resume as old
from mosaickv.evaluation import global_history_resume as study
from mosaickv.evaluation.fox_campaign import digest


def fixture():
    state = dict(
        frontier=[
            dict(
                kind="file",
                path="/synthetic/a.jsonl",
                offset=7,
                signature=[1, 2, 30, 4],
                segments=[],
            ),
            dict(kind="directory", path="/synthetic/b", signature=[1, 3, None, 4]),
        ],
        sample_ids=["old"],
        coco_ids=[],
        media_hashes=[],
        source_group_ids=[],
        scopes=[],
        scoped_identities=[],
        seen_files=["/synthetic/old.json"],
        seen_directories=["/synthetic/old"],
        errors=[dict(path="/old/error", error="keep")],
        aliases=[],
        scheduler_logs_excluded=[],
        bindings={"/synthetic/old.json": dict(signature=[])},
        directories={"/synthetic/old": []},
        processed_files=1,
        projected_records=2,
        binary_or_unstructured_files=0,
        verification_paths=None,
        verification_index=0,
        traversal_complete=False,
        unchanged_future_field={"old": True},
    )
    return study.normalize_seed(state)[0]


def scanner_fixture(owner, *, action=None, verification_steps=None, bad_stat=False):
    observed = {}

    class Budget:
        def __init__(self, **kwargs):
            observed["budget"] = kwargs

    scanner = SimpleNamespace(
        within=lambda path, roots: any(path.is_relative_to(root) for root in roots),
        owner=lambda path, roots, count: (owner + 1) % 8 if "other-owner" in str(path) else owner,
        _parquet=object(),
        MAX_LINE=123,
        Budget=Budget,
    )

    def guard(path, expected):
        observed.setdefault("stat_calls", []).append(str(path))
        if bad_stat:
            raise ValueError("changed synthetic historical signature")

    scanner._stat_guard = guard

    def advance(state, roots, **kwargs):
        observed.update(roots=roots, owner=kwargs["shard_index"], count=kwargs["shard_count"])
        if verification_steps is not None:
            assert state["frontier"] == []
            if state["verification_paths"] is None:
                state["verification_paths"] = sorted(state["directories"]) + sorted(
                    state["bindings"]
                )
            paths = state["verification_paths"]
            begin = state["verification_index"]
            for name in paths[begin : begin + verification_steps]:
                try:
                    scanner._stat_guard(Path(name), [])
                except ValueError as exc:
                    state["errors"].append(dict(path=name, error=str(exc)))
                state["verification_index"] += 1
            state["traversal_complete"] = state["verification_index"] == len(paths)
        elif action is not None:
            action(state, scanner)

    scanner.advance = advance
    return scanner, observed


def consume(state, _scanner):
    for row in state["frontier"]:
        if row["kind"] == "file":
            state["bindings"][row["path"]] = dict(signature=row.get("signature"))
            state["seen_files"].append(row["path"])
            state["processed_files"] += 1
            state["projected_records"] += 1
        else:
            state["directories"][row["path"]] = row.get("signature")
            state["seen_directories"].append(row["path"])
    state["frontier"] = []
    # The real unchanged scanner canonicalizes these sets before publication.
    state["seen_files"].sort()
    state["seen_directories"].sort()


def run(base, queue=None, *, owner=3, action=consume):
    queue = base["frontier"] if queue is None else queue
    scanner, observed = scanner_fixture(owner, action=action)
    original = (scanner.within, scanner._parquet, scanner.MAX_LINE)
    result = study.advance_partition(
        base, queue, ["/synthetic"], (Path("/synthetic"),), scanner, object(), owner_index=owner
    )
    assert (scanner.within, scanner._parquet, scanner.MAX_LINE) == original
    return result, observed


def test_normalization_is_additive_and_preserves_existing_verified_cursor():
    base = fixture()
    del base["projection_pauses"], base[study.EXTRA_MAP]
    base.update(
        frontier=[],
        verification_paths=["/old/verified"],
        verification_index=1,
        traversal_complete=True,
    )
    before = deepcopy(base)
    result, receipt = study.normalize_seed(base)
    assert base == before
    assert result["verification_paths"] == ["/old/verified"]
    assert result["verification_index"] == 1 and result["traversal_complete"] is True
    assert receipt["added_fields"] == {"projection_pauses": [], study.EXTRA_MAP: {}}
    assert receipt["original_state_sha256"] == digest(base)


@pytest.mark.parametrize("owner", range(8))
def test_every_original_owner_uses_unchanged_eight_way_scanner_and_budget(owner):
    base = fixture()
    result, seen = run(base, owner=owner)
    assert seen["owner"] == owner and seen["count"] == 8
    assert seen["budget"] == dict(seconds=210, maximum_bytes=2 * 2**30)
    assert result["owner_index"] == owner and result["owner_count"] == 8
    assert result["summary"]["original_shard_index"] == owner
    assert result["state"]["traversal_complete"] is False
    assert study.verify_result(base, base["frontier"], result, owner_index=owner)


@pytest.mark.parametrize("owner", [-1, 8, True, "7"])
def test_invalid_owner_never_changes_prior_module_globals(owner):
    base = fixture()
    before = old.OWNER_INDEX, old.DICT_FIELDS, old.KNOWN_FIELDS, old._base
    with pytest.raises(ValueError):
        study.partition(base, base["frontier"], owner_index=owner)
    assert (old.OWNER_INDEX, old.DICT_FIELDS, old.KNOWN_FIELDS, old._base) == before


def test_composition_is_nonreentrant_and_restores_all_globals():
    base = fixture()
    before = old.OWNER_INDEX, old.DICT_FIELDS, old.KNOWN_FIELDS, old._base
    with study._composition(0), pytest.raises(RuntimeError, match="process-isolated"):
        study.partition(base, base["frontier"], owner_index=1)
    assert (old.OWNER_INDEX, old.DICT_FIELDS, old.KNOWN_FIELDS, old._base) == before


def test_global_scope_cannot_be_tightened_to_pending_subtrees():
    base = fixture()
    scanner, _ = scanner_fixture(3)
    with pytest.raises(ValueError, match="unchanged historical scanner roots"):
        study.advance_partition(
            base,
            base["frontier"],
            ["/synthetic/a.jsonl"],
            (Path("/synthetic"),),
            scanner,
            object(),
            owner_index=3,
        )


def test_other_source_owner_is_skipped_not_an_execution_error():
    base = fixture()

    def skip(state, scanner):
        assert (
            scanner.within(Path("/synthetic/other-owner/new.json"), (Path("/synthetic"),)) is True
        )
        assert scanner.owner(Path("/synthetic/other-owner/new.json"), (), 8) != 3
        assert state["errors"] == base["errors"]

    result, _ = run(base, base["frontier"][:1], action=skip)
    assert result["state"]["errors"] == base["errors"]


def test_same_owner_unseen_foreign_partition_aborts_without_historical_error():
    base = fixture()

    def foreign(_state, scanner):
        scanner.within(Path("/synthetic/b/new.json"), (Path("/synthetic"),))

    before = deepcopy(base)
    with pytest.raises(old.ExecutionPartitionFenceError):
        run(base, base["frontier"][:1], action=foreign)
    assert base == before


def test_empty_parquet_metadata_additions_are_audited_not_unknown_mutations():
    base = fixture()
    base[study.EXTRA_MAP]["/old/metadata"] = dict(retain=True)

    def action(state, scanner):
        state[study.EXTRA_MAP][state["frontier"][0]["path"]] = dict(identity_effect="none")
        consume(state, scanner)

    result, _ = run(base, action=action)
    compact = study.compact_result(base, base["frontier"], result, owner_index=3)
    expanded = study.expand_result(base, base["frontier"], compact, owner_index=3)
    assert expanded == result
    assert expanded["state"][study.EXTRA_MAP]["/old/metadata"] == dict(retain=True)


def test_delta_contains_only_additions_and_reconstructs_full_state_hash_exactly():
    base = fixture()
    result, _ = run(base)
    delta = study.compact_result(base, base["frontier"], result, owner_index=3)
    assert "state" not in delta
    assert "/synthetic/old.json" not in delta["map_additions"]["bindings"]
    assert delta["counter_deltas"]["processed_files"] == 1
    expanded = study.expand_result(base, base["frontier"], delta, owner_index=3)
    assert expanded == result


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "base",
        "queue",
        "hash",
        "old-map",
        "old-set",
        "negative-counter",
        "boolean-counter",
        "missing-field",
        "extra-field",
    ],
)
def test_resigned_delta_tampering_fails_closed(change):
    base = fixture()
    result, _ = run(base)
    delta = study.compact_result(base, base["frontier"], result, owner_index=3)
    if change == "owner":
        delta["owner_index"] = 2
    elif change == "base":
        delta["base_state_sha256"] = "forged"
    elif change == "queue":
        delta["queue_sha256"] = "forged"
    elif change == "hash":
        delta["state_sha256"] = "forged"
    elif change == "old-map":
        delta["map_additions"]["bindings"]["/synthetic/old.json"] = dict(changed=True)
    elif change == "old-set":
        delta["set_additions"]["sample_ids"] = ["old"]
    elif change in {"negative-counter", "boolean-counter"}:
        delta["counter_deltas"]["processed_files"] = -1 if change == "negative-counter" else True
    elif change == "missing-field":
        del delta["append_tails"]["errors"]
    else:
        delta["unexpected"] = True
    delta["delta_sha256"] = digest(
        {key: value for key, value in delta.items() if key != "delta_sha256"}
    )
    with pytest.raises(ValueError):
        study.expand_result(base, base["frontier"], delta, owner_index=3)


def test_partition_merge_restores_deferred_and_does_not_repeat_completed_queue():
    base = fixture()
    deferred = dict(kind="file", path="/deferred/keep.json")
    base["frontier"].insert(0, deferred)
    plan = study.partition(base, base["frontier"][1:], 1, 2, owner_index=5)
    results = [run(base, queue, owner=5)[0] for queue in plan["queues"]]
    merged = study.merge(base, plan["queues"], results, plan["deferred"], owner_index=5)
    assert merged["frontier"] == [deferred]
    assert merged["processed_files"] == base["processed_files"] + 1
    assert merged["errors"] == base["errors"]


def verified_base():
    base = fixture()
    paths = sorted(base["directories"]) + sorted(base["bindings"])
    base.update(
        frontier=[],
        verification_paths=paths,
        verification_index=len(paths),
        traversal_complete=True,
    )
    return base


def test_explicit_new_epoch_preserves_old_state_and_hash_bound_previous_cursor():
    base = verified_base()
    before = deepcopy(base)
    state, receipt = study.begin_verification_epoch(base)
    assert base == before
    assert state["verification_paths"] is None
    assert state["verification_index"] == 0 and state["traversal_complete"] is False
    assert receipt["original_verification_index"] == 2
    assert receipt["original_verification_paths_sha256"] == digest(base["verification_paths"])
    assert receipt["previous_state_sha256"] == digest(base)
    assert study.verify_epoch(base, state, receipt)


def test_epoch_cannot_open_before_traversal_finishes_or_rewrite_evidence():
    with pytest.raises(ValueError, match="exhausted traversal"):
        study.begin_verification_epoch(fixture())
    base = verified_base()
    state, receipt = study.begin_verification_epoch(base)
    state["errors"] = []
    with pytest.raises(ValueError):
        study.verify_epoch(base, state, receipt)


def test_completed_old_cursor_cannot_silently_count_as_new_verification():
    base = verified_base()
    scanner, _ = scanner_fixture(0, verification_steps=10)
    with pytest.raises(ValueError, match="explicitly opened"):
        study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=0)


def test_current_verification_calls_real_guard_and_resumes_only_remaining_paths():
    base, _receipt = study.begin_verification_epoch(verified_base())
    scanner, seen = scanner_fixture(0, verification_steps=1)
    first = study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=0)
    assert first["state"]["verification_index"] == 1
    assert first["summary"]["actual_stat_guard_calls"] == 1
    assert first["summary"]["verification_exhausted"] is False
    second = study.advance_verification(
        first["state"], (Path("/synthetic"),), scanner, owner_index=0
    )
    assert second["state"]["verification_index"] == 2
    assert second["summary"]["verification_exhausted"] is True
    assert seen["stat_calls"] == sorted(base["directories"]) + sorted(base["bindings"])
    assert second["state"]["errors"] == base["errors"]
    assert second["summary"]["globally_fresh"] is False


def test_changed_historical_path_adds_error_without_erasing_old_error_or_stalling_cursor():
    base, _receipt = study.begin_verification_epoch(verified_base())
    scanner, _seen = scanner_fixture(7, verification_steps=20, bad_stat=True)
    result = study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=7)
    assert result["state"]["errors"][:1] == base["errors"]
    assert result["summary"]["new_errors"] == 2
    assert result["summary"]["actual_stat_guard_calls"] == 2
    assert result["state"]["traversal_complete"] is True
    assert result["summary"]["globally_fresh"] is False


def test_fake_cursor_advancement_without_guard_calls_is_rejected():
    base, _receipt = study.begin_verification_epoch(verified_base())
    scanner, _seen = scanner_fixture(1)

    def fake(state, _roots, **_kwargs):
        state.update(
            verification_paths=sorted(state["directories"]) + sorted(state["bindings"]),
            verification_index=2,
            traversal_complete=True,
        )

    scanner.advance = fake
    with pytest.raises(ValueError, match="actual verification calls"):
        study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=1)


def test_verification_reordered_guard_calls_abort_before_publication():
    base, _receipt = study.begin_verification_epoch(verified_base())
    scanner, _seen = scanner_fixture(1)

    def fake(_state, _roots, **_kwargs):
        scanner._stat_guard(Path("/synthetic/old.json"), [])

    scanner.advance = fake
    with pytest.raises(old.ExecutionPartitionFenceError):
        study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=1)


@pytest.mark.parametrize(
    "change", ["identity", "errors", "counter", "calls", "owner", "false-fresh"]
)
def test_resigned_verification_evidence_tampering_rejected(change):
    base, _receipt = study.begin_verification_epoch(verified_base())
    scanner, _seen = scanner_fixture(2, verification_steps=20)
    result = study.advance_verification(base, (Path("/synthetic"),), scanner, owner_index=2)
    if change == "identity":
        result["state"]["sample_ids"].append("invented")
    elif change == "errors":
        result["state"]["errors"] = []
    elif change == "counter":
        result["state"]["processed_files"] += 1
    elif change == "calls":
        result["summary"]["actual_stat_guard_calls"] = 0
    elif change == "owner":
        result["owner_index"] = 3
    else:
        result["summary"]["globally_fresh"] = True
    result["state_sha256"] = digest(result["state"])
    with pytest.raises(ValueError):
        study.verify_verification(base, result, owner_index=2)


def test_real_scanner_synthetic_global_owner_skip_and_current_verification(tmp_path):
    import evicastkv_review_freshness_stream as scanner

    roots = (tmp_path.resolve(),)
    paths = {}
    for owner in (3, 4):
        path = next(
            tmp_path / f"owner-{owner}-{index}.jsonl"
            for index in range(1000)
            if scanner.owner(tmp_path / f"owner-{owner}-{index}.jsonl", roots, 8) == owner
        )
        path.write_text('{"sample_id":"synthetic-owner-' + str(owner) + '","answer":"HIDDEN"}\n')
        paths[owner] = path
    state = scanner.initial_state(roots)
    state, _ = study.normalize_seed(state)
    result = study.advance_partition(
        state, state["frontier"], [str(roots[0])], roots, scanner, object(), owner_index=3
    )
    assert result["state"]["sample_ids"] == ["synthetic-owner-3"]
    assert str(paths[4]) not in result["state"]["bindings"]
    assert "HIDDEN" not in str(result)
    assert result["state"]["frontier"] == []
    epoch, receipt = study.begin_verification_epoch(result["state"])
    assert study.verify_epoch(result["state"], epoch, receipt)
    checked = study.advance_verification(epoch, roots, scanner, owner_index=3)
    assert checked["summary"]["actual_stat_guard_calls"] == 2
    assert checked["summary"]["verification_exhausted"] is True
    assert checked["summary"]["globally_fresh"] is False


@pytest.mark.parametrize("basename", ["auth.json", "credentials.json"])
@pytest.mark.parametrize("alias", [False, True])
def test_codex_credential_body_is_never_opened_hashed_or_projected(
    tmp_path, monkeypatch, basename, alias
):
    import evicastkv_review_freshness_stream as scanner

    protected = tmp_path / ".codex" / basename
    protected.parent.mkdir()
    protected.write_text('{"token":"SYNTHETIC-NOT-A-CREDENTIAL"}')
    path = protected
    if alias:
        path = tmp_path / "ordinary-looking.json"
        path.symlink_to(protected)
    roots = (tmp_path.resolve(),)
    queue = [dict(kind="file", path=str(path), signature=scanner.signature(path))]
    state = scanner.initial_state(roots)
    state.update(frontier=deepcopy(queue), traversal_complete=False)
    state = study.normalize_seed(state)[0]
    owner = scanner.owner(path, roots, 8)
    previous_process = scanner._process_file
    original_open = Path.open

    def guarded_open(value, *args, **kwargs):
        if value in (protected, path):
            raise AssertionError("protected credential body must not be opened")
        return original_open(value, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = study.advance_partition(
        state, queue, [str(roots[0])], roots, scanner, object(), owner_index=owner
    )
    assert scanner._process_file is previous_process
    assert result["state"]["frontier"] == []
    assert result["state"]["errors"] == [
        dict(path=str(path), error=study.EXCLUDED_CREDENTIAL_ERROR)
    ]
    assert result["state"]["bindings"] == {}
    assert result["state"]["processed_files"] == result["state"]["projected_records"] == 0
    assert result["summary"]["globally_fresh"] is False
    assert result["summary"]["new_errors"] == 1
    assert "SYNTHETIC-NOT-A-CREDENTIAL" not in str(result)


@pytest.mark.parametrize(
    "directory,basename", [(".codex", "config.json"), ("ordinary", "auth.json")]
)
def test_noncredential_files_delegate_to_original_processor(tmp_path, directory, basename):
    import evicastkv_review_freshness_stream as scanner

    path = tmp_path / directory / basename
    path.parent.mkdir()
    path.write_text('{"sample_id":"synthetic-normal-metadata"}')
    roots = (tmp_path.resolve(),)
    queue = [dict(kind="file", path=str(path), signature=scanner.signature(path))]
    state = scanner.initial_state(roots)
    state.update(frontier=deepcopy(queue), traversal_complete=False)
    state = study.normalize_seed(state)[0]
    result = study.advance_partition(
        state,
        queue,
        [str(roots[0])],
        roots,
        scanner,
        object(),
        owner_index=scanner.owner(path, roots, 8),
    )
    assert result["state"]["sample_ids"] == ["synthetic-normal-metadata"]
    assert result["state"]["processed_files"] == 1
    assert result["state"]["errors"] == []
