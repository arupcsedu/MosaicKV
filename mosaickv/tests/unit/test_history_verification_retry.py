"""Reproduce V36's JSON/list-versus-tuple bookkeeping bug on synthetic metadata."""

import json
from copy import deepcopy
from pathlib import Path

import evicastkv_review_freshness_stream as scanner
import pytest

from mosaickv.evaluation import global_history_resume as legacy
from mosaickv.evaluation import history_verification_retry as retry
from mosaickv.evaluation.fox_campaign import digest


def fixture(tmp_path):
    path = tmp_path / "synthetic-source.json"
    path.write_text('{"answer":"NEVER READ THIS SYNTHETIC BODY"}')
    roots = (tmp_path.resolve(),)
    base = scanner.initial_state(roots)
    base.update(
        frontier=[],
        seen_files=[str(path)],
        bindings={str(path): dict(signature=scanner.signature(path))},
        directories={str(tmp_path): scanner.signature(tmp_path)},
        errors=[dict(path="/old/unresolved.yml", error="preserved error")],
        scopes=[["dataset", "revision", "split"]],
        scoped_identities=[["dataset", "revision", "split", "old-id"]],
        sample_ids=["old-id"],
        processed_files=1,
        projected_records=1,
        verification_paths=None,
        verification_index=0,
        traversal_complete=False,
        strict_immutable_boolean=True,
    )
    return legacy.normalize_seed(base)[0], roots, path


def run(base, roots):
    return retry.advance_verification(base, roots, scanner, owner_index=3)


def test_representative_saved_json_scopes_fail_old_validator_but_corrected_retry_passes(tmp_path):
    base, roots, _path = fixture(tmp_path)
    before = deepcopy(base)
    with pytest.raises(ValueError, match="changed traversal or identity"):
        legacy.advance_verification(base, roots, scanner, owner_index=3)
    result = run(base, roots)
    assert result["summary"]["verification_exhausted"] is True
    assert result["summary"]["actual_stat_guard_calls"] == 2
    assert result["summary"]["new_errors"] == 0
    assert result["state"]["scopes"] == base["scopes"]
    assert result["state"]["scoped_identities"] == base["scoped_identities"]
    assert result["state"]["errors"] == base["errors"]
    assert base == before
    assert retry.verify_verification(base, result, owner_index=3)


def test_result_json_roundtrip_and_hash_are_identical_after_container_normalization(tmp_path):
    base, roots, _path = fixture(tmp_path)
    result = run(base, roots)
    saved = json.loads(json.dumps(result))
    assert saved["state_sha256"] == digest(saved["state"])
    assert retry.verify_verification(base, saved, owner_index=3)
    assert saved["summary"]["globally_fresh"] is False
    assert saved["summary"]["fresh_confirmation_authorized"] is False


def test_verification_never_reads_historical_bodies_or_traverses_frontier(tmp_path, monkeypatch):
    base, roots, path = fixture(tmp_path)
    original_open = Path.open

    def guarded_open(value, *args, **kwargs):
        if value == path:
            raise AssertionError("verification must not open source bodies")
        return original_open(value, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(scanner, "_process_file", lambda *_: pytest.fail("no file projection"))
    monkeypatch.setattr(scanner, "_directory", lambda *_: pytest.fail("no rediscovery"))
    assert run(base, roots)["summary"]["actual_stat_guard_calls"] == 2


def test_changed_signature_remains_negative_new_error_and_preserves_old_errors(tmp_path):
    base, roots, path = fixture(tmp_path)
    path.write_text('{"longer_synthetic_body":"changed after original binding"}')
    result = run(base, roots)
    assert result["summary"]["new_errors"] == 1
    assert result["state"]["errors"][: len(base["errors"])] == base["errors"]
    assert result["state"]["errors"][-1]["path"] == str(path)
    assert result["summary"]["globally_fresh"] is False
    assert result["summary"]["verification_exhausted"] is True


def test_partial_current_epoch_resumes_cursor_without_reset_or_duplicate_stat(
    tmp_path, monkeypatch
):
    base, roots, _path = fixture(tmp_path)
    original_budget = scanner.Budget
    observed = []

    def budget(**kwargs):
        observed.append(kwargs)
        return original_budget(**kwargs, maximum_entries=1)

    monkeypatch.setattr(scanner, "Budget", budget)
    first = run(base, roots)
    assert first["state"]["verification_index"] == 1
    assert first["summary"]["verification_exhausted"] is False
    second = run(first["state"], roots)
    assert second["state"]["verification_index"] == 2
    assert second["summary"]["actual_stat_guard_calls"] == 1
    assert second["summary"]["verification_exhausted"] is True
    assert observed == [dict(seconds=210, maximum_bytes=2 * 2**30)] * 2


@pytest.mark.parametrize(
    "change",
    [
        "identity-value",
        "scope-value",
        "scope-order",
        "map",
        "counter",
        "bool-to-int",
        "old-error",
        "cursor",
        "calls",
        "owner",
        "fresh-pass",
        "unknown-container",
    ],
)
def test_resigned_mutations_still_fail_strict_validation(tmp_path, change):
    base, roots, _path = fixture(tmp_path)
    base["strict_unknown_array"] = [["keep"]]
    result = run(base, roots)
    if change == "identity-value":
        result["state"]["sample_ids"] = ["different-id"]
    elif change == "scope-value":
        result["state"]["scopes"][0][1] = "different-revision"
    elif change == "scope-order":
        result["state"]["scopes"][0].reverse()
    elif change == "map":
        result["state"]["bindings"].clear()
    elif change == "counter":
        result["state"]["processed_files"] += 1
    elif change == "bool-to-int":
        result["state"]["strict_immutable_boolean"] = 1
    elif change == "old-error":
        result["state"]["errors"].clear()
    elif change == "cursor":
        result["state"]["verification_index"] = 0
    elif change == "calls":
        result["summary"]["actual_stat_guard_calls"] = 0
    elif change == "owner":
        result["owner_index"] = 4
    elif change == "fresh-pass":
        result["summary"]["globally_fresh"] = True
    else:
        result["state"]["strict_unknown_array"] = [("keep",)]
    result["state_sha256"] = digest(result["state"])
    with pytest.raises(ValueError):
        retry.verify_verification(base, result, owner_index=3)


def test_standalone_validator_normalizes_only_a_view_without_mutating_saved_records(tmp_path):
    base, roots, _path = fixture(tmp_path)
    result = run(base, roots)
    result["state"]["scopes"] = [tuple(row) for row in result["state"]["scopes"]]
    result["state"]["scoped_identities"] = [
        tuple(row) for row in result["state"]["scoped_identities"]
    ]
    before_result, before_base = deepcopy(result), deepcopy(base)
    assert retry.verify_verification(base, result, owner_index=3)
    assert result == before_result and base == before_base


def test_failed_call_restores_legacy_validator_and_process_local_lock(tmp_path):
    base, roots, _path = fixture(tmp_path)
    original = legacy.verify_verification
    bad = deepcopy(base)
    bad["frontier"] = [dict(kind="directory", path=str(roots[0]))]
    with pytest.raises(ValueError):
        run(bad, roots)
    assert legacy.verify_verification is original
    assert run(base, roots)["summary"]["verification_exhausted"] is True
