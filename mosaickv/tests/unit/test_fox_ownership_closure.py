from __future__ import annotations

from copy import deepcopy

import pytest
from evicastkv_review_ownership_closure import (
    INSTALLED_HASHES,
    close_report,
    rule_definitions,
    unresolved_groups,
    validate_bindings,
    validate_capture,
    validate_rule_frames,
    write_compressed_once,
)

from mosaickv.measurements.kv_timeline import AllocationTimeline, valid_whole_request_certificate


def capture(*, role="unknown", name="construct", file="/reviewed.py", reused=False):
    timeline = AllocationTimeline()
    timeline.ingest(
        [
            dict(
                action="alloc",
                addr=128,
                size=64,
                time_us=100,
                stream=0,
                frames=[dict(filename=file, name=name, line=10)],
            ),
            dict(action="free_requested", addr=128, size=64, time_us=110, stream=0),
            dict(action="free_completed", addr=128, size=64, time_us=120, stream=0),
        ]
    )
    if role != "unknown":
        timeline.classify_range(0, 3, role, "explicit fixture operator")
    if reused:
        timeline.ingest(
            [
                dict(
                    action="alloc",
                    addr=128,
                    size=96,
                    time_us=130,
                    stream=0,
                    frames=[dict(filename="/unreviewed.py", name="other", line=20)],
                ),
            ]
        )
    return timeline.report(complete=True, history_complete=True, opaque_workspace_classified=False)


RULE = (("/reviewed.py", "construct", "kv_workspace", 10),)


def test_review_is_immutable_and_deduplicates_alias_observations():
    report = capture()
    report["records"][0]["evidence"] = ["view A aliases backing", "view B aliases backing"]
    original = deepcopy(report)
    result = close_report(report, RULE)
    assert report == original
    assert len(result["records"]) == 1
    assert result["exact_KV_peak_bytes"] == 64
    assert valid_whole_request_certificate(result)
    assert result["independent_certificate_verified"]


def test_pointer_reuse_is_a_new_generation_not_double_counted():
    report = capture(reused=True)
    result = close_report(report, (*RULE, ("/unreviewed.py", "other", "kv_workspace", 20)))
    assert result["exact_KV_peak_bytes"] == 96
    assert [(r["address"], r["generation"]) for r in result["records"]] == [(128, 0), (128, 1)]


def test_unknown_pointer_generation_never_inherits_reviewed_role():
    result = close_report(capture(reused=True), RULE)
    assert result["records"][1]["role"] == "unknown"
    assert result["unknown_allocation_generations"] == 1
    assert result["known_KV_peak_lower_bound_bytes"] == 64
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 96
    assert not valid_whole_request_certificate(result)
    groups = unresolved_groups(result)
    assert len(groups) == 1 and groups[0]["frames"][0][0] == "/unreviewed.py"
    assert groups[0]["largest_generation_bytes"] == 96


def test_rejected_timestamp_mark_is_not_assigned_to_latest_pointer():
    report = capture(reused=True)
    report["blocks_exact_certification"].append("ambiguous_deferred_owner_timestamp")
    result = close_report(report, RULE)
    assert "ambiguous_deferred_owner_timestamp" in result["blocks_exact_certification"]
    assert result["records"][1]["role"] == "unknown"
    assert not result["independent_certificate_verified"]


def test_independent_stacks_may_supersede_rejected_marks_only_when_complete():
    report = capture()
    report["blocks_exact_certification"].append("ambiguous_deferred_owner_timestamp")
    result = close_report(report, RULE)
    assert result["rejected_timestamp_marks_not_used_for_final_attribution"]
    assert result["independent_certificate_verified"]
    assert "ambiguous_deferred_owner_timestamp" in result["original_blocks_exact_certification"]


def test_unknown_is_not_non_kv_and_missing_lifecycle_cannot_be_repaired():
    report = capture(file="/never_reviewed.py")
    report["blocks_exact_certification"].append("allocator_history_incomplete_or_overwritten")
    result = close_report(report, RULE)
    assert result["records"][0]["role"] == "unknown"
    assert not result["independent_certificate_verified"]
    assert "allocator_history_incomplete_or_overwritten" in result["blocks_exact_certification"]


def test_version_hash_is_fail_closed(tmp_path):
    path = tmp_path / "reviewed.py"
    path.write_text("version one")
    from mosaickv.evaluation.fox_campaign import sha

    bindings = {str(path): sha(path)}
    validate_bindings(bindings)
    path.write_text("version two")
    with pytest.raises(ValueError, match="version hash mismatch"):
        validate_bindings(bindings)


def test_missing_source_hash_fails(tmp_path):
    with pytest.raises(ValueError, match="version hash mismatch"):
        validate_bindings({str(tmp_path / "missing.py"): "0" * 64})


def test_suffix_alias_cannot_substitute_an_unbound_source_file(tmp_path):
    report = capture(file=str(tmp_path / "other/reviewed.py"))
    with pytest.raises(ValueError, match="not bound"):
        validate_rule_frames(report, RULE, {str(tmp_path / "reviewed.py"): "0" * 64})


def test_peak_decomposition_is_simultaneous_not_sum_of_role_maxima():
    report = close_report(
        capture(reused=True), (*RULE, ("/unreviewed.py", "other", "kv_persistent", 20))
    )
    decomposition = report["peak_role_decomposition"]
    assert decomposition["total_bytes"] == 96
    assert sum(decomposition["roles"].values()) == 96
    assert decomposition["roles"]["kv_persistent"] == 96
    assert decomposition["roles"]["kv_workspace"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "negative", "overlap", "forged"])
def test_rejects_forged_lifetime_or_certificate(mutation):
    report = capture(reused=True)
    if mutation == "duplicate":
        report["records"][1] = deepcopy(report["records"][0])
    elif mutation == "negative":
        report["records"][0]["bytes"] = -1
    elif mutation == "overlap":
        report["records"][1]["start"] = 1
    else:
        report["exact_whole_request_KV_peak_certified"] = True
        report["exact_KV_peak_bytes"] = 1
    with pytest.raises(ValueError):
        validate_capture(report)


def test_pending_free_storage_is_live_until_free_completed():
    timeline = AllocationTimeline()
    timeline.ingest(
        [
            dict(
                action="alloc",
                addr=128,
                size=64,
                frames=[dict(filename="/reviewed.py", name="construct", line=10)],
            ),
            dict(action="free_requested", addr=128, size=64),
            dict(
                action="alloc",
                addr=256,
                size=32,
                frames=[dict(filename="/reviewed.py", name="construct", line=10)],
            ),
            dict(action="free_completed", addr=128, size=64),
        ]
    )
    report = timeline.report(
        complete=True, history_complete=True, opaque_workspace_classified=False
    )
    assert close_report(report, RULE)["exact_KV_peak_bytes"] == 96


def test_classified_rows_require_evidence():
    report = capture()
    report["records"][0]["role"] = "non_kv"
    with pytest.raises(ValueError, match="attribution evidence"):
        validate_capture(report)


def test_rules_never_exempt_generic_linear_or_broad_model_forward():
    definitions = rule_definitions()
    assert definitions
    for row in definitions:
        assert type(row["line"]) is int and row["line"] > 0
        assert row["reason"]
        assert row["file"] != "torch/nn/modules/linear.py"
        if not row["file"].startswith("mosaickv/"):
            assert row["file"] in INSTALLED_HASHES
    assert all(len(value) == 64 for value in INSTALLED_HASHES.values())


def test_compressed_report_is_write_once(tmp_path):
    import gzip
    import json

    path = tmp_path / "report.json.gz"
    write_compressed_once(path, {"immutable": True})
    with gzip.open(path, "rt") as stream:
        assert json.load(stream) == {"immutable": True}
    with pytest.raises(FileExistsError):
        write_compressed_once(path, {"immutable": False})


def test_prepare_and_cpu_run_preserve_all_six_raw_reports(tmp_path, monkeypatch):
    import json

    import evicastkv_review_ownership_closure as module

    from mosaickv.evaluation.fox_campaign import sha

    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "allocator").mkdir()
    operator = tmp_path / "reviewed.py"
    operator.write_text("# fixed fixture operator\n")
    monkeypatch.setattr(module, "clean_source", lambda root: "a" * 40)
    monkeypatch.setattr(
        module, "source_rules", lambda root: (RULE, {str(operator): sha(operator)}, [])
    )
    (parent / "protocol_lock.json").write_text(
        json.dumps({"source_sha": module.PRODUCER_SHA, "source_root": str(tmp_path)})
    )
    (parent / "analysis/GATE.json").write_text(
        json.dumps({"status": "passed_capture", "rows": 6, "errors": []})
    )
    summaries, raw_hashes = [], {}
    for index in range(6):
        raw = capture(file=str(operator))
        raw.update(run_id=f"run-{index}", sample_id=f"sample-{index // 3}")
        path = parent / f"allocator/run-{index}.json"
        path.write_text(json.dumps(raw))
        summaries.append(dict(allocator_path=str(path), allocator_sha256=sha(path)))
        raw_hashes[str(path)] = sha(path)
    (parent / "analysis/ownership.json").write_text(json.dumps(summaries))
    root = tmp_path / "new-derived-root"
    assert module.prepare(root, parent)["captures"] == 6
    result = module.run(root, parent)
    assert result["status"] == "passed_cpu_review"
    assert result["all_six_exact_KV_peaks_certified"]
    assert not result["new_gpu_inference"]
    assert len(list((root / "reviewed").glob("*.json.gz"))) == 6
    assert {path: sha(module.Path(path)) for path in raw_hashes} == raw_hashes
    with pytest.raises(FileExistsError):
        module.prepare(root, parent)
    with pytest.raises(FileExistsError):
        module.run(root)
