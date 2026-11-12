"""Synthetic alias-recovery proofs; never historical inference or data access."""

import hashlib
import json
import time
from pathlib import PurePosixPath

import evicastkv_derived_alias_recovery_v42 as helper
import pytest


@pytest.fixture
def case(tmp_path, monkeypatch):
    real = tmp_path / "canonical"
    alias = tmp_path / "scratch"
    real.mkdir()
    alias.symlink_to(real, target_is_directory=True)
    parent, derived = real / "original", real / "derived"
    (parent / "analysis").mkdir(parents=True)
    derived.mkdir()
    monkeypatch.setattr(helper, "ALIAS_PREFIX", PurePosixPath(str(alias)))
    monkeypatch.setattr(helper, "CANONICAL_PREFIX", PurePosixPath(str(real)))

    def write(path, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    original = parent / "analysis/report.json"
    original_sha = write(original, b'{"incomplete":')
    plan_sha = write(parent / "plan.json", dict(source_sha=helper.PRODUCER, answer="PRIVATE_PLAN"))
    patch_sha = write(derived / "analysis-source.patch", b"synthetic unchanged patch")
    artifacts = {str(alias / "original/analysis/report.json"): original_sha}
    for name in ("samples.parquet", "summary.parquet"):
        expected = write(parent / "analysis" / name, b"synthetic saved artifact")
        write(derived / name, b"synthetic saved artifact")
        artifacts[str(alias / "original/analysis" / name)] = expected
    report_sha = write(
        derived / "report.json",
        dict(
            status="passed_execution",
            completed_rows=64,
            expected_rows=64,
            plan_sha256=plan_sha,
            sample_id="consumed",
            answer="PRIVATE_REPORT",
        ),
    )
    provenance = dict(
        parent_root=str(alias / "original"),
        original_analysis_artifacts=artifacts,
        gpu_reruns=0,
        original_analysis_preserved=True,
        producer_artifacts_verified=True,
        producer_git_sha=helper.PRODUCER,
        parent_plan_sha256=plan_sha,
        analyzer_patch_sha256=patch_sha,
    )
    provenance_sha = write(derived / "analysis_provenance.json", provenance)
    for name, value in dict(
        ORIGINAL=original,
        DERIVED=derived,
        ORIGINAL_SHA=original_sha,
        PROVENANCE_SHA=provenance_sha,
        REPORT_SHA=report_sha,
    ).items():
        monkeypatch.setattr(helper, name, value)
    row = dict(path=str(original), signature=helper.prior.signature(original))
    return dict(
        row=row,
        roots=[real],
        parent=parent,
        derived=derived,
        alias=alias,
        provenance=provenance,
        write=write,
        monkeypatch=monkeypatch,
    )


def run(case):
    return helper.prove(case["row"], case["roots"], time.monotonic() + 10)


def replace_provenance(case):
    actual = case["write"](case["derived"] / "analysis_provenance.json", case["provenance"])
    case["monkeypatch"].setattr(helper, "PROVENANCE_SHA", actual)


def test_scratch_aliases_compare_equal_without_rewriting_the_manifest(case):
    original = (case["derived"] / "analysis_provenance.json").read_bytes()
    proof = run(case)
    assert proof["identities"]["sample_ids"] == ["consumed"]
    assert (
        proof["full_file_sha256_before"] == proof["full_file_sha256_after"] == helper.ORIGINAL_SHA
    )
    assert proof["canonical_alias_normalization"]["parent_recorded_path"] != str(case["parent"])
    assert not proof["canonical_alias_normalization"]["original_manifest_rewritten"]
    assert (case["derived"] / "analysis_provenance.json").read_bytes() == original
    assert not proof["fresh_confirmation_authorized"]


def test_equal_duplicate_alias_hash_is_explicit_and_accepted(case):
    case["provenance"]["original_analysis_artifacts"][str(helper.ORIGINAL)] = helper.ORIGINAL_SHA
    replace_provenance(case)
    proof = run(case)
    assert len(proof["canonical_alias_normalization"]["original_artifact_bindings"]) == 4


def test_conflicting_alias_hashes_are_not_normalized_away(case):
    case["provenance"]["original_analysis_artifacts"][str(helper.ORIGINAL)] = "e" * 64
    replace_provenance(case)
    with pytest.raises(ValueError, match="conflicting hashes"):
        run(case)


@pytest.mark.parametrize(
    "change", ["producer", "parent", "report-hash", "plan-producer", "saved-data", "patch"]
)
def test_original_unchanged_controls_remain_required(case, change):
    if change == "producer":
        case["provenance"]["producer_git_sha"] = "e" * 40
        replace_provenance(case)
    elif change == "parent":
        case["provenance"]["parent_root"] = str(case["alias"] / "different")
        replace_provenance(case)
    elif change == "report-hash":
        case["monkeypatch"].setattr(helper, "REPORT_SHA", "e" * 64)
    elif change == "plan-producer":
        actual = case["write"](case["parent"] / "plan.json", dict(source_sha="e" * 40))
        case["provenance"]["parent_plan_sha256"] = actual
        replace_provenance(case)
    elif change == "saved-data":
        (case["derived"] / "samples.parquet").write_bytes(b"changed")
    else:
        (case["derived"] / "analysis-source.patch").write_bytes(b"changed")
    with pytest.raises(ValueError):
        run(case)


def test_known_alias_spelling_is_allowed_on_the_requested_report(case):
    case["row"]["path"] = str(case["alias"] / "original/analysis/report.json")
    assert run(case)["path"] == case["row"]["path"]


def test_nonregistered_files_and_outside_aliases_are_rejected_before_open(case, monkeypatch):
    case["row"]["path"] = "/outside/report.json"
    monkeypatch.setattr(helper.prior, "read", lambda *a, **k: pytest.fail("outside body opened"))
    with pytest.raises(ValueError, match="prefix alias pair"):
        run(case)


def test_report_and_plan_benchmark_values_are_never_decoded(case, monkeypatch):
    loads = json.loads

    def guarded(value, *args, **kwargs):
        assert "PRIVATE" not in str(value), "benchmark value decoded"
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    assert run(case)["identities"]["sample_ids"] == ["consumed"]


def test_missing_saved_hash_and_wrong_original_signature_are_rejected(case):
    key = str(case["alias"] / "original/analysis/samples.parquet")
    case["provenance"]["original_analysis_artifacts"].pop(key)
    replace_provenance(case)
    with pytest.raises(ValueError, match="missing"):
        run(case)
    case["row"]["signature"][2] += 1
    with pytest.raises(ValueError, match="signature"):
        run(case)


@pytest.mark.parametrize("path", ["/outside/x", "/scratch/djy8hg/../x", "relative", None])
def test_unknown_or_ambiguous_path_normalization_is_rejected(path):
    with pytest.raises(ValueError):
        helper.normalize(path)
