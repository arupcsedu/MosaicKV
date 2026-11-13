"""Pure archive/generator fixtures; no source, filesystem, or execution access."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_parquet_binding_recovery_v49 import fixture as binding_fixture

from mosaickv.evaluation import source_archive_bindings_v51 as source
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding


def archive_fixture():
    report = dict(
        analysis_source_sha="a" * 40,
        analysis_worktree_dirty=True,
        source_code_hashes={
            "/scratch/djy8hg/code/writer.py": "b" * 64,
            "/scratch/djy8hg/code/helper.py": "c" * 64,
        },
        derived_artifact_hashes={
            "/scratch/djy8hg/runs/example/source.zip": "d" * 64,
            "/scratch/runs/example/summary.parquet": "a" * 64,
        },
    )
    witness = dict(
        path="/scratch/djy8hg/runs/example/source.zip",
        sha256="d" * 64,
        bytes=1000,
        members=[
            dict(name="00/writer.py", bytes=100, sha256="b" * 64),
            dict(name="01/helper.py", bytes=200, sha256="c" * 64),
        ],
    )
    return report, witness


def generator_fixture():
    report = dict(
        generator_sha256="b" * 64,
        producer_source_sha="c" * 40,
        outputs={"/scratch/runs/example/summary.parquet": "a" * 64},
    )
    witness = dict(path="/scratch/djy8hg/code/generator.py", sha256="b" * 64, bytes=100)
    return report, witness


def previous_binding():
    row, transaction, claims, writers = binding_fixture()
    claims[0]["role"] = "artifact_only"
    return classify_binding(row, transaction, claims, reviewed_writers=writers)


def test_complete_dirty_archive_keeps_every_source_and_explicit_noncanonical_scope():
    report, witness = archive_fixture()
    original = deepcopy((report, witness))
    result = source.validate_archive(report, witness)
    assert (report, witness) == original
    assert result["kind"] == "dirty_source_archive"
    assert result["analysis_base_commit"] == "a" * 40
    assert result["analysis_worktree_dirty"] is True
    assert result["expanded_source_bytes"] == 300
    assert len(result["source_files"]) == 2
    assert result["source_files"][0]["source_path"].startswith("/sfs/weka/scratch/")
    assert result["canonical_eligible"] is False
    assert result["original_producer_commit"] is None
    assert result["clean_runtime_execution_claimed"] is False
    assert all(result[name] is False for name in source.contract.FALSE_FIELDS)


def test_archive_validation_and_augmentation_do_no_io(monkeypatch):
    report, witness = archive_fixture()
    previous = previous_binding()

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure archive validation attempted I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("builtins.exec", forbidden)
    for name in ("open", "read_bytes", "read_text", "resolve", "stat", "lstat"):
        monkeypatch.setattr(Path, name, forbidden)
    closure = source.validate_archive(report, witness)
    result = source.augment_binding(previous, report["derived_artifact_hashes"], closure)
    assert result["status"] == source.DIRTY_STATUS


@pytest.mark.parametrize(
    "change",
    [
        "wrong_archive",
        "wrong_archive_hash",
        "dirty_false",
        "dirty_one",
        "input_field",
        "missing_source",
        "extra_source",
        "missing_member",
        "extra_member",
        "wrong_member_hash",
        "same_basename",
        "duplicate_member",
        "member_alias",
        "source_alias",
        "unknown_projection",
        "unknown_witness",
        "unknown_member",
    ],
)
def test_incomplete_mislabeled_or_extraneous_archive_closure_rejected(change):
    report, witness = archive_fixture()
    if change == "wrong_archive":
        witness["path"] = "/scratch/djy8hg/runs/example/other.zip"
    elif change == "wrong_archive_hash":
        witness["sha256"] = "f" * 64
    elif change == "dirty_false":
        report["analysis_worktree_dirty"] = False
    elif change == "dirty_one":
        report["analysis_worktree_dirty"] = 1
    elif change == "input_field":
        report["producer_source_sha"] = report.pop("analysis_source_sha")
    elif change == "missing_source":
        del report["source_code_hashes"]["/scratch/djy8hg/code/helper.py"]
    elif change == "extra_source":
        report["source_code_hashes"]["/scratch/djy8hg/code/extra.py"] = "e" * 64
    elif change == "missing_member":
        witness["members"].pop()
    elif change == "extra_member":
        witness["members"].append(dict(name="02/extra.py", bytes=20, sha256="e" * 64))
    elif change == "wrong_member_hash":
        witness["members"][0]["sha256"] = "f" * 64
    elif change == "same_basename":
        report["source_code_hashes"]["/scratch/djy8hg/other/writer.py"] = "e" * 64
    elif change == "duplicate_member":
        witness["members"].append(deepcopy(witness["members"][0]))
    elif change == "member_alias":
        witness["members"].append(dict(witness["members"][0], name="other/writer.py"))
    elif change == "source_alias":
        report["source_code_hashes"]["/sfs/weka/scratch/djy8hg/code/writer.py"] = "b" * 64
    elif change == "unknown_projection":
        report["canonical_eligible"] = True
    elif change == "unknown_witness":
        witness["executed"] = True
    else:
        witness["members"][0]["symlink_target"] = "writer.py"
    with pytest.raises(ValueError):
        source.validate_archive(report, witness)


@pytest.mark.parametrize(
    "name",
    [
        "../writer.py",
        "/writer.py",
        "00/../writer.py",
        "00//writer.py",
        "00/./writer.py",
        ".codex/writer.py",
        "00/auth.json",
        "00/credentials.json",
        "00/.env",
        "00/.netrc",
        "00\\writer.py",
    ],
)
def test_archive_member_path_never_authorizes_alias_traversal_or_credentials(name):
    report, witness = archive_fixture()
    witness["members"][0]["name"] = name
    with pytest.raises(ValueError):
        source.validate_archive(report, witness)


@pytest.mark.parametrize("count", [0, -1, True, source.MAXIMUM_BYTES + 1])
@pytest.mark.parametrize("where", ["archive", "member"])
def test_archive_and_members_retain_unchanged_positive_byte_bounds(count, where):
    report, witness = archive_fixture()
    target = witness if where == "archive" else witness["members"][0]
    target["bytes"] = count
    with pytest.raises(ValueError):
        source.validate_archive(report, witness)


def test_total_uncompressed_source_size_is_charged_not_only_zip_bytes():
    report, witness = archive_fixture()
    witness["members"][0]["bytes"] = source.MAXIMUM_BYTES
    with pytest.raises(ValueError, match="expanded-source"):
        source.validate_archive(report, witness)


def test_ad_hoc_generator_keeps_input_producer_separate_from_its_code_identity():
    report, witness = generator_fixture()
    result = source.validate_generator(report, witness)
    assert result["kind"] == "ad_hoc_generator"
    assert result["input_producer_source_sha"] == "c" * 40
    assert result["input_producer_is_generator_claimed"] is False
    assert result["original_producer_commit"] is None
    assert result["generator_sha256"] == "b" * 64
    assert result["canonical_eligible"] is False


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "input_as_writer",
        "bad_input_sha",
        "missing_output",
        "bad_bytes",
        "unsafe_path",
        "extra_field",
    ],
)
def test_generator_binding_rejects_source_role_substitution_and_missing_evidence(change):
    report, witness = generator_fixture()
    if change == "hash":
        witness["sha256"] = "f" * 64
    elif change == "input_as_writer":
        report["generator_source_sha"] = report.pop("producer_source_sha")
    elif change == "bad_input_sha":
        report["producer_source_sha"] = "bad"
    elif change == "missing_output":
        report["outputs"] = {}
    elif change == "bad_bytes":
        witness["bytes"] = True
    elif change == "unsafe_path":
        witness["path"] = "/scratch/djy8hg/.codex/auth.json"
    else:
        witness["source_sha"] = "c" * 40
    with pytest.raises(ValueError):
        source.validate_generator(report, witness)


@pytest.mark.parametrize("kind", ["archive", "generator"])
def test_binding_extension_preserves_exact_old_bytes_and_all_negative_scope(kind):
    previous = previous_binding()
    original = deepcopy(previous)
    report, witness = archive_fixture() if kind == "archive" else generator_fixture()
    closure = (
        source.validate_archive(report, witness)
        if kind == "archive"
        else source.validate_generator(report, witness)
    )
    artifacts = report["derived_artifact_hashes"] if kind == "archive" else report["outputs"]
    result = source.augment_binding(previous, artifacts, closure)
    assert previous == original
    assert result["previous_binding_sha256"] == digest(previous)
    assert result["artifact_sha256"] == previous["artifact_sha256"]
    assert result["status"] == (
        source.DIRTY_STATUS if kind == "archive" else source.GENERATOR_STATUS
    )
    assert result["original_producer_commit"] is None
    assert result["canonical_eligible"] is False
    assert result["newly_resolved_occurrences"] == result["GPU_jobs"] == 0
    assert all(result[name] is False for name in source.contract.FALSE_FIELDS)


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "producer",
        "writer",
        "gpu",
        "resolved",
        "fresh",
        "artifact_hash",
        "map_hash",
        "different_map",
        "closure_flag",
        "closure_hash",
        "closure_kind",
    ],
)
def test_extension_cannot_hide_changed_bytes_or_promote_scientific_permissions(change):
    previous = previous_binding()
    report, witness = archive_fixture()
    closure = source.validate_archive(report, witness)
    artifacts = report["derived_artifact_hashes"]
    if change == "status":
        previous["status"] = "runtime_producer_and_artifact_binding_verified"
    elif change == "producer":
        previous["original_producer_commit"] = "a" * 40
    elif change == "writer":
        previous["original_producer_writer_path"] = "mosaickv/scripts/source.py"
    elif change == "gpu":
        previous["GPU_jobs"] = 1
    elif change == "resolved":
        previous["occurrence_resolved"] = True
    elif change == "fresh":
        previous["fresh_confirmation_authorized"] = True
    elif change == "artifact_hash":
        previous["artifact_sha256"] = "f" * 64
    elif change == "map_hash":
        artifacts[previous["path"]] = "f" * 64
    elif change == "different_map":
        artifacts["/scratch/runs/example/extra.parquet"] = "e" * 64
    elif change == "closure_flag":
        closure["canonical_eligible"] = True
    elif change == "closure_hash":
        closure["source_content_sha256"] = "f" * 64
    else:
        closure["kind"] = "clean_runtime"
    with pytest.raises(ValueError):
        source.augment_binding(previous, artifacts, closure)


def test_oversized_projection_and_node_count_fail_without_widening_limits(monkeypatch):
    report, witness = archive_fixture()
    assert source.MAXIMUM_BYTES == 8 * 2**20
    assert source.MAXIMUM_NODES == 100000
    monkeypatch.setattr(source, "MAXIMUM_NODES", 2)
    with pytest.raises(ValueError, match="node/depth"):
        source.validate_archive(report, witness)
