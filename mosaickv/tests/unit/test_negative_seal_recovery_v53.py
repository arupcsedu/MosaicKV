"""Synthetic source-seal projection; never historical Parquet/body I/O."""

import hashlib

import evicastkv_negative_seal_recovery_v53 as runner
import pytest

VALID = b"""classification: sealed_negative_label_blind_calibration_result
immutable_result: true
frozen_candidate_created: false
run_root: /scratch/djy8hg/runs/example
artifacts:
  ranking:
    path: analysis/ranking.parquet
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""


def test_only_source_contract_and_explicit_artifact_map_are_retained():
    raw = VALID + b"unwanted_results:\n  answer: PRIVATE_VALUE\n  score: 0.5\n"
    result = runner.project_yaml(raw, runner.prior.Budget())
    assert set(result) == runner.FIELDS
    assert result["immutable_result"] is True
    assert result["frozen_candidate_created"] is False
    assert result["artifacts"]["ranking"]["sha256"] == "a" * 64
    assert "PRIVATE_VALUE" not in repr(result)


@pytest.mark.parametrize(
    "raw",
    [
        VALID + b"run_root: /scratch/djy8hg/runs/other\n",
        VALID + b"ignored:\n  duplicate: 1\n  duplicate: 2\n",
        VALID + b"ignored: &alias value\n",
        VALID + b"ignored: *unknown\n",
        VALID + b"ignored: !!str tagged\n",
        VALID + b"ignored:\n  <<: {}\n",
        VALID + b"---\nother: document\n",
        b"[not, a, mapping]",
        VALID.replace(b"immutable_result: true", b"immutable_result: yes"),
        VALID.replace(b"frozen_candidate_created: false\n", b""),
        VALID + b"ignored: [unterminated\n",
    ],
)
def test_duplicate_alias_tag_merge_and_malformed_sources_fail_closed(raw):
    with pytest.raises((ValueError, runner.yaml.YAMLError, StopIteration)):
        runner.project_yaml(raw, runner.prior.Budget())


def test_unknown_deep_source_structure_keeps_depth_limit():
    raw = VALID + b"ignored: " + b"[" * 66 + b"0" + b"]" * 66
    with pytest.raises(ValueError, match="64-level"):
        runner.project_yaml(raw, runner.prior.Budget())


@pytest.mark.parametrize("position", ["key", "value"])
def test_unchanged_scalar_width_applies_to_keys_and_discarded_values(position):
    large = b"x" * (2**20 + 1)
    extra = b"? " + large + b"\n: ignored\n" if position == "key" else b"ignored: " + large
    with pytest.raises(ValueError, match="1-MiB"):
        runner.project_yaml(VALID + extra, runner.prior.Budget())


def test_optional_immutable_ledger_root_is_preserved_not_guessed():
    raw = VALID + b"immutable_ledger_root: /scratch/djy8hg/runs/old-ledgers\n"
    result = runner.project_yaml(raw, runner.prior.Budget())
    assert result["immutable_ledger_root"] == "/scratch/djy8hg/runs/old-ledgers"


def test_unmodified_node_limit_applies_to_discarded_values(monkeypatch):
    monkeypatch.setattr(runner.prior, "LIVE_NODES", 20)
    with pytest.raises(ValueError, match="node"):
        runner.project_yaml(VALID, runner.prior.Budget())


def test_unmodified_byte_limit(monkeypatch):
    monkeypatch.setattr(runner.prior, "LIVE_BYTES", 1)
    with pytest.raises(ValueError, match="8-MiB"):
        runner.project_yaml(VALID, runner.prior.Budget())


def test_unmodified_deadline():
    budget = runner.prior.Budget()
    budget.started -= 211
    with pytest.raises(TimeoutError):
        runner.project_yaml(VALID, budget)


def test_git_output_charges_bytes_and_rejects_excess(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"abcd")
    budget = runner.prior.Budget()
    assert runner.git_output(["synthetic"], budget, maximum=4) == b"abcd"
    assert budget.input_bytes == 4
    with pytest.raises(ValueError, match="bounded"):
        runner.git_output(["synthetic"], runner.prior.Budget(), maximum=3)


def test_git_output_does_not_run_when_budget_exhausted(monkeypatch):
    budget = runner.prior.Budget()
    budget.input_bytes = runner.prior.MAX_BYTES
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("Git read before budget authorization"),
    )
    with pytest.raises(ValueError, match="2-GiB"):
        runner.git_output(["synthetic"], budget, maximum=1)


def source_arguments():
    return dict(
        original_seal_commit="a" * 40,
        repository_path="mosaickv/configs/ledgers/choice_c_v95_negative_calibration_seal.yaml",
        git_blob_oid="b" * 40,
        sha256=hashlib.sha256(VALID).hexdigest(),
        targets=[],
    )


def test_old_source_and_current_blob_match_without_runtime_execution(monkeypatch):
    spec = source_arguments()
    outputs = iter([b"b" * 40 + b"\n", b"b" * 40 + b"\n", str(len(VALID)).encode(), VALID])
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: commands.append(a[0]))
    monkeypatch.setattr(runner, "git_output", lambda *a, **k: next(outputs))
    assert runner.original_source(spec, "c" * 40, "d" * 40, runner.prior.Budget()) == VALID
    assert commands[0][1:4] == ["--no-replace-objects", "merge-base", "--is-ancestor"]


@pytest.mark.parametrize("change", ["commit", "path", "blob", "digest"])
def test_forged_or_unavailable_original_seal_is_not_accepted(monkeypatch, change):
    spec = source_arguments()
    if change == "commit":
        spec["original_seal_commit"] = "HEAD"
    if change == "path":
        spec["repository_path"] = ".codex/credentials.json"
    if change == "blob":
        spec["git_blob_oid"] = "f" * 40
    if change == "digest":
        spec["sha256"] = "0" * 64
    outputs = iter([b"b" * 40 + b"\n", b"b" * 40 + b"\n", str(len(VALID)).encode(), VALID])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(runner, "git_output", lambda *a, **k: next(outputs))
    with pytest.raises(ValueError):
        runner.original_source(spec, "c" * 40, "d" * 40, runner.prior.Budget())
