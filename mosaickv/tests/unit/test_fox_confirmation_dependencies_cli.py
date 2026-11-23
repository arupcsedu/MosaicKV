"""Synthetic CPU preparation/dispatch tests; never a fresh experiment."""

import builtins
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import evicastkv_confirmation_dependencies as cli
import pytest

from mosaickv.evaluation.fox_confirmation_inputs import ROUTE_SOURCE_HASHES
from mosaickv.evaluation.paper_readiness_v63 import CapsuleReader

ROOT = Path(__file__).parents[2]


def put(root, name, value=None, *, raw=None):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = raw if raw is not None else json.dumps(value, allow_nan=False).encode()
    path.write_bytes(body)
    return dict(path=name, sha256=hashlib.sha256(body).hexdigest())


@pytest.fixture
def dep_request(tmp_path):
    protocol = (ROOT / "configs/experiments/evicastkv_fox/review_completion_v63.json").read_bytes()
    config = (ROOT / "configs/experiments/evicastkv_fox/coco/mosaickv_exact.json").read_bytes()
    return dict(
        schema=cli.REQUEST_SCHEMA,
        protocol=put(tmp_path, "protocol.json", raw=protocol),
        config=put(tmp_path, "config.json", raw=config),
        route_sources=dict(ROUTE_SOURCE_HASHES),
        frozen=None,
        review=None,
        processor=None,
        proof_capsules={},
        confirmation_evidence=None,
        readiness_capsule=None,
    )


def test_current_unqualified_inputs_produce_blocked_plan_without_model_imports(
    tmp_path, dep_request, monkeypatch
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets"}:
            pytest.fail("model/data dependency imported before admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = cli.resolve_request(
        CapsuleReader(tmp_path), dep_request, source_sha="a" * 40, source_clean=False
    )
    assert result["plan"]["status"] == "BLOCKED_DEPENDENCIES"
    assert result["candidate"]["selected_config"] is None
    assert result["inputs"]["route"]["config_name"] == "default"
    assert result["inputs"]["processor_bound"] is False
    assert result["plan"]["scheduler_commands"] == []
    assert not any(result["plan"]["authorization"].values())
    assert len(result["evidence_bindings"]) == 2


@pytest.mark.parametrize(
    "field", ["delegation", "power_registration", "power_analysis", "cohort_disjointness"]
)
def test_explicit_additional_capsules_are_hash_checked_not_followed(tmp_path, dep_request, field):
    body = {"classification": "synthetic_parser_test_only", "raw_link": "/never/raw.parquet"}
    dep_request[field] = put(tmp_path, f"{field}.json", body)
    result = cli.resolve_request(
        CapsuleReader(tmp_path), dep_request, source_sha="a" * 40, source_clean=False
    )
    assert result["plan"]["prerequisite_metadata_validated"] is False
    assert len(result["evidence_bindings"]) == 3
    (tmp_path / f"{field}.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash differs"):
        cli.resolve_request(
            CapsuleReader(tmp_path), dep_request, source_sha="a" * 40, source_clean=False
        )


def test_preparation_is_write_once_and_explicitly_exploratory_when_dirty(
    tmp_path, dep_request, monkeypatch
):
    binding = put(tmp_path, "request.json", dep_request)
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda cmd, **kw: "a" * 40 + "\n" if cmd[1] == "rev-parse" else "?? pending-change.py\n",
    )
    out = tmp_path / "derived"
    result = cli.prepare(out, binding["path"], binding["sha256"], anchor=tmp_path)
    assert result["source_dirty"] is True
    assert result["completed_proofs_replayed"] == result["GPU_jobs"] == 0
    assert result["model_or_dataset_accessed"] is False
    assert (out / "confirmation_dependencies.json").exists()
    with pytest.raises(FileExistsError):
        cli.prepare(out, binding["path"], binding["sha256"], anchor=tmp_path)


@pytest.mark.parametrize("field", ["GPU_execution_authorized", "auto_start", "materialize"])
def test_extra_request_permissions_fail_before_reading_capsules(
    tmp_path, dep_request, monkeypatch, field
):
    dep_request[field] = True
    monkeypatch.setattr(cli, "read_capsule", lambda *a, **kw: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="unregistered"):
        cli.resolve_request(
            CapsuleReader(tmp_path), dep_request, source_sha="a" * 40, source_clean=True
        )


@pytest.mark.parametrize(
    "path", ["../escape.json", "raw.json", "auth.json", ".codex/auth.json", "/other/metadata.json"]
)
def test_protected_or_unregistered_paths_fail_before_open(tmp_path, monkeypatch, path):
    monkeypatch.setattr(
        os,
        "open",
        lambda *a, **kw: pytest.fail("protected path opened"),
    )
    with pytest.raises(ValueError):
        cli.read_capsule(CapsuleReader(tmp_path), dict(path=path, sha256="a" * 64))


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}'])
def test_duplicate_and_nonfinite_metadata_fail(tmp_path, raw):
    binding = put(tmp_path, "capsule.json", raw=raw)
    with pytest.raises(ValueError):
        cli.read_capsule(CapsuleReader(tmp_path), binding)


def test_changed_metadata_rejected_and_raw_links_are_not_followed(tmp_path):
    binding = put(tmp_path, "capsule.json", dict(raw_link="/never/read/raw.parquet"))
    assert cli.read_capsule(CapsuleReader(tmp_path), binding)["record"]["raw_link"].endswith(
        "parquet"
    )
    (tmp_path / "capsule.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash differs"):
        cli.read_capsule(CapsuleReader(tmp_path), binding)


def test_confirmation_worker_rejects_before_torch_import(tmp_path, monkeypatch):
    import evicastkv_worker as worker

    def blocked(*a, **kw):
        raise ValueError("dependencies remain blocked")

    monkeypatch.setattr(cli, "admit_confirmation_worker", blocked)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("Torch imported before failed admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="dependencies remain blocked"):
        worker.run_worker(tmp_path, "confirmation", 0, "a" * 64)


def test_development_stage_cannot_relabel_confirmation_cohort(tmp_path):
    lock = put(tmp_path, "protocol_lock.json", dict(cohorts={"coco": {"confirmation": {}}}))
    job = put(
        tmp_path,
        "jobs/scoring.json",
        dict(stage="scoring", protocol_sha256=lock["sha256"], jobs=[dict(subset="validation")]),
    )
    with pytest.raises(ValueError, match="separately admitted"):
        cli.reject_confirmation_relabel(tmp_path, "scoring", job["sha256"], 0)


def test_generic_submit_cannot_start_confirmation_or_read_old_lock(tmp_path, monkeypatch):
    import evicastkv_campaign as campaign

    monkeypatch.setattr(campaign, "load_lock", lambda *_: pytest.fail("old lock followed"))
    with pytest.raises(ValueError, match="auto-submitter"):
        campaign.submit(tmp_path, "confirmation")


def test_prospective_matrix_is_complete_and_deterministic():
    jobs = cli.confirmation_jobs()
    assert jobs == cli.confirmation_jobs() and len(jobs) == 32
    assert [i for job in jobs for i in job["indices"]] == list(range(256))
    assert sum(len(job["indices"]) * len(job["arms"]) for job in jobs) == 1792
    assert all(len(job["arms"]) == 7 for job in jobs)


def test_existing_confirmation_mode_invokes_cpu_preparation_only(tmp_path, monkeypatch):
    import evicastkv_campaign as campaign

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(campaign, "load_lock", lambda *_: pytest.fail("old lock followed"))
    assert (
        campaign.main(
            [
                "confirmation",
                "--root",
                str(tmp_path),
                "--dependency-request",
                "request.json",
                "--dependency-request-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    assert calls and "--request" in calls[0]


def worker_packet(tmp_path, monkeypatch, change=None):
    """Synthetic dispatch-only metadata, never actual qualification evidence."""
    source = "a" * 40
    ids = [f"synthetic-{i}" for i in range(256)]
    cohort = dict(
        ledger_path="ledger.json",
        ledger_sha256="1" * 64,
        sample_ids=ids,
        timing_sample_ids=ids[:32],
    )
    methods = {
        method: dict(path=f"configs/{method}.json", file_sha256="2" * 64)
        for method in ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    }
    lock = dict(
        source_sha=source,
        source_root=str(tmp_path),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        enabled_workloads=["coco"],
        budgets_mib={"coco": [20, 40]},
        seed=20271014,
        warmups=5,
        repeats=3,
        authorization=dict(automatic_stage_progression=False),
        cohorts={
            "coco": {
                "confirmation": dict(
                    path=str(tmp_path / "ledger.json"),
                    sha256=cohort["ledger_sha256"],
                    sample_ids=ids,
                )
            }
        },
        confirmation_timing_sample_ids=ids[:32],
        templates={"coco": dict(path="template.json", sha256="3" * 64)},
        historical_bindings={
            str(tmp_path / item["path"]): item["file_sha256"] for item in methods.values()
        },
        resources=dict(
            cpus_per_worker=12, memory_gib_per_worker=32, wall_minutes=120, maximum_a100s=6
        ),
    )
    if change:
        change(lock)
    runtime_lock = put(tmp_path, "protocol_lock.json", lock)
    current = dict(
        plan=dict(
            prerequisite_metadata_validated=True,
            reviewed_runtime_lock_sha256=runtime_lock["sha256"],
        ),
        candidate=dict(candidate_config_file_sha256="3" * 64),
        inputs={},
        confirmation_cohort=cohort,
        protocol_binding={},
        source_method_bindings=methods,
    )
    request_binding = put(tmp_path, "request.json", dict(schema="synthetic_mocked_resolution"))
    packet = put(tmp_path, "packet.json", dict(deepcopy(current), request_binding=request_binding))
    scope_record = dict(
        schema="evicastkv_confirmation_execution_scope_v1",
        source_sha=source,
        dependency_packet_sha256=packet["sha256"],
        dependency_request_sha256=request_binding["sha256"],
        protocol_lock_sha256=runtime_lock["sha256"],
        scope="COCO_confirmation_only_no_followup",
        explicit_execution_reviewed=True,
        fresh_confirmation_authorized=True,
        GPU_execution_authorized=True,
        experiment_submission_authorized=True,
        exclusion_scan_authorized=False,
        materialization_authorized=False,
        paper_admission_authorized=False,
        automatic_stage_progression=False,
    )
    scope = put(tmp_path, "execution_scope.json", scope_record)
    spec = dict(
        stage="confirmation",
        jobs=cli.confirmation_jobs(),
        protocol_sha256=runtime_lock["sha256"],
        confirmation_dependency_binding=packet,
        execution_scope_binding=scope,
    )
    monkeypatch.setattr(cli, "clean_source", lambda *_: source)
    monkeypatch.setattr(cli, "resolve_request", lambda *a, **kw: current)
    return spec, current, scope_record


def test_worker_bridge_accepts_only_mocked_complete_separately_scoped_metadata(
    tmp_path, monkeypatch
):
    spec, current, _ = worker_packet(tmp_path, monkeypatch)
    binding = put(tmp_path, "jobs/confirmation.json", spec)
    assert (
        cli.admit_confirmation_worker(
            tmp_path, binding["sha256"], 0, anchor=tmp_path, source=tmp_path
        )
        == current
    )
    assert not (tmp_path / "ledger.json").exists()  # The raw ledger was never opened.


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", 1),
        ("seed", True),
        ("warmups", 0),
        ("repeats", 1),
        ("inherited_gates", {"raw.json": "x"}),
        ("historical_bindings", {"raw.json": "a" * 64}),
    ],
)
def test_worker_bridge_rejects_changed_runtime_controls_even_with_reviewed_lock_hash(
    tmp_path, monkeypatch, key, value
):
    spec, _, _ = worker_packet(tmp_path, monkeypatch, lambda lock: lock.update({key: value}))
    binding = put(tmp_path, "jobs/confirmation.json", spec)
    with pytest.raises(ValueError):
        cli.admit_confirmation_worker(
            tmp_path, binding["sha256"], 0, anchor=tmp_path, source=tmp_path
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("fresh_confirmation_authorized", False),
        ("GPU_execution_authorized", False),
        ("paper_admission_authorized", True),
        ("automatic_stage_progression", True),
    ],
)
def test_worker_bridge_rejects_missing_or_broadened_scope(tmp_path, monkeypatch, key, value):
    spec, _, scope = worker_packet(tmp_path, monkeypatch)
    scope[key] = value
    spec["execution_scope_binding"] = put(tmp_path, "changed_scope.json", scope)
    binding = put(tmp_path, "jobs/confirmation.json", spec)
    with pytest.raises(ValueError, match="scope"):
        cli.admit_confirmation_worker(
            tmp_path, binding["sha256"], 0, anchor=tmp_path, source=tmp_path
        )


def test_worker_bridge_checks_runtime_lock_against_prospective_review(tmp_path, monkeypatch):
    spec, current, scope = worker_packet(tmp_path, monkeypatch)
    current["plan"]["reviewed_runtime_lock_sha256"] = "0" * 64
    request_binding = put(tmp_path, "other_request.json", {})
    spec["confirmation_dependency_binding"] = put(
        tmp_path,
        "changed_packet.json",
        dict(current, request_binding=request_binding),
    )
    scope["dependency_packet_sha256"] = spec["confirmation_dependency_binding"]["sha256"]
    scope["dependency_request_sha256"] = request_binding["sha256"]
    spec["execution_scope_binding"] = put(tmp_path, "changed_scope.json", scope)
    binding = put(tmp_path, "jobs/confirmation.json", spec)
    with pytest.raises(ValueError, match="runtime lock"):
        cli.admit_confirmation_worker(
            tmp_path, binding["sha256"], 0, anchor=tmp_path, source=tmp_path
        )
