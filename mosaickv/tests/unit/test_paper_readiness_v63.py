import copy
import hashlib
import json
import os

import pytest

from mosaickv.evaluation.paper_readiness_v63 import (
    AUTHORIZATION_KEYS,
    BASE_SOURCE,
    CapsuleReader,
    build_decision,
    disabled_authorization,
)


@pytest.fixture
def inputs():
    authorization = dict.fromkeys(AUTHORIZATION_KEYS, False)
    counts = dict(
        total_artifacts=134,
        runtime_verified=20,
        partial_total=47,
        unbound=67,
        semantic_exposure_unresolved=135,
    )
    common = dict(
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )
    evidence = {
        "cpu": dict(
            common,
            status="passed",
            source_sha=BASE_SOURCE,
            CPU=dict(counts=dict(tests=10761, failures=0, errors=0, skipped=0)),
        ),
        "engineering": dict(
            common,
            source_fidelity=dict(
                status="passed_in_registered_lossless_BF16_MHA_scope",
                constructed_arms=4,
                consumed_inputs=2,
                predictions_per_input=128,
                maximum_logit_difference=0.0,
            ),
            whole_request_peak=dict(whole_device_GPU_peak_certified=False),
            independent_quality_confirmation=False,
            latency_superiority_established=False,
            six_component_benefit_established=False,
        ),
        "target_recovery": dict(common, counts_before_and_after=counts),
        "backup_recovery": dict(
            common,
            counts_before_and_after={k: v for k, v in counts.items() if k != "total_artifacts"},
            source_commits_recovered=0,
        ),
    }
    protocol = dict(
        authorization=dict(authorization),
        status="NO_CONFIG_FROZEN",
        candidate=dict(
            selected_config=None, selected_config_path=None, config_path=None, config_sha256=None
        ),
    )
    baselines = dict(authorization=dict(authorization))
    claims = dict(
        authorization=dict(authorization),
        requirements=[dict(id=f"R{i:02d}", status="blocked") for i in range(1, 11)],
    )
    return copy.deepcopy((evidence, protocol, baselines, claims))


def test_completed_certificates_produce_preparation_only_and_never_admission(inputs):
    assert len({id(asset["authorization"]) for asset in inputs[1:]}) == 3
    before = copy.deepcopy(inputs)
    result = build_decision(*inputs)
    assert inputs == before
    assert result["status"] == "BLOCKED_FOR_NEW_EXPERIMENTS"
    assert result["reusable_CPU_tests"] == 10761
    assert len(result["reviewer_requirements"]) == 10
    assert all(value is False for value in result["authorization"].values())
    for key in (
        "fresh_confirmation_authorized",
        "materialization_authorized",
        "paper_campaign_authorized",
        "superiority_established",
    ):
        assert result[key] is False


@pytest.mark.parametrize("asset", [1, 2, 3])
@pytest.mark.parametrize("field", AUTHORIZATION_KEYS)
@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_every_execution_flag_requires_literal_false(inputs, asset, field, value):
    inputs[asset]["authorization"][field] = value
    with pytest.raises(ValueError, match="explicitly false"):
        build_decision(*inputs)


@pytest.mark.parametrize("authorization", [None, {}, {"unknown_permission": True}])
def test_absent_or_incomplete_authorization_is_not_default_deny_evidence(authorization):
    with pytest.raises(ValueError):
        disabled_authorization(authorization)


def test_extra_permission_cannot_enable_unregistered_followup(inputs):
    inputs[1]["authorization"]["other_action"] = True
    with pytest.raises(ValueError):
        build_decision(*inputs)


@pytest.mark.parametrize(
    "field", ["selected_config", "selected_config_path", "config_path", "config_sha256"]
)
def test_no_candidate_is_inferred_or_silently_frozen(inputs, field):
    inputs[1]["candidate"][field] = "unqualified"
    with pytest.raises(ValueError, match="selected candidate"):
        build_decision(*inputs)


@pytest.mark.parametrize("value", [True, 10761.0, 1, "10761", None])
def test_reused_cpu_count_is_exact_typed_evidence(inputs, value):
    inputs[0]["cpu"]["CPU"]["counts"]["tests"] = value
    with pytest.raises(ValueError, match="CPU certificate"):
        build_decision(*inputs)


@pytest.mark.parametrize("field", ["constructed_arms", "consumed_inputs", "predictions_per_input"])
def test_scope_numerical_counts_cannot_be_boolean_or_float(inputs, field):
    inputs[0]["engineering"]["source_fidelity"][field] = True
    with pytest.raises(ValueError, match="scope changes"):
        build_decision(*inputs)


def test_scoped_peak_cannot_be_upgraded_to_whole_device_claim(inputs):
    inputs[0]["engineering"]["whole_request_peak"]["whole_device_GPU_peak_certified"] = True
    with pytest.raises(ValueError, match="scope changes"):
        build_decision(*inputs)


def test_changed_recovery_state_requires_new_review_not_automatic_progression(inputs):
    inputs[0]["target_recovery"]["counts_before_and_after"]["unbound"] = 0
    with pytest.raises(ValueError, match="separate review"):
        build_decision(*inputs)


@pytest.mark.parametrize("replacement", [[], [None], [dict(id="R01")] * 10])
def test_review_matrix_requires_all_ten_distinct_requirements(inputs, replacement):
    inputs[3]["requirements"] = replacement
    with pytest.raises(ValueError, match="ten Fox"):
        build_decision(*inputs)


def test_reader_reads_only_named_leaf_without_following_raw_artifact_links(tmp_path):
    body = json.dumps({"raw_link": "/never/open/this/raw.parquet"}).encode()
    (tmp_path / "gate.json").write_bytes(body)
    reader = CapsuleReader(tmp_path)
    assert reader.read("gate.json", hashlib.sha256(body).hexdigest()) == json.loads(body)
    assert len(reader.bindings) == 1
    assert reader.input_bytes == len(body)
    assert (
        reader.bindings[0]["computed_git_blob_oid"]
        == hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    )


@pytest.mark.parametrize(
    "relative",
    ["../outside.json", "/outside.json", "credential_fixture.json", ".codex/session.json"],
)
def test_reader_protected_paths_rejected_before_any_open(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("protected path opened"))
    with pytest.raises(ValueError):
        CapsuleReader(tmp_path).read(relative, None)


@pytest.mark.parametrize("ancestor", [False, True])
def test_reader_rejects_leaf_and_ancestor_symlinks(tmp_path, ancestor):
    (tmp_path / "actual.json").write_text("{}")
    if ancestor:
        (tmp_path / "redirect").symlink_to(tmp_path, target_is_directory=True)
        name = "redirect/actual.json"
    else:
        (tmp_path / "redirect.json").symlink_to(tmp_path / "actual.json")
        name = "redirect.json"
    with pytest.raises(OSError):
        CapsuleReader(tmp_path).read(name, None)


def test_reader_rejects_hardlinks(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    os.link(tmp_path / "a.json", tmp_path / "b.json")
    with pytest.raises(ValueError, match="regular-file"):
        CapsuleReader(tmp_path).read("b.json", None)


@pytest.mark.parametrize("body", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e400}', b"[]"])
def test_reader_rejects_ambiguous_nonfinite_or_nonobject_json(tmp_path, body):
    (tmp_path / "gate.json").write_bytes(body)
    with pytest.raises(ValueError):
        CapsuleReader(tmp_path).read("gate.json", None)


def test_reader_hash_mismatch_never_uses_current_file_as_new_certificate(tmp_path):
    (tmp_path / "gate.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        CapsuleReader(tmp_path).read("gate.json", "a" * 64)


def test_reader_keeps_original_deadline(tmp_path, monkeypatch):
    import mosaickv.evaluation.paper_readiness_v63 as module

    reader = CapsuleReader(tmp_path)
    monkeypatch.setattr(module.time, "monotonic", lambda: reader.started + 211)
    with pytest.raises(TimeoutError):
        reader.read("gate.json", None)


def test_reader_counts_and_bounds_json_keys_as_metadata(tmp_path, monkeypatch):
    import mosaickv.evaluation.paper_readiness_v63 as module

    monkeypatch.setattr(module, "MAX_SCALAR", 20)
    (tmp_path / "gate.json").write_text(json.dumps({"x" * 21: 0}))
    with pytest.raises(ValueError, match="scalar key"):
        CapsuleReader(tmp_path).read("gate.json", None)
    (tmp_path / "valid.json").write_text('{"a":1}')
    reader = CapsuleReader(tmp_path)
    assert reader.read("valid.json", None) == {"a": 1}
    assert reader.nodes == 3  # root object, key and value


def test_reader_conservative_live_metadata_cap_is_fail_closed(tmp_path, monkeypatch):
    import mosaickv.evaluation.paper_readiness_v63 as module

    monkeypatch.setattr(module, "MAX_BYTES", 256)
    (tmp_path / "gate.json").write_text('{"a":[1,2,3]}')
    with pytest.raises(ValueError, match="metadata"):
        CapsuleReader(tmp_path).read("gate.json", None)


def test_runner_preserves_failed_cpu_decision_and_never_reuses_output(tmp_path, monkeypatch):
    import prepare_fox_review_packet_v63 as runner

    revision = "a" * 40
    base = tmp_path / "outputs"
    run = base / ("run-20261114-" + revision[:7] + "-1")
    run.mkdir(parents=True)
    monkeypatch.setattr(runner, "OUTPUT_BASE", base)
    monkeypatch.setattr(runner, "clean_source", lambda *args: revision)

    def failed_cpu(*args):
        raise ValueError("synthetic failed CPU suite")

    monkeypatch.setattr(runner, "cpu_gate", failed_cpu)
    monkeypatch.setattr(runner, "CapsuleReader", lambda: CapsuleReader(tmp_path))
    output = run / "packet"
    assert runner.prepare(output) is False
    result = json.loads((output / "readiness_decision.json").read_text())
    assert result["status"] == "BLOCKED_EVIDENCE_INVALID"
    assert result["authorization"] == dict.fromkeys(AUTHORIZATION_KEYS, False)
    assert result["input_bindings"] == []
    assert result["raw_links_followed"] == 0
    before = (output / "readiness_decision.json").read_bytes()
    with pytest.raises(ValueError, match="new registered"):
        runner.prepare(output)
    assert (output / "readiness_decision.json").read_bytes() == before
