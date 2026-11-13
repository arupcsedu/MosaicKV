"""Synthetic gate capsules only; no historical proof, tensor, model, or file I/O."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import review_engineering_matrix_v52 as matrix
from mosaickv.evaluation.fox_campaign import digest


def parity():
    return dict(
        bitwise_logit_parity=True,
        pass_128=True,
        predictions=128,
        prediction_agreement=128,
        maximum_logit_difference=0.0,
    )


def fixture(monkeypatch):
    storage = {
        field: (10 if field.startswith("initial") else 20 if field.startswith("final") else 30)
        for field in matrix.STORAGE_FIELDS
    }
    observations = [
        dict(sample_id=sample, exact_bytes_matched=True, source=dict(storage), raw=dict(storage))
        for sample in matrix.SOURCE_SAMPLES
    ]
    native = dict(
        status="passed",
        stage="correctness",
        source_sha=matrix.NATIVE_PRODUCER,
        blockers=[],
        tests_actually_executed=dict(
            actual=128,
            completed=128,
            expected=128,
            pairing_count=64,
            workers_completed=4,
            workers_expected=4,
        ),
    )
    source = dict(
        source_sha=matrix.PRODUCER,
        plan_sha256=matrix.SOURCE_PLAN_SHA,
        complete=True,
        source_fidelity_128_pass=True,
        raw_same_selected_128_pass=True,
        whole_request_KV_byte_match_pass=True,
        failures=[],
        independent_quality_confirmation=False,
        paper_campaign_authorized=False,
        whole_request_GPU_peak_certified=False,
        observations=deepcopy(observations),
        completed_arms=[
            dict(
                worker=index,
                sample_id=matrix.SOURCE_SAMPLES[index // 2],
                arm="source" if index % 2 == 0 else "raw",
                parity=parity(),
            )
            for index in range(4)
        ],
    )
    raw_source = json.dumps(source).encode()
    source_sha = hashlib.sha256(raw_source).hexdigest()
    whole = dict(
        complete=True,
        source_fidelity_128_pass=True,
        whole_request_KV_byte_match_pass=True,
        whole_request_KV_peak_certified=True,
        independent_replay_from_actual_GPU_artifacts=True,
        source_sha=matrix.PRODUCER,
        actual_execution_source_sha=matrix.PRODUCER,
        validation_source_sha=matrix.NUMERICAL_PRODUCER,
        gate_origin="derived_validation_not_new_GPU_execution",
        actual_execution_root=matrix.SOURCE_ROOT,
        actual_execution_root_sha256=digest(matrix.SOURCE_ROOT),
        actual_plan_path=matrix.SOURCE_ROOT + "/plan.json",
        actual_plan_sha256=matrix.SOURCE_PLAN_SHA,
        actual_gate_path=matrix.SOURCE_ROOT + "/analysis/GATE_bounded_reader_v28.json",
        actual_gate_sha256=source_sha,
        input_gate_sha256=source_sha,
        input_plan_sha256=matrix.SOURCE_PLAN_SHA,
        failures=[],
        whole_request_GPU_peak_certified=False,
        independent_quality_confirmation=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
        observations=deepcopy(observations),
        records=[
            dict(
                worker=index,
                sample_id=matrix.SOURCE_SAMPLES[index // 2],
                arm="source" if index % 2 == 0 else "raw",
                parity=parity(),
                boundaries=dict(
                    storage,
                    whole_request_KV_peak_certified=True,
                    scope=matrix.KV_SCOPE,
                    decode_updates=127,
                    whole_request_GPU_peak_certified=False,
                ),
                native_role_crosslink=dict(
                    published_native_owner_ranges_bound=True,
                    retired_native_owner_ranges_absent=True,
                    verified_layers=32,
                ),
                ownership_artifact=matrix.SOURCE_ROOT
                + f"/workers/{index}/traces/whole_allocator_ownership.json",
                ownership_sha256="f" * 64,
                observation_count=1234,
            )
            for index in range(4)
        ],
    )
    coco = dict(
        complete=True,
        corrected_boundary_128_pass=True,
        historical_native_boundary_failure_preserved=True,
        source_sha=matrix.NUMERICAL_PRODUCER,
        plan_sha256=matrix.COCO_PLAN_SHA,
        source_scope="FP16_GQA_COCO_FullKV_evaluation_only_boundary_control",
        failures=[],
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
        production_defaults_changed=False,
        quality_or_latency_claim_allowed=False,
        source_reconstruction_validated=False,
        reports=[
            dict(
                index=index,
                sample_id=sample,
                pass_128=True,
                bitwise_logits=True,
                initial_cache_bitwise_equal=True,
                final_cache_bitwise_equal=True,
                prediction_agreement=128,
                predictions=128,
                ordinary_continuation_steps=127,
                unchanged_absolute_tolerance=1e-4,
                maximum_logit_difference=0.0,
                per_prediction_maximum_logit_difference=[0.0] * 128,
            )
            for index, sample in enumerate(matrix.COCO_SAMPLES)
        ],
    )
    values = dict(native=native, source=source, whole=whole, coco=coco)
    raws = {name: json.dumps(value).encode() for name, value in values.items()}
    bindings = {
        name: dict(matrix.KNOWN_BINDINGS[name], sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in raws.items()
    }
    monkeypatch.setattr(matrix, "KNOWN_BINDINGS", bindings)
    bridge = dict(
        actual_GPU_execution_base_sha=matrix.PRODUCER,
        new_control_source_sha=matrix.NUMERICAL_PRODUCER,
        all_original_producer_files_byte_identical=True,
        original_V9_transport_sha256=matrix.TRANSPORT_SHA,
        source_change_policy="additions_only_all_original_files_byte_identical",
        added_paths=["mosaickv/scripts/synthetic_cpu_tool.py"],
    )
    monkeypatch.setattr(matrix, "SOURCE_BRIDGE_SHA", digest(bridge))
    return values, raws, bridge


def run(raws, bridge):
    return matrix.validate_engineering_gates(raws, source_bridge=bridge)


def rebind(values, raws, name):
    raws[name] = json.dumps(values[name]).encode()
    matrix.KNOWN_BINDINGS[name]["sha256"] = hashlib.sha256(raws[name]).hexdigest()


def test_current_controls_are_positive_only_in_their_registered_scope(monkeypatch):
    _, raws, bridge = fixture(monkeypatch)
    original = deepcopy((raws, bridge))
    result = run(raws, bridge)
    assert (raws, bridge) == original
    assert result["source_fidelity"]["constructed_arms"] == 4
    assert result["source_fidelity"]["predictions_per_input"] == 128
    assert result["coco_numerical_controls"]["native_subset_checks"] == 128
    assert result["coco_numerical_controls"]["native_subset_inputs"] == 32
    assert result["whole_request_peak"]["whole_device_GPU_peak_certified"] is False
    assert result["whole_request_peak"]["memory_advantage_established"] is False
    assert result["earlier_lossy_source_failures_preserved"] is True
    assert result["earlier_native_boundary_failures_preserved"] is True
    for field in matrix.contract.FALSE_FIELDS:
        assert result[field] is False
    for field in (
        "superiority_established",
        "independent_quality_confirmation",
        "six_component_benefit_established",
        "latency_superiority_established",
        "raw_trajectory_or_ownership_proofs_replayed",
    ):
        assert result[field] is False


def test_matrix_is_pure_and_does_not_repeat_old_proofs(monkeypatch):
    _, raws, bridge = fixture(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure gate adapter accessed the filesystem")

    monkeypatch.setattr("builtins.open", forbidden)
    for method in ("open", "read_text", "read_bytes", "stat", "lstat", "resolve"):
        monkeypatch.setattr(Path, method, forbidden)
    assert run(raws, bridge)["GPU_jobs"] == 0


@pytest.mark.parametrize("name", ["native", "source", "whole", "coco"])
def test_missing_or_rehashed_current_artifacts_cannot_replace_original_pins(monkeypatch, name):
    _, raws, bridge = fixture(monkeypatch)
    changed = dict(raws)
    del changed[name]
    with pytest.raises(ValueError, match="all four"):
        run(changed, bridge)
    raws[name] += b" "
    with pytest.raises(ValueError, match="hash differs"):
        run(raws, bridge)


@pytest.mark.parametrize(
    "key",
    [
        "source_sha",
        "actual_execution_source_sha",
        "validation_source_sha",
        "actual_plan_sha256",
        "actual_gate_sha256",
        "actual_execution_root_sha256",
        "gate_origin",
    ],
)
def test_original_execution_and_derived_replay_provenance_cannot_be_substituted(monkeypatch, key):
    values, raws, bridge = fixture(monkeypatch)
    values["whole"][key] = "unreviewed-current-observation"
    rebind(values, raws, "whole")
    with pytest.raises(ValueError, match="provenance or scope"):
        run(raws, bridge)


@pytest.mark.parametrize("field", matrix.STORAGE_FIELDS)
def test_all_six_constructed_physical_quantities_must_match_not_only_configured_caps(
    monkeypatch, field
):
    values, raws, bridge = fixture(monkeypatch)
    values["whole"]["observations"][0]["raw"][field] += 1
    rebind(values, raws, "whole")
    with pytest.raises(ValueError, match="six actual physical"):
        run(raws, bridge)


@pytest.mark.parametrize(
    "key,value",
    [
        ("prediction_agreement", 127),
        ("predictions", 32),
        ("maximum_logit_difference", 1e-8),
        ("bitwise_logit_parity", False),
    ],
)
def test_source_passing_flag_without_actual_128_bitwise_predictions_is_rejected(
    monkeypatch, key, value
):
    values, raws, bridge = fixture(monkeypatch)
    values["source"]["completed_arms"][0]["parity"][key] = value
    rebind(values, raws, "source")
    with pytest.raises(ValueError):
        run(raws, bridge)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_arm",
        "wrong_worker",
        "duplicate_sample",
        "missing_sidecar_hash",
        "no_retirement",
        "31_layers",
        "wrong_boundary",
        "missing_observations",
        "GPU_peak_claim",
        "byte_match_flag_only",
    ],
)
def test_independent_ownership_certificate_requires_complete_actual_crosslinks(
    monkeypatch, failure
):
    values, raws, bridge = fixture(monkeypatch)
    whole = values["whole"]
    if failure == "missing_arm":
        whole["records"].pop()
    elif failure == "wrong_worker":
        whole["records"][0]["worker"] = 1
    elif failure == "duplicate_sample":
        whole["observations"][1]["sample_id"] = matrix.SOURCE_SAMPLES[0]
    elif failure == "missing_sidecar_hash":
        del whole["records"][0]["ownership_sha256"]
    elif failure == "no_retirement":
        whole["records"][0]["native_role_crosslink"]["retired_native_owner_ranges_absent"] = False
    elif failure == "31_layers":
        whole["records"][0]["native_role_crosslink"]["verified_layers"] = 31
    elif failure == "wrong_boundary":
        whole["records"][0]["boundaries"]["peak_tensor_backing_bytes"] += 1
    elif failure == "missing_observations":
        whole["records"][0]["observation_count"] = 0
    elif failure == "GPU_peak_claim":
        whole["whole_request_GPU_peak_certified"] = True
    else:
        whole["observations"] = []
    rebind(values, raws, "whole")
    with pytest.raises(ValueError):
        run(raws, bridge)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_input",
        "wrong_sample",
        "prediction31",
        "nonzero_error",
        "short_errors",
        "changed_tolerance",
        "cache_not_bitwise",
        "native_failure_erased",
        "source_GQA_claim",
        "quality_claim",
    ],
)
def test_coco_keeps_actual_counts_bitwise_errors_and_evaluation_only_scope(monkeypatch, failure):
    values, raws, bridge = fixture(monkeypatch)
    coco = values["coco"]
    if failure == "missing_input":
        coco["reports"].pop()
    elif failure == "wrong_sample":
        coco["reports"][0]["sample_id"] = "fresh-coco-not-authorized"
    elif failure == "prediction31":
        coco["reports"][0]["prediction_agreement"] = 31
    elif failure == "nonzero_error":
        coco["reports"][0]["per_prediction_maximum_logit_difference"][7] = 1e-8
    elif failure == "short_errors":
        coco["reports"][0]["per_prediction_maximum_logit_difference"].pop()
    elif failure == "changed_tolerance":
        coco["reports"][0]["unchanged_absolute_tolerance"] = 0.001
    elif failure == "cache_not_bitwise":
        coco["reports"][0]["final_cache_bitwise_equal"] = False
    elif failure == "native_failure_erased":
        coco["historical_native_boundary_failure_preserved"] = False
    elif failure == "source_GQA_claim":
        coco["source_reconstruction_validated"] = True
    else:
        coco["quality_or_latency_claim_allowed"] = True
    rebind(values, raws, "coco")
    with pytest.raises(ValueError):
        run(raws, bridge)


@pytest.mark.parametrize(
    "failure",
    [
        "modified_original",
        "not_additive",
        "nonadditive_path",
        "wrong_source",
        "invalid_analyzer",
        "forged_digest",
    ],
)
def test_cpu_source_compatibility_proof_is_hash_bound_and_must_not_modify_runtime(
    monkeypatch, failure
):
    _, raws, bridge = fixture(monkeypatch)
    if failure == "modified_original":
        bridge["all_original_producer_files_byte_identical"] = False
    elif failure == "not_additive":
        bridge["source_change_policy"] = "modifications_allowed"
    elif failure == "nonadditive_path":
        bridge["added_paths"] = ["mosaickv/src/runtime.py"]
    elif failure == "wrong_source":
        bridge["actual_GPU_execution_base_sha"] = "f" * 40
    elif failure == "invalid_analyzer":
        bridge["new_control_source_sha"] = "HEAD"
    else:
        bridge["original_V9_transport_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="CPU proof"):
        run(raws, bridge)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"])
def test_malformed_duplicate_or_nonfinite_gate_json_fails_closed(monkeypatch, raw):
    _, raws, bridge = fixture(monkeypatch)
    raws["native"] = raw
    matrix.KNOWN_BINDINGS["native"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        run(raws, bridge)


def test_gate_projected_bytes_remain_bounded(monkeypatch):
    _, raws, bridge = fixture(monkeypatch)
    monkeypatch.setattr(matrix, "MAXIMUM_BYTES", 3)
    with pytest.raises(ValueError, match="byte bound"):
        run(raws, bridge)


@pytest.mark.parametrize(
    "name,field", [("source", "completed_arms"), ("whole", "records"), ("coco", "reports")]
)
def test_explicit_record_shapes_are_required(monkeypatch, name, field):
    values, raws, bridge = fixture(monkeypatch)
    values[name][field][0] = []
    rebind(values, raws, name)
    with pytest.raises(ValueError):
        run(raws, bridge)


@pytest.mark.parametrize(
    "name,field,key",
    [
        ("source", "completed_arms", "worker"),
        ("whole", "records", "worker"),
        ("coco", "reports", "index"),
    ],
)
def test_boolean_record_indices_are_not_integer_worker_ids(monkeypatch, name, field, key):
    values, raws, bridge = fixture(monkeypatch)
    values[name][field][1][key] = True
    rebind(values, raws, name)
    with pytest.raises(ValueError):
        run(raws, bridge)


def test_combined_projected_node_limit_is_enforced(monkeypatch):
    _, raws, bridge = fixture(monkeypatch)
    monkeypatch.setattr(matrix, "MAXIMUM_NODES", 3)
    with pytest.raises(ValueError, match="node/depth"):
        run(raws, bridge)


def test_registered_bridge_paths_cannot_escape_the_source_tree(monkeypatch):
    _, raws, bridge = fixture(monkeypatch)
    bridge["added_paths"] = ["mosaickv/../escape.py"]
    monkeypatch.setattr(matrix, "SOURCE_BRIDGE_SHA", digest(bridge))
    with pytest.raises(ValueError, match="canonical unique"):
        run(raws, bridge)
