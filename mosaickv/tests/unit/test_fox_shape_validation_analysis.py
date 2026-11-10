"""Synthetic saved-capture ABI and provenance tests; never model/GPU inference."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.types import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    PreparedInputs,
)
from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.evaluation.fox_campaign import digest, sha
from mosaickv.evaluation.fox_shape_transport import remove_dummy_cache_slot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("evicastkv_boundary_shape_validation")
    root = tmp_path / "shape-control"
    root.mkdir()
    revision = "a" * 40

    def clean(source, expected=None):
        if expected is not None and expected != revision:
            raise ValueError("synthetic fixture source SHA mismatch")
        return revision

    monkeypatch.setattr(module, "clean_source", clean)
    monkeypatch.setattr(module.boundary, "clean_source", clean)
    samples = [
        dict(sample_id=f"fixture-{i}", prompt="synthetic input only", media=[]) for i in range(2)
    ]
    write(root / "inputs.json", dict(samples=samples))
    write(root / "resolved_config.json", dict(synthetic_fixture=True))
    prior = root / "prior/GATE_boundary_linear.json"
    write(prior, dict(status="passed_operator_diagnostic", native_operator_replay_all_matched=True))
    protocol = dict(
        source_root=str(module.SOURCE),
        source_sha=revision,
        inputs_sha256=sha(root / "inputs.json"),
        resolved_config_sha256=sha(root / "resolved_config.json"),
        config_sha256="c" * 64,
        bindings={str(prior): sha(prior)},
        sample_count=2,
        maximum_capture_bytes_per_input=64 * 2**20,
        fullkv_chunked_first_logit_absolute_tolerance=1e-4,
    )
    write(root / "protocol.json", protocol)
    transport = dict(
        source_sha=revision,
        base_protocol_sha256=sha(root / "protocol.json"),
        prior_gate=str(prior),
        prior_gate_sha256=sha(prior),
        transport="dummy-prefix-plus-native-shape-matched-linear-and-attention",
        implementation_sha256=sha(
            module.SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
        ),
        continuation_predictions=16,
        tolerance=1e-4,
        same_dtype_and_weights=True,
        placeholder_is_not_held_token=True,
        native_historical_gate_superseded=False,
        headline_timing_eligible=False,
        online_default=False,
        fresh_confirmation_authorized=False,
    )
    write(root / "transport.json", transport)
    write(root / "environment.json", dict(source_sha=revision, config_sha256="c" * 64))
    records = []
    for i, sample in enumerate(samples):
        name = "layer.0.attention.q_proj:input"
        entry = dict(
            vector=torch.zeros(1, 1, 8, dtype=torch.float16),
            source_shape=[1, 5, 8],
            dtype="torch.float16",
        )
        capture = dict(
            captures=dict(A={name: entry}, B={name: copy.deepcopy(entry)}),
            orders=dict(A=[name], B=[name]),
            logits_A=torch.zeros(1, 12, dtype=torch.float32),
            logits_B=torch.zeros(1, 12, dtype=torch.float32),
            held_token_forward_count=1,
            original_logical_length=5,
            prefix_logical_length=4,
            captured_bytes=32,
            source_KV_modified=False,
            compressed_cache_used=False,
            shape_transport_continuation=dict(
                predictions=16,
                reference_token_ids=[0] * 16,
                candidate_token_ids=[0] * 16,
                token_agreement=1.0,
                maximum_logit_difference=0.0,
                per_prediction_maximum_logit_difference=[0.0] * 16,
                unchanged_absolute_tolerance=1e-4,
                reference_prefix_forced_on_both_paths=True,
                initial_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
                final_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
                operation_counts=dict(
                    linear_padded_calls=14, attention_padded_calls=2, observed_prefill_layers=2
                ),
            ),
            new_backend_variant_not_native_history_repair=True,
        )
        path = root / f"captures/{i:02d}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(capture, path)
        records.append(
            dict(
                index=i,
                sample_id=sample["sample_id"],
                status="completed",
                capture_path=str(path),
                capture_sha256=sha(path),
                input_sha256=digest(sample),
                captured_bytes=32,
            )
        )
    result = dict(
        source_sha=revision,
        protocol_sha256=sha(root / "protocol.json"),
        environment_sha256=sha(root / "environment.json"),
        measurement_type="operator_diagnostic_not_timing_or_scoring",
        records=records,
    )
    write(root / "result.json", result)
    return SimpleNamespace(
        root=root,
        module=module,
        result=result,
        protocol=protocol,
        transport=transport,
        samples=samples,
    )


def mutate_capture(artifacts, callback, index=0):
    record = artifacts.result["records"][index]
    path = Path(record["capture_path"])
    capture = torch.load(path, map_location="cpu", weights_only=True)
    callback(capture)
    torch.save(capture, path)
    record["capture_sha256"] = sha(path)
    write(artifacts.root / "result.json", artifacts.result)


def test_actual_records_abi_produces_new_transport_gate_not_native_waiver(artifacts):
    root, module = artifacts.root, artifacts.module
    original = (root / "result.json").read_bytes()
    gate = module.analyze(root)
    assert gate["status"] == "passed_new_shape_transport"
    assert gate["complete"] and len(gate["reports"]) == 2
    assert gate["historical_native_gate_still_negative"]
    assert gate["production_default_changed"] is False
    assert gate["quality_or_latency_claim_allowed"] is False
    assert gate["fresh_confirmation_authorized"] is False
    assert gate["unchanged_tolerance"] == 1e-4
    assert (root / "result.json").read_bytes() == original
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    assert manifest["result_sha256"] == sha(root / "result.json")
    assert manifest["gate_sha256"] == sha(root / "GATE_shape_transport.json")
    assert manifest["evidence_type"] == "new_evaluation_transport_not_native_gate_waiver"


def test_analyzer_is_write_once_preserving_original_artifacts(artifacts):
    artifacts.module.analyze(artifacts.root)
    with pytest.raises(FileExistsError):
        artifacts.module.analyze(artifacts.root)


def test_failed_second_case_is_preserved_with_denominator_two(artifacts):
    artifacts.result["records"][1] = dict(
        index=1, sample_id="fixture-1", status="failed", error="synthetic capture failed"
    )
    write(artifacts.root / "result.json", artifacts.result)
    gate = artifacts.module.analyze(artifacts.root)
    assert gate["status"] == "blocked_new_shape_transport"
    assert gate["complete"] is False
    assert len(gate["reports"]) == 1
    assert gate["failures"][0]["sample_id"] == "fixture-1"


@pytest.mark.parametrize("key", ["source_sha", "protocol_sha256", "environment_sha256"])
def test_result_provenance_drift_rejected_before_comparison(artifacts, key):
    artifacts.result[key] = "f" * 64
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError, match="provenance"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize(
    "key", ["base_protocol_sha256", "prior_gate_sha256", "implementation_sha256"]
)
def test_transport_implementation_and_prior_bindings_are_checked(artifacts, key):
    artifacts.transport[key] = "f" * 64
    write(artifacts.root / "transport.json", artifacts.transport)
    with pytest.raises(ValueError, match="binding"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize(
    "key,value",
    [
        ("continuation_predictions", 8),
        ("tolerance", 0.01),
        ("native_historical_gate_superseded", True),
        ("online_default", True),
    ],
)
def test_scope_tolerance_or_native_gate_waiver_is_rejected(artifacts, key, value):
    artifacts.transport[key] = value
    write(artifacts.root / "transport.json", artifacts.transport)
    with pytest.raises(ValueError, match="scope or tolerance"):
        artifacts.module.analyze(artifacts.root)


def test_saved_environment_tampering_is_rejected(artifacts):
    write(artifacts.root / "environment.json", dict(source_sha="other"))
    with pytest.raises(ValueError, match="provenance"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize("mutation", ["missing", "reverse", "duplicate", "sample_id"])
def test_complete_ordered_two_case_grid_is_required(artifacts, mutation):
    if mutation == "missing":
        artifacts.result["records"].pop()
    elif mutation == "reverse":
        artifacts.result["records"].reverse()
    elif mutation == "duplicate":
        artifacts.result["records"][1] = artifacts.result["records"][0]
    else:
        artifacts.result["records"][1]["sample_id"] = "unregistered"
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError, match="complete ordered"):
        artifacts.module.analyze(artifacts.root)


def test_capture_hash_mismatch_is_rejected(artifacts):
    artifacts.result["records"][0]["capture_sha256"] = "f" * 64
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError, match="capture changed"):
        artifacts.module.analyze(artifacts.root)


def test_capture_cannot_escape_registered_root(artifacts):
    old = Path(artifacts.result["records"][0]["capture_path"])
    outside = artifacts.root.parent / "outside.pt"
    outside.write_bytes(old.read_bytes())
    artifacts.result["records"][0]["capture_path"] = str(outside)
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError):
        artifacts.module.analyze(artifacts.root)


def test_numerical_failure_is_retained_not_rounded_into_pass(artifacts):
    def negative(capture):
        capture["logits_B"].fill_(0.001)
        difference = float((capture["logits_B"] - capture["logits_A"]).abs().max())
        capture["shape_transport_continuation"]["per_prediction_maximum_logit_difference"][0] = (
            difference
        )
        capture["shape_transport_continuation"]["maximum_logit_difference"] = difference

    mutate_capture(artifacts, negative)
    gate = artifacts.module.analyze(artifacts.root)
    assert gate["complete"]
    assert gate["status"] == "blocked_new_shape_transport"
    assert gate["reports"][0]["first_logit"]["maximum_absolute_difference"] > 1e-4


def test_attention_transport_must_actually_engage(artifacts):
    mutate_capture(
        artifacts,
        lambda c: c["shape_transport_continuation"]["operation_counts"].update(
            attention_padded_calls=0
        ),
    )
    gate = artifacts.module.analyze(artifacts.root)
    assert gate["status"] == "blocked_new_shape_transport"


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "short_tokens",
        "short_differences",
        "wrong_agreement",
        "wrong_maximum",
        "held_twice",
        "logical_length",
        "source_modified",
        "compressed",
    ],
)
def test_tampered_continuation_contract_is_rejected(artifacts, mutation):
    def mutate(capture):
        continuation = capture["shape_transport_continuation"]
        if mutation == "count":
            continuation["predictions"] = 15
        elif mutation == "short_tokens":
            continuation["candidate_token_ids"].pop()
        elif mutation == "short_differences":
            continuation["per_prediction_maximum_logit_difference"].pop()
        elif mutation == "wrong_agreement":
            continuation["candidate_token_ids"][1] = 2
        elif mutation == "wrong_maximum":
            continuation["maximum_logit_difference"] = 0.01
        elif mutation == "held_twice":
            capture["held_token_forward_count"] = 2
        elif mutation == "logical_length":
            capture["original_logical_length"] += 1
        elif mutation == "source_modified":
            capture["source_KV_modified"] = True
        else:
            capture["compressed_cache_used"] = True

    mutate_capture(artifacts, mutate)
    with pytest.raises(ValueError, match="continuation or unchanged FullKV"):
        artifacts.module.analyze(artifacts.root)


def test_input_sha_must_match_registered_sample(artifacts):
    artifacts.result["records"][0]["input_sha256"] = "f" * 64
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError, match="input changed"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01])
def test_per_prediction_errors_must_be_finite_and_nonnegative(artifacts, bad):
    def mutate(capture):
        # A NaN after the first element can be hidden by Python max([0, NaN]).
        continuation = capture["shape_transport_continuation"]
        continuation["per_prediction_maximum_logit_difference"][1] = bad
        continuation["maximum_logit_difference"] = max(
            continuation["per_prediction_maximum_logit_difference"]
        )

    mutate_capture(artifacts, mutate)
    with pytest.raises(ValueError, match="continuation or unchanged FullKV"):
        artifacts.module.analyze(artifacts.root)


def test_first_reported_error_must_equal_recomputed_saved_logits(artifacts):
    mutate_capture(artifacts, lambda c: c["logits_B"].fill_(0.001))
    with pytest.raises(ValueError, match="continuation or unchanged FullKV"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize(
    "flag",
    ["reference_prefix_forced_on_both_paths", "new_backend_variant_not_native_history_repair"],
)
def test_variant_and_common_prefix_scope_flags_cannot_be_disabled(artifacts, flag):
    def mutate(capture):
        owner = (
            capture
            if flag == "new_backend_variant_not_native_history_repair"
            else capture["shape_transport_continuation"]
        )
        owner[flag] = False

    mutate_capture(artifacts, mutate)
    with pytest.raises(ValueError, match="continuation or unchanged FullKV"):
        artifacts.module.analyze(artifacts.root)


@pytest.mark.parametrize("key", ["source_sha", "config_sha256"])
def test_rebound_environment_still_must_match_protocol_identity(artifacts, key):
    environment = json.loads((artifacts.root / "environment.json").read_text())
    environment[key] = "f" * 64
    write(artifacts.root / "environment.json", environment)
    artifacts.result["environment_sha256"] = sha(artifacts.root / "environment.json")
    write(artifacts.root / "result.json", artifacts.result)
    with pytest.raises(ValueError, match="environment source/config"):
        artifacts.module.analyze(artifacts.root)


def test_trimmed_prefix_restores_all_valid_flag_and_private_mask():
    n = 4
    key, value = torch.randn(1, 2, n + 1, 3), torch.randn(1, 2, n + 1, 3)
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(key, value, 2),), tuple, "tuple", n + 1, CachedKeyState.POST_ROPE
    )
    adapter = SimpleNamespace(
        borrow_past_key_values=lambda _: snapshot,
        inject_past_key_values=lambda s: tuple((layer.key, layer.value) for layer in s.layers),
    )
    prefix = PreparedInputs(
        dict(input_ids=torch.arange(n)[None], attention_mask=torch.ones(1, n, dtype=torch.long)),
        (ModalitySpan(0, n, Modality.TEXT),),
        n,
    )
    state = DecodeState(
        ((key, value),),
        torch.tensor([[1, 1, 1, 1, 0]]),
        n + 1,
        n + 1,
        n + 1,
        (),
        dict(mosaickv_cache_all_valid=False),
    )
    trimmed = remove_dummy_cache_slot(adapter, state, prefix)
    assert trimmed.model_state["mosaickv_cache_all_valid"] is True
    assert state.model_state["mosaickv_cache_all_valid"] is False
    assert trimmed.attention_mask.data_ptr() != prefix.model_inputs["attention_mask"].data_ptr()
    assert (
        trimmed.past_key_values[0][0].untyped_storage().data_ptr()
        != key.untyped_storage().data_ptr()
    )
