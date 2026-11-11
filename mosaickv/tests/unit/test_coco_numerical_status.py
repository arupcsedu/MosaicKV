"""Synthetic policy fixtures, not new GPU measurements or benchmark evidence."""

from copy import deepcopy

import pytest

from mosaickv.evaluation import coco_numerical_status as numerical


def native():
    return dict(
        status="passed",
        stage="correctness",
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


def fixture():
    report = dict(
        index=0,
        passed=True,
        first_logit=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
        continuation=dict(
            reference_token_ids=list(range(16)),
            candidate_token_ids=list(range(16)),
            predictions=16,
            token_agreement=1.0,
            reference_prefix_forced_on_both_paths=True,
            ordinary_continuation_after_boundary=True,
            per_prediction_maximum_logit_difference=[0.0] * 16,
            maximum_logit_difference=0.0,
            unchanged_absolute_tolerance=1e-4,
            initial_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
            final_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
            operation_counts=dict(
                observed_prefill_layers=32,
                attention_padded_calls=32,
                linear_padded_calls=224,
                normalization_expected_calls=65,
                normalization_padded_calls=65,
                normalization_input_calls=32,
                normalization_post_attention_calls=32,
                normalization_final_calls=1,
                normalization_phase_entries=1,
                normalization_phase_completions=1,
            ),
        ),
    )
    second = deepcopy(report)
    second["index"] = 1
    gate = dict(
        source_sha=numerical.SHAPE_SOURCE,
        status="passed_new_shape_transport",
        complete=True,
        failures=[],
        unchanged_tolerance=1e-4,
        historical_native_gate_still_negative=True,
        native_norm_boundary=True,
        ordinary_continuation_after_boundary=True,
        shared_transport_only=True,
        production_default_changed=False,
        quality_or_latency_claim_allowed=False,
        fresh_confirmation_authorized=False,
        reports=[report, second],
    )
    transport = dict(
        continuation_predictions=16,
        tolerance=1e-4,
        native_norm_boundary=True,
        native_historical_gate_superseded=False,
        no_output_or_logit_patch=True,
        online_default=False,
    )
    inputs = dict(samples=[dict(sample_id=value) for value in numerical.CONSUMED_IDS])
    config = dict(
        model=numerical.MODEL,
        method="full_kv",
        execution=dict(backend="huggingface", attention_implementation="eager"),
        cache=dict(retention_ratio=1.0),
    )
    return gate, transport, inputs, config


def test_native_checks_are_not_128_predictions():
    result = numerical.validate_native(native())
    assert result["executed_checks"] == 128
    assert result["consumed_inputs"] == 32
    assert result["checks_per_input"] == 4
    assert result["continuation_128_predictions_established"] is False


@pytest.mark.parametrize(
    "field",
    ["actual", "completed", "expected", "pairing_count", "workers_completed", "workers_expected"],
)
def test_native_missing_or_boolean_counts_rejected(field):
    record = native()
    record["tests_actually_executed"][field] = True
    with pytest.raises(ValueError, match="counts"):
        numerical.validate_native(record)


def test_boundary_retains_16_per_input_and_negative_native_history():
    result = numerical.validate_shape(*fixture())
    assert result["consumed_inputs"] == 2
    assert result["predictions_per_input"] == 16
    assert result["total_predictions"] == 32
    assert result["native_historical_gate_still_negative"] is True
    assert result["continuation_128_predictions_established"] is False
    assert result["online_default_changed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "native_norm_boundary",
        "ordinary_continuation_after_boundary",
        "shared_transport_only",
        "historical_native_gate_still_negative",
    ],
)
def test_positive_scope_requires_strict_boolean(field):
    values = fixture()
    values[0][field] = 1
    with pytest.raises(ValueError, match="scope drift"):
        numerical.validate_shape(*values)


@pytest.mark.parametrize(
    "field",
    [
        "production_default_changed",
        "quality_or_latency_claim_allowed",
        "fresh_confirmation_authorized",
    ],
)
def test_diagnostic_cannot_authorize_broader_claim(field):
    values = fixture()
    values[0][field] = True
    with pytest.raises(ValueError, match="scope drift"):
        numerical.validate_shape(*values)


@pytest.mark.parametrize(
    "change",
    [
        "tokens",
        "logits",
        "counts",
        "cache",
        "identity",
        "model",
        "retention",
        "tolerance",
        "online",
        "prediction_count",
    ],
)
def test_negative_or_scope_expanded_boundary_rejected(change):
    gate, transport, inputs, config = values = fixture()
    continuation = gate["reports"][0]["continuation"]
    if change == "tokens":
        continuation["candidate_token_ids"][1] = 99
    elif change == "logits":
        continuation["per_prediction_maximum_logit_difference"][1] = 0.0078125
    elif change == "counts":
        continuation["operation_counts"]["normalization_padded_calls"] = 66
    elif change == "cache":
        continuation["initial_cache"]["bitwise_equal"] = False
    elif change == "identity":
        inputs["samples"][0]["sample_id"] = "unregistered"
    elif change == "model":
        config["model"] = dict(config["model"], precision="bf16")
    elif change == "retention":
        config["cache"]["retention_ratio"] = 0.5
    elif change == "tolerance":
        gate["unchanged_tolerance"] = 0.02
    elif change == "online":
        transport["online_default"] = True
    else:
        continuation["predictions"] = 128
    with pytest.raises(ValueError):
        numerical.validate_shape(*values)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -0.001])
def test_invalid_logit_differences_rejected(nonfinite):
    values = fixture()
    values[0]["reports"][1]["continuation"]["per_prediction_maximum_logit_difference"][8] = (
        nonfinite
    )
    with pytest.raises(ValueError, match="threshold"):
        numerical.validate_shape(*values)


def test_changed_immutable_evidence_rejected_before_JSON_decode(monkeypatch):
    monkeypatch.setattr(numerical, "sha", lambda _path: "changed")
    with pytest.raises(ValueError, match="content changed"):
        numerical.audit()
