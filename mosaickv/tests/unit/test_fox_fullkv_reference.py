"""Recovery cannot waive model, schedule, storage, or different failure controls."""

import copy

import evicastkv_fullkv_reference as module
import pytest


@pytest.fixture
def fullkv():
    row = dict(
        status="failed",
        error=module.POSTCHECK_ERROR,
        method="full_kv",
        budget_bytes=None,
        run_id="fixture",
        sample_id="synthetic",
    )
    trace = dict(
        status="completed",
        method="full_kv",
        requested_method="full_kv",
        effective_method="full_kv",
        retention_ratio=1.0,
        tier_mode="full_cache",
        run_id="fixture",
        sample_id="synthetic",
        generated_token_ids=[1] * 64,
        answer="synthetic",
        decode_runtime_policy=dict(planned_decode_updates=63),
        timing_breakdown=dict(per_token_decode=[0.1] * 63),
        selected_blocks=[],
        prototypes=[],
        repair_events=[],
        prefixkv_decode_events=[],
        elastic_cache_decode_events=[],
        residual_kv_bytes=0,
        packed_padding_kv_bytes=0,
        prefill_attention_collection=dict(model_output_collector=False, private_module_hooks=False),
        source_kv_layout=[
            dict(
                key_shape=[1, 2, 4, 3],
                value_shape=[1, 2, 4, 3],
                key_element_size=2,
                value_element_size=2,
            )
        ],
        source_prefill_kv_bytes=96,
        cache_lifecycle=dict(
            initial_active_kv_bytes=96,
            initial_active_kv_storage_bytes=96,
            final_active_kv_bytes=1608,
            final_active_kv_storage_bytes=1608,
            peak_active_kv_storage_bytes=1608,
        ),
        quality_decoding=dict(timed_token_count=64, scored_token_count=8),
    )
    return row, trace


def test_completed_trace_can_be_recovered_without_mutating_failed_raw_record(fullkv):
    before = copy.deepcopy(fullkv)
    recovered = module.recover_trace(*fullkv)
    assert recovered["final_kv_bytes"] == 1608
    assert recovered["initial_kv_bytes"] == 96
    assert fullkv == before


@pytest.mark.parametrize(
    "change",
    [
        dict(error="CUDA OOM"),
        dict(status="completed"),
        dict(method="mosaickv_exact"),
        dict(budget_bytes=96),
        dict(sample_id="different"),
    ],
)
def test_unrelated_failure_or_identity_is_never_recovered(fullkv, change):
    row, trace = fullkv
    row.update(change)
    with pytest.raises(ValueError, match="identity FullKV"):
        module.recover_trace(row, trace)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_token_ids", [1] * 63),
        ("prototypes", ["prototype"]),
        ("selected_blocks", [1]),
        ("repair_events", [1]),
        ("packed_padding_kv_bytes", 1),
        ("residual_kv_bytes", 1),
    ],
)
def test_transformed_or_truncated_cache_cannot_be_a_fullkv_reference(fullkv, field, value):
    row, trace = fullkv
    trace[field] = value
    with pytest.raises(ValueError):
        module.recover_trace(row, trace)


@pytest.mark.parametrize(
    "field",
    [
        "initial_active_kv_storage_bytes",
        "final_active_kv_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    ],
)
def test_every_storage_boundary_must_reconcile(fullkv, field):
    row, trace = fullkv
    trace["cache_lifecycle"][field] += 1
    with pytest.raises(ValueError, match="accounting"):
        module.recover_trace(row, trace)


def test_all_original_exact_parity_controls_required():
    passed = dict(
        generated_tokens=16,
        token_agreement=1.0,
        reference_token_ids=[1] * 16,
        candidate_token_ids=[1] * 16,
        maximum_logit_difference=0.0,
    )
    parity = {
        k: copy.deepcopy(passed)
        for k in ("ordinary_generate", "retention_one", "mosaickv_runtime_retention_one")
    }
    module.parity_passed(parity)
    parity["ordinary_generate"]["maximum_logit_difference"] = 0.1
    with pytest.raises(ValueError, match="exactly"):
        module.parity_passed(parity)
