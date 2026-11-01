import pytest
from evicastkv_common_streaming import method_of, row_is_complete


@pytest.mark.parametrize(
    "variant, method",
    [
        ("source3_direct_fp32", "mosaickv_exact"),
        ("baseline_prompt_attention_topk__fp32", "prompt_attention_topk"),
        ("baseline_prefixkv_reimpl__fp32", "prefixkv_reimpl"),
        ("baseline_vl_cache_reimpl__fp32", "vl_cache_reimpl"),
        ("baseline_lookm_reimpl__fp32", "lookm_reimpl"),
    ],
)
def test_frozen_method_dispatch_is_explicit(variant, method):
    assert method_of(dict(evaluation_variant=variant)) == method


@pytest.mark.parametrize(
    "variant",
    [
        "full_kv",
        "baseline_lookm_official__fp32",
        "baseline_prefixkv_reimpl__fp16",
        "minimum__source3_direct_fp32",
    ],
)
def test_unregistered_or_lossy_header_fallback_labels_rejected(variant):
    with pytest.raises(ValueError):
        method_of(dict(evaluation_variant=variant))


def complete():
    return dict(
        status="completed",
        payload_and_memberships_identical=True,
        generation_128_logits_bitwise=True,
        complete_teacher_logits_bitwise=True,
        prefill_logits_bitwise=True,
        prefill_storage=dict(
            original_KV_tensor_objects_alive=0,
            CPU_offload_or_residual_bytes=0,
            final_prefill_cache_bytes=100,
            observed_prefill_cache_peak_bytes=200,
        ),
        generation_storage=dict(initial_active_kv_storage_bytes=100),
        teacher_forced=dict(evaluated_token_count=97),
    )


def test_complete_transport_requires_numerical_and_byte_evidence():
    assert row_is_complete(complete())
    for key in ("payload_and_memberships_identical", "complete_teacher_logits_bitwise"):
        row = complete()
        row[key] = False
        assert not row_is_complete(row)
    for key, value in (
        ("original_KV_tensor_objects_alive", 1),
        ("CPU_offload_or_residual_bytes", 20),
        ("final_prefill_cache_bytes", 99),
    ):
        row = complete()
        row["prefill_storage"][key] = value
        assert not row_is_complete(row)
    assert not row_is_complete(dict(status="failed"))


def test_negative_identity_gate_cannot_promote_to_timing_or_more_examples(monkeypatch, tmp_path):
    import evicastkv_common_streaming as module

    records = []
    monkeypatch.setattr(module.previous.common, "inputs", lambda *a: {"exact_minimum_only": True})
    monkeypatch.setattr(
        module.previous, "read", lambda *a: {"complete": False, "failures": ["bit mismatch"]}
    )
    monkeypatch.setattr(module, "write_once", lambda path, value: records.append(value))
    monkeypatch.setattr(
        module, "prepare", lambda *a, **k: pytest.fail("negative gate prepared a followup")
    )
    monkeypatch.setattr(
        module, "submit", lambda *a, **k: pytest.fail("negative gate submitted GPU work")
    )
    module.advance(tmp_path, "fake")
    assert records == [
        dict(
            authorized=False,
            reason="common exact-minimum identity gate did not pass",
            paper_campaign_authorized=False,
        )
    ]
