from __future__ import annotations

import inspect
import time
from types import SimpleNamespace

import evicastkv_source_precision_control as subject
import pytest
import torch


def test_actual_operator_is_required_before_checkpoint_loader():
    calls = []
    result, model = subject.operator_before_model(
        lambda: calls.append("operator") or dict(passed=True),
        lambda: calls.append("checkpoint") or "model",
    )
    assert calls == ["operator", "checkpoint"]
    assert result["passed"] and model == "model"


@pytest.mark.parametrize("result", [{}, {"passed": False}, {"passed": "yes"}])
def test_operator_failure_cannot_start_pretrained_inference(result):
    with pytest.raises(ValueError, match="before checkpoint"):
        subject.operator_before_model(lambda: result, lambda: pytest.fail("model loaded"))


def fixture():
    modules = [SimpleNamespace(self_attn=object()) for _ in range(2)]
    adapter = SimpleNamespace(_language_layers=lambda: modules)
    normalized = SimpleNamespace(records={i: torch.tensor([float(i)]) for i in range(2)})
    return adapter, modules, normalized


def packet(bits=8):
    data = torch.zeros(3)
    return SimpleNamespace(
        source=SimpleNamespace(bits=bits), kind=subject.LABEL, tensors=lambda: (data,)
    )


def test_scoped_fit_maps_attention_to_original_layer_without_reference_access():
    from mosaickv.evaluation import shared_source_format

    adapter, modules, normalized = fixture()
    calls = []
    old = shared_source_format.fit_transport

    def constructor(parent, original_source, attention):
        index = next(i for i, module in enumerate(modules) if module.self_attn is attention)
        assert original_source is normalized.records[index]
        assert parent == f"unpatched-three-bit-{index}"
        calls.append(index)
        return packet(), dict(source_bits=8)

    def builder(*args):
        assert args[-1] == "source3_attention_optional"
        outputs = [
            shared_source_format.fit_transport(f"unpatched-three-bit-{i}", layer.self_attn)[0]
            for i, layer in enumerate(modules)
        ]
        return outputs, [{"layer": i, "mask_sha256": str(i)} for i in range(2)]

    formats, proof, records = subject.eight_bit_formats(
        adapter, None, None, None, normalized, None, builder=builder, constructor=constructor
    )
    assert calls == [0, 1]
    assert len(formats) == len(proof) == len(records) == 2
    assert shared_source_format.fit_transport is old
    assert not {"prefix", "labels", "reference", "logits"}.intersection(
        inspect.signature(subject.eight_bit_formats).parameters
    )


def test_fit_patch_is_restored_after_constructor_failure():
    from mosaickv.evaluation import shared_source_format

    adapter, modules, normalized = fixture()
    original = shared_source_format.fit_transport

    def builder(*args):
        return shared_source_format.fit_transport("parent", modules[0].self_attn)

    with pytest.raises(RuntimeError, match="constructor failed"):
        subject.eight_bit_formats(
            adapter,
            None,
            None,
            None,
            normalized,
            None,
            builder=builder,
            constructor=lambda *args: (_ for _ in ()).throw(RuntimeError("constructor failed")),
        )
    assert shared_source_format.fit_transport is original


def test_duplicate_attention_identity_is_rejected():
    adapter, modules, normalized = fixture()
    modules[1].self_attn = modules[0].self_attn
    with pytest.raises(ValueError, match="unique original attention"):
        subject.eight_bit_formats(adapter, None, None, None, normalized, None)


def test_wrong_precision_or_missing_layer_never_silently_falls_back():
    from mosaickv.evaluation import shared_source_format

    adapter, modules, normalized = fixture()

    def builder(*args):
        result = [shared_source_format.fit_transport(None, layer.self_attn)[0] for layer in modules]
        return result, []

    with pytest.raises(ValueError, match="silent fallback"):
        subject.eight_bit_formats(
            adapter,
            None,
            None,
            None,
            normalized,
            None,
            builder=builder,
            constructor=lambda *args: (packet(bits=3), {}),
        )


def test_constructor_deadline_precedes_any_transform():
    from mosaickv.evaluation import shared_source_format

    adapter, modules, normalized = fixture()
    with pytest.raises(TimeoutError):
        subject.eight_bit_formats(
            adapter,
            None,
            None,
            None,
            normalized,
            None,
            builder=lambda *args: shared_source_format.fit_transport(None, modules[0].self_attn),
            constructor=lambda *args: pytest.fail("constructor ran"),
            deadline=time.monotonic() - 1,
        )


@pytest.mark.parametrize("bad", ["fallback", "unretired", "no_savings", "mask_changed"])
def test_request_enforces_original_retirement_storage_and_mask_policy(tmp_path, monkeypatch, bad):
    import evicastkv_review_source_controls as source

    from mosaickv.evaluation import storage_runs

    pair = (torch.zeros(1, 1, 4, 128).bfloat16(), torch.ones(1, 1, 4, 128).bfloat16())
    raw = sum(item.untyped_storage().nbytes() for item in pair)
    adapter = SimpleNamespace(_legacy_layers=lambda value: (value, None))
    prefill = SimpleNamespace(state=SimpleNamespace(past_key_values=(pair,)))
    prefix = tuple(range(128))
    calls = []

    def formats(*args, **kwargs):
        assert len(args) == 6
        assert "prefix" not in kwargs
        return (
            (packet(),),
            [dict(layer=0, mask_sha256="changed" if bad == "mask_changed" else "frozen")],
            [],
        )

    monkeypatch.setattr(subject, "eight_bit_formats", formats)

    def decode(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["forced"] is prefix
        assert kwargs["initial_physical_byte_cap"] == raw
        return torch.zeros(128, 4), dict(
            storage=dict(initial_active_kv_storage_bytes=raw if bad == "no_savings" else raw - 1),
            used_fullkv_fallback=bad == "fallback",
            inactive_original_source_retired=bad != "unretired",
            raw_source_kv_bytes=raw,
        )

    monkeypatch.setattr(storage_runs, "run", decode)

    def request(*args):
        built, _ = source._formats(
            adapter, None, prefill, None, None, None, "source3_attention_optional"
        )
        return storage_runs.run(adapter, prefill, built, tokens=128)

    monkeypatch.setattr(source, "_request", request)
    with pytest.raises(ValueError):
        subject.request(
            adapter,
            {},
            None,
            prefix=prefix,
            expected_protection=[dict(layer=0, mask_sha256="frozen")],
        )
    assert bool(calls) is (bad != "mask_changed")


def test_malformed_prefix_rejected_before_runtime():
    with pytest.raises(ValueError, match="complete fixed FullKV prefix"):
        subject.request(None, {}, None, prefix=(1,), expected_protection=[])


def test_adapter_deadline_and_restore_keep_parent_implementation():
    adapter = SimpleNamespace(prefill=lambda: "prefill", decode_one_token=lambda: "decode")
    original = adapter.prefill
    with subject.bounded_adapter(adapter, time.monotonic() - 1), pytest.raises(TimeoutError):
        adapter.prefill()
    assert adapter.prefill is original


def test_policy_retains_strict_guard_and_disclaims_shared_budget():
    assert subject.POLICY["guard"] == dict(
        tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0
    )
    assert subject.POLICY["independent_quality_N"] == 0
    assert subject.POLICY["no_shared_budget_claim"]
    assert subject.POLICY["source_bits"] == 8


def good_accounting():
    job = dict(
        index=0,
        sample_id="fixture-only",
        task="llava_description",
        expected_protection=[dict(layer=i, mask_sha256=str(i)) for i in range(32)],
    )
    detail = dict(
        source_bits=8,
        protected_original_KV_bitwise=True,
        original_logical_positions_preserved=True,
        no_original_source_alias=True,
    )
    row = dict(
        index=0,
        sample_id=job["sample_id"],
        task=job["task"],
        method=subject.LABEL,
        source_bits=8,
        used_fullkv_fallback=False,
        source_fallback=False,
        inactive_original_source_retired=True,
        raw_source_kv_bytes=1000,
        storage=dict(
            initial_active_kv_storage_bytes=500,
            final_active_kv_storage_bytes=600,
            peak_active_kv_storage_bytes=700,
            history_updates_per_layer=[127] * 32,
        ),
        evidence=dict(
            raw_FullKV_ceiling_bytes=1000,
            construction_records=[dict(layer=i, detail=dict(detail)) for i in range(32)],
            protection_mask_digests=[(i, str(i)) for i in range(32)],
        ),
        phase_wall_seconds={
            key: 0.0
            for key in (
                "media_and_processor",
                "prefill_with_diagnostic_captures",
                "construction",
                "retire_original_sources",
                "decode_including_capture_and_retirement",
            )
        },
    )
    return row, job


def test_accounting_uses_actual_boundaries_not_smaller_cache_flag():
    row, job = good_accounting()
    assert subject.verify_row_accounting(row, job)
    row["storage"]["initial_active_kv_storage_bytes"] = 1000
    row["initial_smaller_than_raw_FullKV"] = True
    with pytest.raises(ValueError, match="actual source bytes"):
        subject.verify_row_accounting(row, job)


def test_incomplete_history_or_constructor_layers_rejected():
    row, job = good_accounting()
    row["storage"]["history_updates_per_layer"][-1] = 126
    with pytest.raises(ValueError, match="complete history"):
        subject.verify_row_accounting(row, job)
    row, job = good_accounting()
    row["evidence"]["construction_records"].pop()
    with pytest.raises(ValueError, match="construction evidence"):
        subject.verify_row_accounting(row, job)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_timing_is_not_accepted_as_completed(value):
    row, job = good_accounting()
    row["phase_wall_seconds"]["construction"] = value
    with pytest.raises(ValueError, match="timing fields"):
        subject.verify_row_accounting(row, job)
