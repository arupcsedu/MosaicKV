"""Separate lossless-source lane; real CPU packet and mocked request boundaries."""

from __future__ import annotations

import copy
import inspect
import time
from types import SimpleNamespace

import evicastkv_lossless_source_control as study
import pytest
import torch

from mosaickv.evaluation.lossless_source_control import native_component


class Attention(torch.nn.Module):
    def __init__(self, heads=8, dim=4):
        super().__init__()
        self.k_proj = torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16()
        self.v_proj = torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16()
        self.num_key_value_groups = 1
        self.eval()


def fixture():
    torch.manual_seed(19)
    modules = [SimpleNamespace(self_attn=Attention()) for _ in range(2)]
    records = {i: torch.randn(1, 11, 32).bfloat16() for i in range(2)}
    phases = (torch.ones(1, 11, 4).bfloat16(), torch.zeros(1, 11, 4).bfloat16())
    pairs = tuple(
        tuple(native_component(records[i], layer.self_attn, c, *phases) for c in (0, 1))
        for i, layer in enumerate(modules)
    )
    adapter = SimpleNamespace(
        _language_layers=lambda: modules, _legacy_layers=lambda value: (value, None)
    )
    prefill = SimpleNamespace(state=SimpleNamespace(past_key_values=pairs))
    return adapter, prefill, SimpleNamespace(records=records), SimpleNamespace(phases=phases)


def test_direct_original_source_constructor_has_no_three_bit_or_reference_input():
    adapter, prefill, normalized, phases = fixture()
    calls = []
    from mosaickv.evaluation.lossless_source_control import construct

    def record(source, pair, attention, rotary, positions, **kwargs):
        calls.append((source, pair, positions))
        return construct(source, pair, attention, rotary, positions, **kwargs)

    packets, records, evidence = study.lossless_formats(
        adapter, object(), prefill, object(), normalized, phases, constructor=record
    )
    assert len(calls) == len(packets) == len(records) == 2
    assert all(r["all_original_KV_bitwise"] for r in records)
    assert all(r["observed_ATen_operations"] > 0 for r in records)
    assert evidence["persistent_format_bytes"] < evidence["raw_FullKV_ceiling_bytes"]
    assert (
        evidence["construction_observed_tensor_backing_peak_bytes"]
        >= evidence["raw_FullKV_ceiling_bytes"]
    )
    for i, packet in enumerate(packets):
        assert calls[i][0] is normalized.records[i]
        assert calls[i][1] is prefill.state.past_key_values[i]
        assert torch.equal(calls[i][2], torch.arange(11))
        actual = packet.materialize(adapter._language_layers()[i].self_attn)
        assert all(
            study.bitwise(a, b)
            for a, b in zip(actual, prefill.state.past_key_values[i], strict=True)
        )
    assert not {"prefix", "labels", "answers", "reference_logits"}.intersection(
        inspect.signature(study.lossless_formats).parameters
    )


def test_missing_source_or_duplicate_attention_and_deadline_fail_before_constructor():
    adapter, prefill, normalized, phases = fixture()
    bad = SimpleNamespace(records={0: normalized.records[0]})
    with pytest.raises(ValueError, match="every layer"):
        study.lossless_formats(adapter, None, prefill, None, bad, phases)
    adapter._language_layers()[1].self_attn = adapter._language_layers()[0].self_attn
    with pytest.raises(ValueError, match="unique"):
        study.lossless_formats(adapter, None, prefill, None, normalized, phases)
    adapter, prefill, normalized, phases = fixture()
    with pytest.raises(TimeoutError):
        study.lossless_formats(
            adapter,
            None,
            prefill,
            None,
            normalized,
            phases,
            deadline=time.monotonic() - 1,
            constructor=lambda *a, **k: pytest.fail("constructor ran"),
        )


def test_every_original_word_is_independently_checked():
    adapter, prefill, normalized, phases = fixture()
    from mosaickv.evaluation.lossless_source_control import construct

    def broken(*args, **kwargs):
        construct(*args, **kwargs)
        return SimpleNamespace(
            materialize=lambda *a, **k: tuple(torch.ones_like(t) * 91 for t in args[1])
        )

    with pytest.raises(ValueError, match="changed a bit"):
        study.lossless_formats(adapter, None, prefill, None, normalized, phases, constructor=broken)


def test_generic_shared_workspace_accepts_exact_source_format_without_dispatch_fallback():
    from mosaickv.evaluation.storage_runs import cache_from_formats

    adapter, prefill, normalized, phases = fixture()
    formats, _, _ = study.lossless_formats(adapter, None, prefill, None, normalized, phases)
    cache = cache_from_formats(
        adapter,
        formats,
        updates=128,
        audit=True,
        source_history=True,
        source_history_constructor="direct",
        shared_workspace=True,
        workspace_graph=False,
    )
    assert all(
        layer.representation is packet for layer, packet in zip(cache.layers, formats, strict=True)
    )
    assert cache.initial_bytes >= sum(p.component_bytes for p in formats)
    assert not cache.graphs_enabled


@pytest.mark.parametrize("bad", ["fallback", "unretired", "no_savings", "wrong_raw"])
def test_request_rejects_storage_or_retirement_failure_and_restores_scope(monkeypatch, bad):
    import evicastkv_review_source_controls as source

    from mosaickv.evaluation import storage_runs

    original_builder = source._formats
    prefix = tuple(range(128))
    monkeypatch.setattr(
        study, "lossless_formats", lambda *a, **k: ((), [], dict(raw_FullKV_ceiling_bytes=100))
    )

    def decode(*args, **kwargs):
        assert kwargs["forced"] is prefix
        assert kwargs["audit"] is True
        assert kwargs["initial_physical_byte_cap"] == 100
        return torch.zeros(128, 2), dict(
            storage=dict(initial_active_kv_storage_bytes=100 if bad == "no_savings" else 90),
            used_fullkv_fallback=bad == "fallback",
            inactive_original_source_retired=bad != "unretired",
            raw_source_kv_bytes=99 if bad == "wrong_raw" else 100,
        )

    monkeypatch.setattr(storage_runs, "run", decode)

    def request(*args):
        source._formats(None, None, None, None, None, None, "old_lifecycle_only")
        return storage_runs.run(None, audit=False)

    monkeypatch.setattr(source, "_request", request)
    with pytest.raises(ValueError, match="retirement/storage/fallback"):
        study.request(None, None, None, prefix=prefix)
    assert source._formats is original_builder
    assert storage_runs.run is decode


def valid_row():
    job = dict(index=0, sample_id="consumed", task="mmvet_prefixkv")
    row = dict(
        **job,
        method=study.LABEL,
        used_fullkv_fallback=False,
        source_fallback=False,
        inactive_original_source_retired=True,
        raw_source_kv_bytes=100,
        output_tokens=128,
        storage=dict(
            initial_active_kv_storage_bytes=60,
            final_active_kv_storage_bytes=80,
            peak_active_kv_storage_bytes=90,
            history_updates_per_layer=[127] * 32,
        ),
        evidence=dict(
            raw_FullKV_ceiling_bytes=100,
            construction_records=[
                dict(
                    layer=i,
                    all_original_KV_bitwise=True,
                    no_original_or_weight_alias=True,
                    persistent_bytes=60,
                    detail=dict(
                        raw_kv_bytes=100, reference="original_prefill_KV_not_lossy_parent_decoder"
                    ),
                )
                for i in range(32)
            ],
        ),
        phase_wall_seconds={
            name: 0.1
            for name in [
                "media_and_processor",
                "prefill_with_diagnostic_captures",
                "construction",
                "retire_original_sources",
                "decode_including_capture_and_retirement",
            ]
        },
    )
    return row, job


def test_row_accounting_checks_actual_bytes_complete_history_and_all_layers():
    row, job = valid_row()
    assert study.verify_row(row, job)
    for mutate in [
        lambda r: r.update(used_fullkv_fallback=True),
        lambda r: r["storage"].update(initial_active_kv_storage_bytes=100),
        lambda r: r["evidence"]["construction_records"].pop(),
        lambda r: r["evidence"]["construction_records"][0].update(all_original_KV_bitwise=False),
        lambda r: r["storage"].update(history_updates_per_layer=[126] * 32),
        lambda r: r["phase_wall_seconds"].update(construction=float("nan")),
    ]:
        bad = copy.deepcopy(row)
        mutate(bad)
        with pytest.raises(ValueError):
            study.verify_row(bad, job)


def test_operator_precedes_checkpoint_and_failure_stops_loading():
    calls = []
    assert study.operator_before_model(
        lambda: calls.append("operator") or {"passed": True},
        lambda: calls.append("checkpoint") or object(),
    )[0]["passed"]
    assert calls == ["operator", "checkpoint"]
    with pytest.raises(ValueError):
        study.operator_before_model(
            lambda: {"passed": False}, lambda: pytest.fail("weights loaded")
        )


def test_policy_is_separate_from_eight_bit_lane_and_has_no_promotion():
    import evicastkv_source_precision_control as eight

    assert study.POLICY is not eight.POLICY
    assert study.POLICY["guard"] == eight.POLICY["guard"]
    assert study.POLICY["source_precision"] == "original_BF16_no_quantization"
    assert study.POLICY["independent_quality_N"] == 0
    assert study.POLICY["no_shared_byte_or_latency_claim"]
