"""CPU-only controls for native reading of the SAME intentionally lossy packet."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.native_quantized_source_control import (
    NativeQuantizedSourceControl,
    native_quantized_fill,
    workspace_upper_bound,
)
from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.shared_source_format import SharedSourceFormat, construct


def example():
    torch.manual_seed(20261111)
    generator = torch.Generator().manual_seed(20261111)
    heads, length, dim = 2, 17, 128
    normalized = torch.randn((1, length, heads * dim), generator=generator).bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16().eval(),
        v_proj=torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16().eval(),
        num_key_value_groups=1,
    )
    angles = torch.randn((1, length, dim // 2), generator=generator).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    with torch.inference_mode():
        key = attention.k_proj(normalized).reshape(1, length, heads, dim).transpose(1, 2)
        value = attention.v_proj(normalized).reshape(1, length, heads, dim).transpose(1, 2)
        key = (
            key * phases[0][:, None]
            + torch.cat((-key[..., 64:], key[..., :64]), -1) * phases[1][:, None]
        )
    original = key, value
    identity = (
        torch.arange(length, dtype=torch.int32),
        torch.arange(length, dtype=torch.int16)[None].expand(heads, length),
    )
    packet = construct(
        normalized, original, ((0, 3, 8, 16), (1, 5, 9, 16)), (0, 16), phases, identity
    )
    return packet, attention, normalized, original


@torch.inference_mode()
def expected_native(packet, attention):
    source = unpack_cpu(packet.source).squeeze(1)
    _, heads, length, dim = packet.shape
    key = attention.k_proj(source).reshape(1, length, heads, dim).transpose(1, 2)
    value = attention.v_proj(source).reshape(1, length, heads, dim).transpose(1, 2)
    key = (
        key * packet.cos[:, None]
        + torch.cat((-key[..., dim // 2 :], key[..., : dim // 2]), -1) * packet.sin[:, None]
    )
    result = key.contiguous(), value.contiguous()
    ids = packet.row_positions.long()
    for target, body in zip(result, packet.exact, strict=True):
        target[0, ids // length, ids % length] = decoded(body)[0, 0]
    return result


def test_native_control_is_bitwise_quantized_reference_and_preserves_packet_storage():
    packet, attention, normalized, original = example()
    # The diagnostic must NOT apply fused-reader transport corrections. Keep
    # those real buffers in the packet and its accounting nevertheless.
    patches = tuple(
        SparseBitPatch(torch.tensor([0], dtype=torch.int32), torch.tensor([71.0]).bfloat16())
        for _ in range(2)
    )
    packet = replace(packet, transport_patches=patches)
    inputs = (
        *packet.tensors(),
        normalized,
        *original,
        attention.k_proj.weight,
        attention.v_proj.weight,
    )
    snapshots = tuple(t.detach().clone() for t in inputs)
    owners = tuple(t.untyped_storage()._cdata for t in packet.tensors())
    expected = expected_native(packet, attention)
    before_bytes = packet.component_bytes
    with NativeQuantizedSourceControl((packet,), (attention,)) as control:
        actual = packet.materialize(attention)
        for want, got in zip(expected, actual, strict=True):
            assert torch.equal(want.view(torch.int16), got.view(torch.int16))
        assert control.report["persistent_tensor_bytes_added"] == 0
        assert control.report["packet_bytes"] == before_bytes
        assert control.report["native_fill_calls"] == 1
    assert packet.component_bytes == before_bytes
    assert tuple(t.untyped_storage()._cdata for t in packet.tensors()) == owners
    assert all(torch.equal(before, now) for before, now in zip(snapshots, inputs, strict=True))
    # Explicitly do not claim the three-bit source recreates original FullKV.
    assert any(not torch.equal(a, b) for a, b in zip(actual, original, strict=True))


def test_strided_interior_leases_keep_surroundings_and_report_bounded_transient_owners():
    packet, attention, _, _ = example()
    length = packet.shape[2]
    backing = tuple(torch.full((1, 2, length + 8, 128), -71).bfloat16() for _ in range(2))
    targets = tuple(t[..., 3 : length + 5, :] for t in backing)
    assert all(not t.is_contiguous() for t in targets)
    observed = []

    def observe(tensors):
        observed.append(backing_bytes(tensors))

    with NativeQuantizedSourceControl((packet,), (attention,)):
        packet.fill(attention, targets, direct=True, workspace_observer=observe)
    expected = expected_native(packet, attention)
    for owner_tensor, got, want in zip(backing, targets, expected, strict=True):
        assert torch.equal(got[..., :length, :].view(torch.int16), want.view(torch.int16))
        assert (owner_tensor[..., :3, :] == -71).all()
        assert (owner_tensor[..., length + 3 :, :] == -71).all()
    assert observed and 0 < max(observed) <= workspace_upper_bound(packet)


def test_foreign_request_and_attention_fail_closed_and_methods_restore_on_error():
    packet, attention, _, _ = example()
    foreign, foreign_attention, _, _ = example()
    original_methods = SharedSourceFormat.fill, SharedSourceFormat.materialize
    with (
        pytest.raises(ValueError, match="Foreign packet"),
        NativeQuantizedSourceControl((packet,), (attention,)),
    ):
        foreign.materialize(foreign_attention)
    assert (SharedSourceFormat.fill, SharedSourceFormat.materialize) == original_methods
    with NativeQuantizedSourceControl((packet,), (attention,)):
        with pytest.raises(ValueError, match="Foreign packet"):
            packet.materialize(foreign_attention)
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            NativeQuantizedSourceControl((packet,), (attention,)),
        ):
            pass
    assert (SharedSourceFormat.fill, SharedSourceFormat.materialize) == original_methods


def test_observer_failure_restores_methods_and_does_not_mutate_source():
    packet, attention, _, _ = example()
    source_before = tuple(t.clone() for t in packet.tensors())
    methods = SharedSourceFormat.fill, SharedSourceFormat.materialize

    def fail(_tensors):
        raise RuntimeError("observer failed")

    with (
        pytest.raises(RuntimeError, match="observer failed"),
        NativeQuantizedSourceControl((packet,), (attention,)),
    ):
        targets = tuple(torch.empty(packet.shape, dtype=torch.bfloat16) for _ in range(2))
        packet.fill(attention, targets, direct=True, workspace_observer=fail)
    assert (SharedSourceFormat.fill, SharedSourceFormat.materialize) == methods
    assert all(
        torch.equal(before, after)
        for before, after in zip(source_before, packet.tensors(), strict=True)
    )


def test_rejects_changed_weights_training_gqa_and_nonoriginal_phases():
    packet, attention, _, _ = example()
    control = NativeQuantizedSourceControl((packet,), (attention,))
    with torch.no_grad():
        attention.k_proj.weight.add_(1)
    with pytest.raises(RuntimeError, match="signature changed"), control:
        pass
    attention.k_proj.train()
    with pytest.raises(ValueError, match="Frozen eval"):
        NativeQuantizedSourceControl((packet,), (attention,))
    attention.k_proj.eval()
    attention.num_key_value_groups = 2
    with pytest.raises(ValueError, match="full-MHA"):
        NativeQuantizedSourceControl((packet,), (attention,))
    attention.num_key_value_groups = 1
    phases = packet.cos.clone()
    phases[..., 64:] += 1
    with pytest.raises(ValueError, match="original RoPE"):
        NativeQuantizedSourceControl((replace(packet, cos=phases),), (attention,))


def test_standalone_rejects_destination_alias_and_invalid_precision():
    packet, attention, _, _ = example()
    destination = torch.empty(packet.shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must not alias"):
        native_quantized_fill(packet, attention, (destination, destination), direct=True)
    with pytest.raises(ValueError, match="complete BF16"):
        native_quantized_fill(
            packet, attention, tuple(torch.empty(packet.shape) for _ in range(2)), direct=True
        )


def test_scope_rejects_cross_thread_reads_and_restores_original_reader():
    packet, attention, _, _ = example()
    original_reader = SharedSourceFormat.fill
    with (
        NativeQuantizedSourceControl((packet,), (attention,)),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        result = executor.submit(packet.materialize, attention)
        with pytest.raises(RuntimeError, match="thread boundaries"):
            result.result(timeout=5)
    assert SharedSourceFormat.fill is original_reader
    for want, actual in zip(
        expected_native(packet, attention), packet.materialize(attention), strict=True
    ):
        assert torch.equal(want, actual)


def test_native_operation_order_is_not_silently_changed_by_autocast():
    packet, attention, _, _ = example()
    targets = tuple(torch.empty(packet.shape, dtype=torch.bfloat16) for _ in range(2))
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="autocast disabled"),
    ):
        native_quantized_fill(packet, attention, targets, direct=True)
