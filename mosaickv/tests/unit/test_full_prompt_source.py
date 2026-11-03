from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import full_prompt_source as native
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.storage_runs import cache_from_formats


def example(length=33):
    from mosaickv.evaluation.shared_source_format import construct

    torch.manual_seed(20261103)
    h, n, d = 2, length, 128
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    # ExactLayer needs the same module weight inventory as a real attention.
    attention.parameters = lambda: (*attention.k_proj.parameters(), *attention.v_proj.parameters())
    x = torch.randn(1, n, h * d).bfloat16()
    angles = torch.randn(1, n, 64).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    pair = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    packet = construct(
        x,
        pair,
        (tuple(range(n)),) * h,
        (0, n - 1),
        phases,
        (torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None].expand(h, n)),
    )
    return attention, packet


@pytest.mark.parametrize("policy", ("kv", "v_only"))
def test_full_prompt_reuses_one_source_and_preserves_all_parent_bits(policy, monkeypatch):
    attention, packet = example()
    candidate = native.fit(packet, attention, native.POLICIES[policy])
    before = backing_bytes(candidate.tensors())
    original = packet.materialize(attention)
    calls = []
    decode = native.decode_into

    def counted(packet, workspace):
        calls.append(workspace.data_ptr())
        decode(packet, workspace)

    monkeypatch.setattr(native, "decode_into", counted)
    workspace = native.new_workspace(candidate, torch.device("cpu"))
    assert workspace.shape == (1, 33, 256)
    targets = tuple(torch.full((1, 2, 40, 128), -71).bfloat16() for _ in range(2))
    for component in (0, 1):
        candidate.fill_component(attention, component, targets[component], workspace)
        assert torch.equal(
            targets[component][..., :33, :].view(torch.int16), original[component].view(torch.int16)
        )
        assert (targets[component][..., 33:, :] == -71).all()
    assert calls == [workspace.data_ptr()]
    assert backing_bytes(candidate.tensors()) == before
    assert candidate.component_bytes >= packet.component_bytes


@pytest.mark.parametrize("policy", ("kv", "v_only"))
def test_layer_workspace_is_charged_released_and_request_private(policy):
    attention, packet = example()
    candidate = native.fit(packet, attention, native.POLICIES[policy])
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])

    def cache():
        return cache_from_formats(
            adapter,
            (candidate,),
            updates=16,
            audit=False,
            sealed=True,
            sequential_lease=True,
            source_history=True,
            sequential_source_history=True,
            consolidated_history=True,
            source_history_constructor="direct_batched_checks",
        )

    a, b = cache(), cache()
    pair = tuple(torch.randn(1, 2, 1, 128).bfloat16() for _ in range(2))
    for c in (a, b):
        layer = c.layers[0]
        layer.source_pending = True
        layer.update(*pair)
        if policy == "kv":
            assert layer.source_workspace is not None
            assert layer.lease_bytes == backing_bytes((*layer.leased, layer.source_workspace))
            assert c.storage_record()["final_active_kv_storage_bytes"] == backing_bytes(
                (*c.shared_tensors, *layer.owned_tensors())
            )
        else:
            assert getattr(layer, "source_workspace", None) is None
    if policy == "kv":
        assert a.layers[0].source_workspace.data_ptr() != b.layers[0].source_workspace.data_ptr()
    for c in (a, b):
        layer = c.layers[0]
        layer.materialize_component(1)
        assert layer.source_workspace is None
        assert layer.native_full_source_decodes == 1
        assert layer.native_full_component_fills == len(native.POLICIES[policy])
        assert layer.lease_bytes == backing_bytes(layer.leased)
        layer.release()
        record = c.storage_record()
        assert record["outstanding_layer_leases"] == 0
        assert record["final_active_kv_storage_bytes"] == backing_bytes(
            (*c.shared_tensors, *layer.owned_tensors())
        )


def test_declined_cap_never_silently_falls_back_and_order_is_deterministic(monkeypatch):
    attention, packet = example()
    candidate = native.fit(packet, attention, (0, 1))
    monkeypatch.setattr(native, "fit", lambda *a: candidate)
    first, a = native.convert((packet,) * 8, (attention,) * 8, "kv")
    second, b = native.convert((packet,) * 8, (attention,) * 8, "kv")
    assert a == b and a["converted_layers"] == list(range(8))
    assert all(x is candidate for x in (*first, *second))
    monkeypatch.setattr(native, "MAXIMUM_REPAIR_BYTES", -1)
    with pytest.raises(native.FeasibilityDeclined) as error:
        native.convert((packet,) * 8, (attention,) * 8, "kv")
    assert not error.value.report["enabled"]


def test_invalid_policy_layout_and_missing_workspace_fail_explicitly():
    attention, packet = example()
    for policy in (None, True, "native", "guess"):
        with pytest.raises(ValueError):
            native.validate(policy)
    candidate = native.fit(packet, attention, (0, 1))
    target = torch.empty(packet.shape, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="owned"):
        candidate.fill_component(attention, 0, target, None)
    with pytest.raises(ValueError, match="original fused"):
        native.decode_into(
            replace(packet, input_overlay=object()), native.new_workspace(candidate, "cpu")
        )
    with pytest.raises(ValueError, match="native BF16"):
        native.project_into(
            packet, attention, 0, target.float(), native.new_workspace(candidate, "cpu")
        )


def test_common_option_is_an_explicit_noop_for_ineligible_baselines():
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.lossless_bf16 import PackedBF16

    value = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
    packet = ExactFormat(
        "kv", tuple(value.shape), pair=(PackedBF16(tuple(value.shape), True, (value,)),) * 2
    )
    result, report = native.convert((packet,), (object(),), "kv")
    assert result[0] is packet and not report["enabled"] and report["eligible_layers"] == 0


def test_repairs_are_lossless_and_stride_dependent_drift_is_rejected(monkeypatch):
    attention, packet = example()
    original = native.project_into

    def damaged(packet, attention, component, destination, workspace):
        original(packet, attention, component, destination, workspace)
        destination.view(torch.int16)[0, 0, 0, 0] ^= 1

    monkeypatch.setattr(native, "project_into", damaged)
    candidate = native.fit(packet, attention, (0, 1))
    assert all(p.indices.numel() > 0 for p in candidate.repairs)
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(
            packet.materialize(attention), candidate.materialize(attention), strict=True
        )
    )

    def stride_damaged(packet, attention, component, destination, workspace):
        original(packet, attention, component, destination, workspace)
        if destination.shape[-2] > packet.shape[-2]:
            destination.view(torch.int16)[0, 0, 0, 0] ^= 1

    monkeypatch.setattr(native, "project_into", stride_damaged)
    with pytest.raises(RuntimeError, match="decoder bit"):
        native.fit(packet, attention, (0, 1))


def test_code_replay_preserves_native_wrapper_repairs_and_accounts_workspace(monkeypatch):
    from test_layer_code_delta import cpu_decode, cpu_encode

    from mosaickv.evaluation import layer_code_delta

    attention, packet = example(64)  # complete 128-byte codec groups, unchanged codec guard
    candidate = native.fit(packet, attention, (0, 1))
    packets = tuple(
        replace(
            candidate,
            base=replace(packet, source=replace(packet.source, codes=packet.source.codes.clone())),
        )
        for _ in range(8)
    )
    monkeypatch.setattr(layer_code_delta, "encode", cpu_encode)
    monkeypatch.setattr(layer_code_delta, "decode", cpu_decode)
    encoded, proof = layer_code_delta.convert(packets)
    assert proof["enabled"]
    bound = layer_code_delta.bind(encoded)
    assert native.source_format(bound[0]).repairs is candidate.repairs
    target = torch.full((1, 2, 66, 128), -71).bfloat16()
    workspace = native.new_workspace(bound[0], "cpu")
    bound[0].fill_component(attention, 0, target, source_workspace=workspace)
    assert torch.equal(
        target[..., :64, :].view(torch.int16), packet.materialize(attention)[0].view(torch.int16)
    )
    assert (target[..., 64:, :] == -71).all()


def test_release_clears_workspace_after_projection_failure(monkeypatch):
    attention, packet = example()
    candidate = native.fit(packet, attention, (0, 1))
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    cache = cache_from_formats(
        adapter,
        (candidate,),
        updates=2,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        consolidated_history=True,
        source_history_constructor="direct_batched_checks",
    )

    def failure(*args):
        raise RuntimeError("synthetic projection failure")

    monkeypatch.setattr(native, "project_into", failure)
    layer = cache.layers[0]
    layer.source_pending = True
    value = torch.zeros(1, 2, 1, 128).bfloat16()
    with pytest.raises(RuntimeError, match="synthetic projection"):
        layer.update(value, value)
    assert layer.source_workspace is not None
    layer.release()
    assert layer.source_workspace is None and not layer.leased and layer.lease_bytes == 0
    assert cache.storage_record()["outstanding_layer_leases"] == 0


def test_constructor_has_no_benchmark_label_or_optimizer_interface():
    import inspect

    assert tuple(inspect.signature(native.convert).parameters) == (
        "packets",
        "attentions",
        "policy",
    )
    assert tuple(inspect.signature(native.fit).parameters) == ("base", "attention", "components")
    body = inspect.getsource(native)
    assert ".backward(" not in body and "torch.optim" not in body
