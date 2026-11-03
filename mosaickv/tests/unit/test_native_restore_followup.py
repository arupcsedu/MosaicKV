from dataclasses import replace
from types import SimpleNamespace

import evicastkv_native_restore_followup as driver
import pytest
import torch
from test_full_prompt_source import example

from mosaickv.evaluation import full_prompt_source as native
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def test_merged_restore_preserves_bitwise_values_history_and_storage():
    attention, packet = example(length=64)
    candidate = native.fit(packet, attention, (0, 1))
    fused = native.enable_fused_restore(candidate)
    expected = candidate.materialize(attention)
    for padding in (0, 1, 64, 128):
        workspace = native.new_workspace(fused, "cpu")
        targets = tuple(torch.full((1, 2, 64 + padding, 128), -71).bfloat16() for _ in (0, 1))
        for c in (0, 1):
            fused.fill_component(attention, c, targets[c], workspace)
            assert torch.equal(
                expected[c].view(torch.int16), targets[c][..., :64, :].view(torch.int16)
            )
            assert (targets[c][..., 64:, :] == -71).all()
    assert fused.component_bytes == backing_bytes(fused.tensors())
    assert fused.fused_restore and not candidate.fused_restore
    assert fused.component_bytes >= packet.component_bytes
    with pytest.raises(ValueError, match="unfused"):
        native.enable_fused_restore(fused)


def test_fused_cache_workspace_lifetime_activity_and_private_state():
    from mosaickv.evaluation.storage_runs import cache_from_formats

    attention, packet = example(length=64)
    fused = native.enable_fused_restore(native.fit(packet, attention, (0, 1)))
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    caches = [
        cache_from_formats(
            adapter,
            (fused,),
            updates=16,
            audit=False,
            sealed=True,
            sequential_lease=True,
            source_history=True,
            sequential_source_history=True,
            consolidated_history=True,
            source_history_constructor="direct_batched_checks",
        )
        for _ in (0, 1)
    ]
    workspaces = []
    for cache in caches:
        layer = cache.layers[0]
        layer.source_pending = True
        layer.update(*(torch.zeros(1, 2, 1, 128).bfloat16() for _ in (0, 1)))
        workspaces.append(layer.source_workspace)
        assert layer.lease_bytes == backing_bytes((*layer.leased, layer.source_workspace))
        assert layer.native_fused_restore_fills == 1
        layer.materialize_component(1)
        assert layer.source_workspace is None and layer.native_fused_restore_fills == 2
        layer.release()
        assert cache.storage_record()["outstanding_layer_leases"] == 0
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()


@pytest.mark.parametrize("bad_indices", ([0], [-1], [2 * 64 * 128], [128, 128]))
def test_fused_restore_rejects_protected_out_of_range_or_duplicate_writes(bad_indices):
    attention, packet = example(length=64)
    candidate = native.fit(packet, attention, (0, 1))
    bad = SparseBitPatch(
        torch.tensor(bad_indices, dtype=torch.int32),
        torch.zeros(len(bad_indices), dtype=torch.bfloat16),
    )
    with pytest.raises(RuntimeError, match="overlapping"):
        native.enable_fused_restore(replace(candidate, repairs=(bad, candidate.repairs[1])))


def test_old_and_new_patches_cannot_overlap_and_nonempty_old_backing_is_counted():
    attention, packet = example(length=64)
    candidate = native.fit(packet, attention, (0, 1))
    old = SparseBitPatch(torch.tensor([1024], dtype=torch.int32), torch.tensor([9.0]).bfloat16())
    zero = SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
    candidate = replace(
        candidate,
        base=replace(packet, transport_patches=(old, zero), protected_patches_disjoint=True),
        repairs=(old, zero),
    )
    with pytest.raises(RuntimeError, match="overlapping"):
        native.enable_fused_restore(candidate)
    fused = native.enable_fused_restore(replace(candidate, repairs=(zero, zero)))
    assert torch.equal(fused.repairs[0].indices, old.indices)
    assert fused.repairs[0].indices.data_ptr() != old.indices.data_ptr()
    assert fused.component_bytes == candidate.component_bytes + 6


def test_unproven_nonempty_original_patch_disjointness_fails_closed():
    attention, packet = example(length=64)
    candidate = native.fit(packet, attention, (0, 1))
    candidate = replace(candidate, base=replace(packet, transport_patches=candidate.repairs))
    with pytest.raises(ValueError, match="constructor-proven"):
        native.enable_fused_restore(candidate)


def rows():
    return [
        dict(
            cases=[
                dict(
                    layer=layer,
                    padding=padding,
                    aggregate={
                        m: dict(mean_cuda_ms=0.8 if m == "fused" else 1.0)
                        for m in ("fused", "bmm", "original")
                    },
                    correctness={
                        "fused": dict(
                            bitwise=True, history_sentinel=True, extra_output_workspace_bytes=0
                        )
                    },
                )
                for layer in (0, 16, 31)
                for padding in (0, 64, 127)
            ]
        )
        for _ in (0, 1)
    ]


def test_operator_gate_requires_both_inputs_every_geometry_and_zero_extra_buffer():
    good = rows()
    assert driver.eligible(good)
    assert not driver.eligible(good[:1])
    good[1]["cases"][0]["aggregate"]["fused"]["mean_cuda_ms"] = 0.91
    assert not driver.eligible(good)
    good = rows()
    good[0]["cases"][1]["correctness"]["fused"]["extra_output_workspace_bytes"] = 1
    assert not driver.eligible(good)
