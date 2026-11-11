"""Original-cache bitwise target and complete accounting, CPU-only fixtures."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import lossless_source_control as study
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture(*, heads=8, length=11, dim=4):
    torch.manual_seed(17)
    width = heads * dim
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        v_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        num_key_value_groups=1,
    )
    x = torch.randn(1, length, width).bfloat16()
    angle = torch.randn(1, length, dim // 2).repeat(1, 1, 2)
    phases = angle.cos().bfloat16(), angle.sin().bfloat16()
    positions = torch.arange(length, dtype=torch.int64)
    pair = tuple(study.native_component(x, attention, c, *phases) for c in (0, 1))
    return x, pair, attention, phases, positions


def bitwise(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def test_full_source_retains_every_original_word_and_smaller_actual_storage():
    args = fixture()
    packet = study.construct(*args)
    actual = packet.materialize(args[2])
    assert all(bitwise(a, b) for a, b in zip(args[1], actual, strict=True))
    assert packet.component_bytes < backing_bytes(args[1])
    assert packet.component_bytes == backing_bytes(packet.tensors())
    assert packet.diagnostics["patch_scalars"] == [0, 0]
    assert packet.diagnostics["all_prompt_positions_retained"]
    assert packet.diagnostics["reference"] == "original_prefill_KV_not_lossy_parent_decoder"


def test_sparse_exact_patch_targets_original_kv_not_unmodified_projection():
    x, pair, attention, phases, positions = fixture()
    original = tuple(t.clone() for t in pair)
    original[0][0, 1, 3, 2] += 1
    packet = study.construct(x, original, attention, phases, positions)
    assert packet.diagnostics["patch_scalars"] == [1, 0]
    assert packet.diagnostics["patch_bytes"] == 6
    assert bitwise(packet.materialize(attention)[0], original[0])


@pytest.mark.parametrize("tail", [0, 1, 127])
def test_history_stride_and_sentinels_are_untouched(tail):
    args = fixture()
    packet = study.construct(*args)
    shape = (*packet.shape[:2], packet.shape[-2] + tail, packet.shape[-1])
    targets = tuple(torch.full(shape, -71, dtype=torch.bfloat16) for _ in range(2))
    packet.fill(args[2], targets, direct=True)
    for source, target in zip(args[1], targets, strict=True):
        assert bitwise(source, target[..., : packet.shape[-2], :])
        assert bool((target[..., packet.shape[-2] :, :] == -71).all())


def test_every_source_header_patch_and_map_has_private_backing():
    args = fixture()
    packet = study.construct(*args)
    original_owners = {
        study.owner(t)
        for t in (
            args[0],
            *args[1],
            *args[3],
            args[4],
            args[2].k_proj.weight,
            args[2].v_proj.weight,
        )
    }
    assert not original_owners.intersection(study.owner(t) for t in packet.tensors() if t.numel())
    assert not any(
        field.name in {"original", "original_pair", "reference"}
        for field in dataclasses.fields(packet)
    )


def test_complete_dense_patch_cost_declines_without_raw_fallback():
    x, pair, attention, phases, positions = fixture()
    unrelated = tuple(torch.full_like(t, 11) for t in pair)
    with pytest.raises(study.FeasibilityDeclined) as error:
        study.construct(x, unrelated, attention, phases, positions)
    assert error.value.report["persistent_bytes"] >= error.value.report["raw_kv_bytes"]
    assert error.value.report["patch_density"] > 0.9


def test_cap_cannot_be_exceeded_or_reinterpreted():
    args = fixture()
    packet = study.construct(*args)
    with pytest.raises(study.FeasibilityDeclined):
        study.construct(*args, maximum_persistent_bytes=packet.component_bytes - 1)
    assert (
        study.construct(*args, maximum_persistent_bytes=packet.component_bytes).component_bytes
        == packet.component_bytes
    )
    with pytest.raises(ValueError, match="Positive"):
        study.construct(*args, maximum_persistent_bytes=True)


def test_gqa_and_unordered_positions_are_not_silently_supported():
    x, pair, attention, phases, positions = fixture()
    attention.num_key_value_groups = 4
    with pytest.raises(ValueError, match="GQA"):
        study.construct(x, pair, attention, phases, positions)
    attention.num_key_value_groups = 1
    positions[3] = positions[2]
    with pytest.raises(ValueError, match="monotonic"):
        study.construct(x, pair, attention, phases, positions)


def test_packet_rejects_weight_mutation_and_destination_alias():
    args = fixture()
    packet = study.construct(*args)
    with torch.no_grad():
        args[2].v_proj.weight.add_(1)
    with pytest.raises(ValueError, match="version changed"):
        packet.materialize(args[2])
    fresh = fixture()
    packet = study.construct(*fresh)
    destination = packet.normalized.view(
        1, packet.shape[-2], packet.shape[1], packet.shape[-1]
    ).transpose(1, 2)
    with pytest.raises(ValueError, match="alias"):
        packet.fill_component(fresh[2], 1, destination)


def test_workspace_and_decoded_outputs_remain_observable_and_not_claimed_peak():
    args = fixture()
    packet = study.construct(*args)
    observations = []
    packet.materialize(
        args[2], workspace_observer=lambda ts: observations.append(backing_bytes(ts))
    )
    assert observations and max(observations) > 0
    assert packet.diagnostics["workspace_charged_separately"] is True
    assert packet.diagnostics["whole_request_peak_certified"] is False
    assert packet.diagnostics["latency_claim"] is False


def test_lossless_inputs_remain_unmodified():
    args = fixture()
    tensors = (args[0], *args[1], *args[3], args[4], args[2].k_proj.weight, args[2].v_proj.weight)
    before = tuple(t.clone() for t in tensors)
    study.construct(*args)
    assert all(torch.equal(t, old) for t, old in zip(tensors, before, strict=True))


def test_direct_packet_update_cannot_bypass_byte_cap_or_patch_range():
    packet = study.construct(*fixture())
    with pytest.raises(study.FeasibilityDeclined):
        dataclasses.replace(packet, maximum_persistent_bytes=packet.component_bytes - 1)
    invalid = study.SparseBitPatch(
        torch.tensor([-1], dtype=torch.int32), torch.ones(1, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="in-range"):
        dataclasses.replace(packet, patches=(invalid, packet.patches[1]))


def test_nonzero_key_and_value_patches_receive_head_major_contiguous_storage(monkeypatch):
    from mosaickv.evaluation import shared_source_format

    x, pair, attention, phases, positions = fixture()
    original = tuple(t.clone() for t in pair)
    original[0][0, 2, 3, 1] += 1
    original[1][0, 3, 2, 0] -= 1
    original_apply = shared_source_format.apply_transport_patch
    patched, observations = [], []

    def checked_apply(patch, target, *, prompt_length):
        assert patch.indices.numel() == 1
        assert target.is_contiguous()
        assert target.stride(-2) == target.shape[-1]
        patched.append(target.stride())
        original_apply(patch, target, prompt_length=prompt_length)

    def observe(tensors):
        observations.append(tuple((tuple(t.shape), t.stride(), study.owner(t)) for t in tensors))

    monkeypatch.setattr(shared_source_format, "apply_transport_patch", checked_apply)
    packet = study.construct(x, original, attention, phases, positions, workspace_observer=observe)
    targets = tuple(torch.full((1, 8, 18, 4), -71, dtype=torch.bfloat16) for _ in range(2))
    packet.fill(attention, targets, direct=True, workspace_observer=observe)
    assert len(patched) == 4
    for reference, target in zip(original, targets, strict=True):
        assert bitwise(reference, target[..., :11, :])
        assert bool((target[..., 11:, :] == -71).all())
    assert any(
        len(row) == 2
        and row[0][0] == row[1][0] == packet.shape
        and row[0][1][-2] == 32
        and row[1][1][-2] == 4
        and row[0][2] != row[1][2]
        for row in observations
    ), "Both sequence-major native output and head-major patch copy must be charged"
