"""Original full-width source semantics and one-component workspace controls."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import bounded_component_source as study
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import native_component


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(25)
    heads, length, dim = 4, 7, 4
    width = heads * dim
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        v_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        num_key_value_groups=1,
    )
    source = torch.randn(1, length, width).bfloat16()
    angle = torch.randn(1, length, dim // 2).repeat(1, 1, 2)
    phases = angle.cos().bfloat16(), angle.sin().bfloat16()
    pair = tuple(native_component(source, attention, c, *phases) for c in (0, 1))
    return source, pair, attention, phases


def bits(first, second):
    return torch.equal(first.contiguous().view(torch.int16), second.contiguous().view(torch.int16))


def test_native_full_width_zero_patch_and_single_buffer_reuse(monkeypatch):
    args = fixture()
    mm = torch.mm
    shapes = []

    def native_mm(left, right, *, out):
        shapes.append((tuple(left.shape), tuple(right.shape), tuple(out.shape)))
        return mm(left, right, out=out)

    monkeypatch.setattr(study.torch, "mm", native_mm)
    packet = study.construct(*args)
    assert packet.diagnostics["patch_scalars"] == [0, 0]
    buffer = torch.full((1, 17, 4, 4), -71, dtype=torch.bfloat16)
    target = buffer.transpose(1, 2)
    scratch = torch.empty((20, 4), dtype=torch.bfloat16)
    for component in (0, 1):
        packet.fill_component(args[2], component, target, rope_scratch=scratch)
        assert bits(target[..., :7, :], args[1][component])
        assert bool((target[..., 7:, :] == -71).all())
    assert shapes == [((7, 16), (16, 16), (7, 16))] * 4
    assert packet.component_bytes == backing_bytes(packet.tensors())


def test_sequence_major_sparse_patches_restore_original_words():
    source, pair, attention, phases = fixture()
    pair = tuple(t.clone() for t in pair)
    pair[0][0, 2, 1, 3] += 2
    pair[1][0, 1, 6, 0] -= 3
    packet = study.construct(source, pair, attention, phases)
    assert packet.diagnostics["patch_scalars"] == [1, 1]
    assert packet.diagnostics["patch_bytes"] == 12
    assert int(packet.patches[0].indices[0]) == (1 * 4 + 2) * 4 + 3
    target = torch.empty((1, 12, 4, 4), dtype=torch.bfloat16).transpose(1, 2)
    for component in (0, 1):
        packet.fill_component(attention, component, target)
        assert bits(target[..., :7, :], pair[component])


def test_shared_phases_capture_transfer_and_immutable_originals():
    args = fixture()
    originals = (args[0], *args[1], *args[3], args[2].k_proj.weight, args[2].v_proj.weight)
    before = tuple(t.clone() for t in originals)
    one = study.construct(*args, take_source_ownership=True)
    two = study.construct(*args)
    assert one.normalized is args[0]
    assert study.owner(two.normalized) != study.owner(args[0])
    assert one.cos is two.cos is args[3][0]
    assert backing_bytes((*one.tensors(), *two.tensors())) == (
        one.component_bytes + two.component_bytes - backing_bytes(args[3])
    )
    assert all(bits(a, b) for a, b in zip(originals, before, strict=True))


def test_layout_alias_and_frozen_projection_checks():
    args = fixture()
    packet = study.construct(*args)
    head_major = torch.empty(packet.shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Projection-layout"):
        packet.fill_component(args[2], 1, head_major)
    source_alias = packet.normalized.view(1, 7, 4, 4).transpose(1, 2)
    with pytest.raises(ValueError, match="alias"):
        packet.fill_component(args[2], 1, source_alias)
    target = torch.empty((1, 9, 4, 4), dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(ValueError, match="disjoint"):
        packet.fill_component(
            args[2], 0, target, rope_scratch=target.transpose(1, 2).reshape(-1, 4)
        )
    with torch.no_grad():
        args[2].v_proj.weight.add_(1)
    with pytest.raises(ValueError, match="version changed"):
        packet.fill_component(args[2], 1, target)


def test_cap_and_nonfinite_source_fail_without_raw_fallback():
    args = fixture()
    packet = study.construct(*args)
    with pytest.raises(study.FeasibilityDeclined):
        study.construct(*args, maximum_persistent_bytes=packet.component_bytes - 1)
    assert (
        study.construct(*args, maximum_persistent_bytes=packet.component_bytes).component_bytes
        == packet.component_bytes
    )
    args[0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        study.construct(*args)


def test_observer_counts_only_one_component_and_one_head_scratch():
    args = fixture()
    packet = study.construct(*args)
    target = torch.empty((1, 14, 4, 4), dtype=torch.bfloat16).transpose(1, 2)
    scratch = torch.empty((20, 4), dtype=torch.bfloat16)
    records = []
    packet.fill_component(
        args[2], 0, target, rope_scratch=scratch, workspace_observer=records.append
    )
    assert records == [(target, scratch)]
    records.clear()
    packet.fill_component(args[2], 1, target, workspace_observer=records.append)
    assert records == [(target,)]
    assert packet.diagnostics["simultaneous_full_layer_KV_pair"] is False
    assert packet.diagnostics["whole_request_peak_certified"] is False
