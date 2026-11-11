"""CPU-only original-word, bounded-reader and functional-allocation controls."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import bounded_source_format as study
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import native_component


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(23)
    heads, length, dim = 4, 7, 4
    width = heads * dim
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        v_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        num_key_value_groups=1,
    )
    source = torch.randn(1, length, width).bfloat16()
    angles = torch.randn(1, length, dim // 2).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    original = tuple(native_component(source, attention, c, *phases) for c in (0, 1))
    return source, original, attention, phases


def bits(first, second):
    return torch.equal(first.contiguous().view(torch.int16), second.contiguous().view(torch.int16))


def test_original_word_patches_not_approximate_projection():
    source, original, attention, phases = fixture()
    original = tuple(t.clone() for t in original)
    original[0][0, 2, 1, 3] += 2
    original[1][0, 1, 6, 0] -= 3
    before = tuple(t.clone() for t in (source, *original, *phases))
    packet = study.construct(source, original, attention, phases)
    assert packet.patches[2][0].indices.numel() >= 1
    assert packet.patches[1][1].indices.numel() >= 1
    k, v = (torch.full((1, 2, 19, 4), -71, dtype=torch.bfloat16) for _ in range(2))
    packet.fill_group(attention, 0, 1, 3, k, rope_scratch=v)
    packet.fill_group(attention, 1, 1, 3, v)
    assert bits(k[..., :7, :], original[0][:, 1:3])
    assert bits(v[..., :7, :], original[1][:, 1:3])
    assert bool((k[..., 7:, :] == -71).all())
    assert bool((v[..., 7:, :] == -71).all())
    assert all(bits(a, b) for a, b in zip((source, *original, *phases), before, strict=True))
    assert packet.component_bytes == backing_bytes(packet.tensors())
    assert study.owner(packet.normalized) != study.owner(source)
    assert packet.cos is phases[0] and packet.sin is phases[1]


def test_shared_phases_count_once_across_layers_and_capture_transfer_is_explicit():
    source, original, attention, phases = fixture()
    first = study.construct(source, original, attention, phases, take_source_ownership=True)
    second = study.construct(source, original, attention, phases)
    assert first.normalized is source
    assert first.source_ownership == "caller_transferred_private_capture"
    assert backing_bytes((*first.tensors(), *second.tensors())) == (
        first.component_bytes + second.component_bytes - backing_bytes(phases)
    )


def test_functional_raw_heads_remove_patches_and_skip_projection(monkeypatch):
    args = fixture()
    packet = study.construct(*args)
    old_bytes = packet.component_bytes
    expected_increment = packet.raw_increment_costs[1]
    cached = study.reconfigure_cached_heads(packet, args[1], (1,))
    assert cached.cached_heads == (1,)
    assert cached.patches[1] is None
    assert cached.component_bytes == old_bytes + expected_increment
    assert cached.raw_increment_costs[1] == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("A functional raw head must bypass projection")

    monkeypatch.setattr(study, "_project_head", forbidden)
    target = torch.empty((1, 1, 11, 4), dtype=torch.bfloat16)
    for component in (0, 1):
        cached.fill_group(args[2], component, 1, 2, target)
        assert bits(target[..., :7, :], args[1][component][:, 1:2])


@pytest.mark.parametrize("bad", [(-1,), (4,), (True,), (2, 1), (0, 0), (0, 1, 2, 3)])
def test_invalid_functional_head_set_rejected(bad):
    with pytest.raises(ValueError, match="cached heads"):
        study.construct(*fixture(), cached_heads=bad)


def test_source_weights_and_destination_aliases_rejected():
    source, original, attention, phases = fixture()
    packet = study.construct(source, original, attention, phases)
    alias = packet.normalized.view(1, 7, 4, 4).transpose(1, 2)
    with pytest.raises(ValueError):
        packet.fill_group(attention, 1, 0, 4, alias)
    target = torch.empty((1, 1, 7, 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="disjoint"):
        packet.fill_group(attention, 0, 0, 1, target, rope_scratch=target)
    with torch.no_grad():
        attention.k_proj.weight.add_(1)
    with pytest.raises(ValueError, match="version changed"):
        packet.fill_group(attention, 1, 0, 1, target)


def test_nonfinite_source_and_bad_shapes_fail_closed():
    source, original, attention, phases = fixture()
    source[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        study.construct(source, original, attention, phases)
    source, original, attention, phases = fixture()
    with pytest.raises(ValueError, match="source"):
        study.construct(source[..., :-1], original, attention, phases)


def test_observer_sees_existing_scratch_not_full_layer_reconstruction():
    args = fixture()
    packet = study.construct(*args)
    target, scratch = (torch.empty((1, 1, 10, 4), dtype=torch.bfloat16) for _ in range(2))
    records = []
    packet.fill_group(
        args[2], 0, 0, 1, target, rope_scratch=scratch, workspace_observer=records.append
    )
    assert records == [(target, scratch)]
    assert packet.diagnostics["full_layer_reconstruction"] is False
    assert packet.diagnostics["whole_request_peak_certified"] is False
