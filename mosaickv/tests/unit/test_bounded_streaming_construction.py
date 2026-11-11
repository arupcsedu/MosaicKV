"""CPU fixtures for allocation-free verification and streaming raw/source construction."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.bounded_component_source import construct
from mosaickv.evaluation.bounded_ragged_raw import build_raw_packet_layer, build_raw_packets
from mosaickv.evaluation.bounded_word_equality import equal_words
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import native_component


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_fixture():
    torch.manual_seed(27)
    heads, length, dim = 2, 7, 128
    width = heads * dim
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        v_proj=torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16).eval(),
        num_key_value_groups=1,
    )
    source = torch.randn(1, length, width).bfloat16()
    angles = torch.randn(1, length, dim // 2).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    pair = tuple(native_component(source, attention, c, *phases) for c in (0, 1))
    return source, pair, attention, phases


def test_word_comparison_obeys_logical_strides_and_preserves_inputs():
    source = torch.arange(2 * 7 * 128).bfloat16().view(1, 2, 7, 128)
    n_major = source.transpose(1, 2).contiguous().transpose(1, 2)
    before = source.clone(), n_major.clone()
    scratch = ComponentWorkspace(device="cpu", heads=2).head
    assert not n_major.is_contiguous()
    assert equal_words(source, n_major, scratch=scratch)
    assert torch.equal(source, before[0]) and torch.equal(n_major, before[1])
    n_major.view(torch.int16)[0, 1, 2, 3] ^= 1
    assert not equal_words(source, n_major, scratch=scratch)


def test_word_comparison_distinguishes_signed_zero_and_nan_payloads():
    original = torch.tensor([0, -32768, 32704, 32705], dtype=torch.int16).view(1, 1, 1, 4)
    first, second = original.view(torch.bfloat16), original.clone().view(torch.bfloat16)
    scratch = torch.empty(64, dtype=torch.bfloat16)
    assert equal_words(first, second, scratch=scratch)
    second.view(torch.int16)[0, 0, 0, 0] = -32768
    assert not equal_words(first, second, scratch=scratch)
    second.view(torch.int16).copy_(original)
    second.view(torch.int16)[0, 0, 0, 2] = 32705
    assert not equal_words(first, second, scratch=scratch)


@pytest.mark.parametrize("case", ("dtype", "shape", "width_stride", "small_scratch", "alias"))
def test_word_comparison_rejects_unsupported_or_aliasing_inputs(case):
    first = torch.zeros(1, 2, 7, 128, dtype=torch.bfloat16)
    second = first.clone()
    scratch = torch.empty(1024, dtype=torch.bfloat16)
    if case == "dtype":
        second = second.float()
    elif case == "shape":
        second = second[..., :6, :]
    elif case == "width_stride":
        first, second = first[..., ::2], second[..., ::2]
    elif case == "small_scratch":
        scratch = scratch[:2]
    else:
        scratch = first.view(-1)
    with pytest.raises(ValueError):
        equal_words(first, second, scratch=scratch)


def test_exact_source_reuses_workspace_and_requires_no_patch_storage(monkeypatch):
    source, pair, attention, phases = source_fixture()
    workspace = ComponentWorkspace(device="cpu", heads=2)
    empty = torch.empty
    allocations = []

    def only_empty_patch(*args, **kwargs):
        result = empty(*args, **kwargs)
        allocations.append(result.numel())
        return result

    monkeypatch.setattr(torch, "empty", only_empty_patch)
    seen = []
    packet = construct(
        source,
        pair,
        attention,
        phases,
        workspace=workspace,
        exact_only=True,
        take_source_ownership=True,
        workspace_observer=seen.append,
    )
    assert packet.normalized is source
    assert packet.diagnostics["patch_scalars"] == [0, 0]
    assert packet.diagnostics["patch_bytes"] == 0
    assert allocations == [0, 0, 0, 0]
    assert packet.component_bytes == backing_bytes((source, *phases))
    assert len(seen) == 3
    assert all(
        any(t is workspace.component for t in tensors) and any(t is workspace.head for t in tensors)
        for tensors in seen
    )
    for component in (0, 1):
        packet.fill_component(
            attention, component, workspace.component_view(7), rope_scratch=workspace.head
        )
        assert equal_words(pair[component], workspace.component_view(7), scratch=workspace.head)


def test_exact_source_rejects_changed_original_instead_of_patching_or_fallback():
    source, pair, attention, phases = source_fixture()
    pair[1][0, 1, 3, 4] += 2
    workspace = ComponentWorkspace(device="cpu", heads=2)
    with pytest.raises(RuntimeError, match="Exact-only streaming source changed"):
        construct(source, pair, attention, phases, workspace=workspace, exact_only=True)
    with pytest.raises(ValueError, match="charged workspace"):
        construct(source, pair, attention, phases, exact_only=True)


@pytest.mark.parametrize("case", ("heads", "alias", "dtype"))
def test_source_workspace_rejects_incompatible_or_aliasing_owners(case):
    source, pair, attention, phases = source_fixture()
    workspace = ComponentWorkspace(device="cpu", heads=2)
    if case == "heads":
        workspace = ComponentWorkspace(device="cpu", heads=3)
    elif case == "alias":
        workspace.component = source[0]
    else:
        workspace.head = workspace.head.float()
    with pytest.raises(ValueError):
        construct(source, pair, attention, phases, workspace=workspace, exact_only=True)


def raw_fixture():
    torch.manual_seed(27)
    pairs = tuple(tuple(torch.randn(1, 3, 7, 128).bfloat16() for _ in range(2)) for _ in range(2))
    scores = {i: torch.rand(1, 3, 2, 7) for i in range(2)}
    return SimpleNamespace(_language_layers=lambda: (0, 1)), pairs, scores


def test_streamed_raw_equals_whole_builder_positions_words_and_exact_global_bytes():
    adapter, pairs, windows = raw_fixture()
    expected, _ = build_raw_packets(adapter, pairs, windows, 20 * 512, (0, 6))
    workspace = ComponentWorkspace(device="cpu", heads=3)
    actual, reports, events = [], [], []
    for index, pair in enumerate(pairs):
        scores = windows[index][0].float().mean(-2)
        packet, report = build_raw_packet_layer(
            pair,
            scores,
            (0, 6),
            layer_index=index,
            total_layers=2,
            target_bytes=20 * 512,
            workspace=workspace,
            workspace_observer=events.append,
        )
        actual.append(packet)
        reports.append(report)
    assert sum(p.component_bytes for p in actual) == 20 * 512
    assert [p.prompt_lengths for p in actual] == [(4, 4, 3), (3, 3, 3)]
    for first, second in zip(expected, actual, strict=True):
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(first.tensors(), second.tensors(), strict=True)
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(first.original_positions, second.original_positions, strict=True)
        )
    assert all(report["constructor_KV_gather_temporary_allocations"] == 0 for report in reports)
    assert len(events) == 14
    assert all(any(t is workspace.head for t in event) for event in events)


@pytest.mark.parametrize(
    "override",
    ({"layer_index": 2}, {"total_layers": 0}, {"target_bytes": 513}, {"mandatory": (6, 0)}),
)
def test_streamed_raw_rejects_bad_global_capacity_or_mandatory(override):
    _, pairs, windows = raw_fixture()
    values = dict(layer_index=0, total_layers=2, target_bytes=20 * 512, mandatory=(0, 6))
    values.update(override)
    mandatory = values.pop("mandatory")
    with pytest.raises(ValueError):
        build_raw_packet_layer(
            pairs[0],
            windows[0][0].mean(-2),
            mandatory,
            workspace=ComponentWorkspace(device="cpu", heads=3),
            **values,
        )


def test_streamed_raw_rejects_unreduced_scores_and_aliasing_index_scratch():
    _, pairs, windows = raw_fixture()
    workspace = ComponentWorkspace(device="cpu", heads=3)
    kwargs = dict(layer_index=0, total_layers=2, target_bytes=20 * 512, workspace=workspace)
    with pytest.raises(ValueError, match="pre-reduced"):
        build_raw_packet_layer(pairs[0], windows[0], (0, 6), **kwargs)
    workspace.head = pairs[0][0].view(-1)
    with pytest.raises(ValueError, match=r"alias|compatible"):
        build_raw_packet_layer(pairs[0], windows[0][0].mean(-2), (0, 6), **kwargs)


def test_streamed_raw_rejects_shape_valid_original_alias_scratch():
    pair = tuple(torch.zeros(1, 3, 1024, 128, dtype=torch.bfloat16) for _ in range(2))
    workspace = ComponentWorkspace(device="cpu", heads=3)
    workspace.head = pair[0][0, 0]
    with pytest.raises(ValueError, match="alias immutable"):
        build_raw_packet_layer(
            pair,
            torch.ones(3, 1024),
            (0,),
            layer_index=0,
            total_layers=1,
            target_bytes=3 * 128 * 512,
            workspace=workspace,
        )
