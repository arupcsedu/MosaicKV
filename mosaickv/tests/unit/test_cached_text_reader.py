from dataclasses import dataclass
from itertools import product

import pytest
import torch

from mosaickv.evaluation.cached_text_reader import CachedTextFormat, build, convert, eligible
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.fused_lossless_attention import bits_equal
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_kv_lease import fill_component
from mosaickv.evaluation.text_source_storage import TextSourceFormat


@dataclass(frozen=True)
class SharedExact(ExactFormat):
    def shared(self):
        return (self.phase,)

    @property
    def phase(self):
        return self.pair[0].buffers[0]


class Projection:
    def __init__(self):
        self.slots = torch.tensor([[1, 4, -1], [0, -1, 6]], dtype=torch.int16)
        self.union = torch.arange(3)
        self.values = torch.arange(2 * 2 * 3 * 128, dtype=torch.int16).reshape(2, 2, 3, 128)
        self.values[0, 0, 0, :3] = torch.tensor([-32768, 32705, -128], dtype=torch.int16)

    def private(self):
        return self.slots, self.values

    def shared(self):
        return ()

    def fill(self, attention, targets):
        for c, target in enumerate(targets):
            for head in range(2):
                valid = self.slots[head] >= 0
                target[0, head, self.slots[head, valid].long()] = self.values[c, head, valid].view(
                    torch.bfloat16
                )


def packet():
    words = torch.arange(2 * 7 * 128, dtype=torch.int16).reshape(1, 2, 7, 128)
    base = SharedExact(
        "kv",
        tuple(words.shape),
        tuple(
            PackedBF16(tuple(words.shape), True, ((words + c).view(torch.bfloat16),))
            for c in (0, 42)
        ),
    )
    return TextSourceFormat(base, Projection())


@pytest.mark.parametrize("modes", list(product(("encoded", "resident"), repeat=2)))
def test_exact_rows_preserve_absent_memberships_shared_data_and_strides(modes):
    source = packet()
    candidate = build(source, None, modes)
    assert isinstance(candidate, CachedTextFormat)
    assert candidate.flat_tokens.tolist() == [1, 4, 7, 13]
    assert candidate.base is source.base
    assert candidate.shared()[0] is source.base.phase
    assert candidate.component_bytes == backing_bytes(candidate.tensors())
    for c in (0, 1):
        oracle = torch.empty(source.shape, dtype=torch.bfloat16)
        fill_component(source, None, c, oracle)
        backing = torch.full((1, 2, 12, 128), -71, dtype=torch.bfloat16)
        fill_component(candidate, None, c, backing[..., 2:11, :])
        assert bits_equal(oracle, backing[..., 2:9, :])
        assert (backing[..., :2, :] == -71).all() and (backing[..., 9:, :] == -71).all()
    for a, b in zip(candidate.materialize(None), source.materialize(None), strict=True):
        assert bits_equal(a, b)


def test_conversion_cap_labels_input_binding_and_isolation():
    source = packet()
    policy = dict(
        modes=[["encoded", "resident"]],
        maximum_added_bytes=0,
        base_private_bytes=backing_bytes(source.private()),
    )
    first, report = convert((source,), (None,), policy)
    second, _ = convert((source,), (None,), policy)
    assert report["cached_overlay_layers"] == [0]
    assert (
        first[0].protected[1].buffers[0].data_ptr() != second[0].protected[1].buffers[0].data_ptr()
    )
    with pytest.raises(ValueError, match="labels forbidden"):
        convert((source,), (None,), dict(policy, answer="hidden"))
    with pytest.raises(ValueError, match="different input"):
        convert((source,), (None,), dict(policy, base_private_bytes=1))
    assert build(source, None, ("source", "source")) is source
    assert not eligible(source.base)
    with pytest.raises(ValueError):
        build(source.base, None, ("resident", "resident"))


def test_layer_code_mapper_keeps_cached_rows_and_subclass():
    from mosaickv.evaluation.layer_code_delta import map_bodies

    value = build(packet(), None, ("encoded", "encoded"))
    mapped = map_bodies(value, lambda body: body)
    assert isinstance(mapped, CachedTextFormat)
    assert mapped.protected is value.protected
    assert mapped.shared()[0] is value.base.phase


def test_duplicate_memberships_fail_closed():
    source = packet()
    source.projection.slots[0, 1] = 1
    with pytest.raises(ValueError, match="unique"):
        build(source, None, ("encoded", "encoded"))
