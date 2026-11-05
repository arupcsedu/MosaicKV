from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache


@pytest.mark.parametrize("audit", [False, True])
def test_audit_exclusions_only_walk_parameters_when_actually_observed(audit):
    cache = RetiringPrefillCache(1, audit=audit)
    original = tuple(torch.randn((1, 1, 3, 128), dtype=torch.bfloat16) for _ in range(2))
    cache.layers[0].update(*original)
    calls = []

    def parameters():
        calls.append(True)
        if not audit:
            raise AssertionError("disabled audit may not enumerate unused parameter exclusions")
        return iter(())

    def encoder(index, attention, pair, args, kwargs, output):
        return ExactFormat(
            "kv",
            tuple(pair[0].shape),
            pair=tuple(PackedBF16(tuple(t.shape), True, (t.clone(),)) for t in pair),
        )

    cache.seal(0, SimpleNamespace(parameters=parameters), encoder)
    assert calls == ([True] if audit else [])
    assert cache.next_layer == 1
    assert cache.layers[0].keys is cache.layers[0].values is None
    for actual, wanted in zip(cache.layers[0].packet.pair, original, strict=True):
        assert torch.equal(actual.buffers[0].view(torch.int16), wanted.view(torch.int16))
        assert actual.buffers[0].untyped_storage().data_ptr() != wanted.untyped_storage().data_ptr()
