from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.source_replacement import compact


def formats():
    n, d = 32, 128
    phases = torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16()
    union, slots = torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None]
    result, mapping = [], None
    for _ in range(2):
        body = pack(torch.zeros(1, 1, n, d).bfloat16(), bits=3, axis=2)
        smaller, _ = compact(body, torch.ones(n, dtype=torch.bool))
        if mapping is None:
            mapping = smaller.row_map
        result.append(
            SharedSourceFormat(
                replace(smaller, row_map=mapping, row_map_shared=True),
                (1, 1, n, d),
                torch.empty(0, dtype=torch.int32),
                (),
                *phases,
                union,
                slots,
            )
        )
    return tuple(result)


@pytest.mark.parametrize("encoded", (False, True))
def test_shared_map_is_not_charged_as_private_in_specs_or_bound_bodies(encoded):
    from mosaickv.evaluation.layer_code_delta import EncodedBody
    from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes

    packets = formats()
    if encoded:
        packets = tuple(
            replace(
                p,
                source=EncodedBody(
                    replace(p.source, codes=None),
                    DeviceCodes(p.source.codes.numel(), p.source.codes),
                ),
            )
            for p in packets
        )
    maps = tuple(p.source_shared()[0] for p in packets)
    assert maps[0] is maps[1]
    assert all(not any(t is maps[0] for t in p.private()) for p in packets)
    private = sum(backing_bytes(p.private()) for p in packets)
    shared = backing_bytes(tuple(t for p in packets for t in p.shared()))
    independent = backing_bytes(tuple(t for p in packets for t in p.tensors()))
    assert private + shared == independent


def test_actual_cache_counters_match_independent_enumeration_and_include_initial_shared():
    from mosaickv.evaluation.storage_runs import cache_from_formats

    packets = formats()
    adapter = SimpleNamespace(
        _language_layers=lambda: (
            SimpleNamespace(self_attn=torch.nn.Module()),
            SimpleNamespace(self_attn=torch.nn.Module()),
        )
    )
    cache = cache_from_formats(adapter, packets, updates=1, audit=True, sealed=False)
    actual = backing_bytes(
        (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))
    )
    result = cache.storage_record()  # must keep the independent safety check
    assert cache.initial_bytes == actual
    assert result["initial_active_kv_storage_bytes"] == actual
    assert result["final_active_kv_storage_bytes"] == actual
    assert result["peak_active_kv_storage_bytes"] == actual


def test_shared_flag_cannot_hide_an_absent_map():
    p = formats()[0]
    p = replace(p, source=replace(p.source, row_map=None))
    with pytest.raises(ValueError, match="real map"):
        p.private()
