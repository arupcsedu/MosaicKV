import torch
from transformers.cache_utils import DynamicCache

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.packed_attention_decode import AppendAudit, kv_owners


def test_native_append_peak_includes_actual_overlapping_owners():
    cache = DynamicCache()
    key = torch.ones(1, 2, 4, 128, dtype=torch.bfloat16)
    cache.update(key.clone(), key.clone(), 0)
    initial = backing_bytes(kv_owners(cache))
    audit = AppendAudit(cache)
    with audit:
        for _ in range(3):
            cache.update(key[..., :1, :].clone(), key[..., :1, :].clone(), 0)
    final = backing_bytes(kv_owners(cache))
    assert audit.appends == 6
    assert final > initial and audit.peak > final
    assert final == 2 * 1 * 2 * 7 * 128 * 2
