import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, lossless_kv_encoder


def test_cached_ownership_matches_fresh_full_scan_at_every_audit_callback():
    cache = RetiringPrefillCache(3, audit=True)
    original = cache.observe
    expected_peak = 0

    def check(extra=()):
        nonlocal expected_peak
        expected_peak = max(expected_peak, backing_bytes((*cache.owned_tensors(), *extra)))
        original(extra)
        assert cache.peak_bytes == expected_peak

    cache.observe = check
    attention = torch.nn.Linear(2, 2)
    backing = torch.zeros(512, dtype=torch.int32)
    for i in range(3):
        pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
        cache.update(*pair, i)
        cache.extra_tensors = (backing[:4], backing[8:9])
        cache.seal(i, attention, lossless_kv_encoder)
        cache.extra_tensors = ()
        cache.observe()
    assert cache.record()["observed_prefill_cache_peak_bytes"] == expected_peak
    assert cache.record()["observed_prefill_cpu_peak_bytes"] == expected_peak
    assert cache.record()["observed_prefill_cuda_peak_bytes"] == 0
    assert cache.record()["combined_peak_includes_CPU_encoder_tensor_temporaries"]


def test_audit_does_not_rescan_all_sealed_packet_descriptors_on_every_operation():
    cache = RetiringPrefillCache(1, audit=True)
    calls = []
    original = cache.owned_tensors

    def counted():
        calls.append(1)
        return original()

    cache.owned_tensors = counted
    extra = torch.zeros(1024)
    for _ in range(100):
        cache.observe((extra,))
    assert len(calls) == 1
    cache.extra_tensors = (extra[:3],)
    cache.observe()
    assert len(calls) == 2
    assert cache.peak_bytes == extra.untyped_storage().nbytes()
