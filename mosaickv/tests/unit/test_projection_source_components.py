import torch

from mosaickv.evaluation.projection_source_components import compare


def test_zero_query_separates_key_and_value_damage_without_mutating_controls():
    torch.manual_seed(419)
    key = torch.randn(1, 2, 8, 128).bfloat16()
    value = torch.randn_like(key)
    query = torch.zeros(1, 2, 3, 128).bfloat16()
    before = key.clone(), value.clone()
    out = compare(
        query,
        key,
        value,
        key,
        value,
        key + 2,
        value + 1,
        torch.arange(8).repeat(2, 1),
        torch.arange(5, 8),
    )
    assert out["refitted_K_old_quantized_V"]["error_sse"] == 0
    assert out["old_quantized_K_refitted_V"]["error_sse"] > 0
    assert out["FullKV_K_refitted_V"]["error_sse"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, (key, value), strict=True))
