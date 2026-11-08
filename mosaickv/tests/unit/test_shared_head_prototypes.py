import gc

import pytest
import torch

from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.shared_head_prototypes import construct, integers, serialize


def body(n=128, width=64):
    torch.manual_seed(19)
    return pack(torch.randn(1, 1, n, width).bfloat16(), bits=3, axis=2)


@pytest.mark.parametrize("heads", (1, 2, 8, 32))
def test_prototype_covers_every_head_without_original_code_storage(heads):
    original = body(width=128)
    transaction = construct(original, ((4, 5, 6, 7),), heads=heads, mandatory=(0, 127))
    expected = integers(original.codes, 128, 128)
    expected[4:8] = expected[4:8].float().mean(0).round().to(torch.int32)
    actual = integers(transaction.body.codes, 125, 128)[transaction.body.row_map.long()]
    assert torch.equal(actual, expected)
    assert transaction.body.minimum is original.minimum
    assert transaction.body.scale is original.scale
    assert transaction.record()["affected_KV_heads"] == list(range(heads))
    assert (
        transaction.body.codes.untyped_storage().data_ptr()
        != original.codes.untyped_storage().data_ptr()
    )
    del original
    gc.collect()
    assert not transaction.record()["old_source_tensor_alive"]


def test_all_head_residual_repair_restores_source_words_and_positions():
    original = body()
    words = unpack_cpu(original).view(torch.int16).clone()
    transaction = construct(original, ((3, 4, 5), (11, 12, 13)), heads=4)
    del original
    repaired = transaction.promote()
    assert repaired.row_map is None
    assert torch.equal(unpack_cpu(repaired).view(torch.int16), words)
    assert transaction.repair_events == 1
    with pytest.raises(RuntimeError):
        transaction.promote()


@pytest.mark.parametrize("groups", (((0, 1),), ((127, 128),), ((3, 4), (4, 5)), ((4, 3),)))
def test_unsafe_groups_rejected(groups):
    with pytest.raises(ValueError):
        construct(body(256), groups, heads=4, mandatory=(0,))


def test_retention_one_never_changes_source_and_capture_repair_rejected():
    original = body()
    assert construct(original, (), heads=4, retention_one=True) is original
    with pytest.raises(ValueError, match="retention one"):
        construct(original, ((3, 4),), heads=4, retention_one=True)
    transaction = construct(original, ((3, 4),), heads=4)
    with pytest.raises(RuntimeError, match="uncaptured"):
        transaction.promote(captured=True)


def test_exact_code_roundtrip_and_actual_map_accounting():
    original = body(width=128)
    values = integers(original.codes, 128, 128)
    assert torch.equal(serialize(values), original.codes)
    transaction = construct(original, (tuple(range(8, 24)),), heads=32)
    record = transaction.record()
    assert (
        record["original_body_bytes"] - record["replacement_body_bytes"]
        == 15 * 128 * 3 // 8 - 128 * 4
    )
    assert record["CPU_residual_bytes"] == 16 * 128 * 3 // 8
    assert transaction.residual_codes.device.type == "cpu"


def test_caller_mutation_and_request_isolation():
    first, second = body(), body()
    a = construct(first, ((3, 4, 5),), heads=4)
    b = construct(second, ((3, 4, 5),), heads=4)
    before = b.body.codes.clone()
    with torch.inference_mode():
        a.body.codes.zero_()
    a.promote()
    assert torch.equal(before, b.body.codes)
