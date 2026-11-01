from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_kv_lease import fill_component
from mosaickv.evaluation.sparse_original_exceptions import construct, formats


def test_header_provenance_distinguishes_lossless_encoding_from_lossy_fp16():
    from evicastkv_shared_source_runtime import precision_header_label

    for name in (
        "source3_exact_exceptions",
        "baseline_lookm_reimpl__exact_exceptions",
        "minimum__source3_fp32_tiled_text",
    ):
        assert precision_header_label(name) == "lossless_bf16_minimum_fp32_scale"
    assert precision_header_label("baseline_prefixkv_reimpl__fp16") == "fp16"
    assert precision_header_label("source3_fp32_tiled_text") == "fp32"


def fixture():
    original = tuple(torch.arange(64).reshape(1, 2, 4, 8).bfloat16() + i for i in (0, 4))
    actual = tuple(t + 1 for t in original)
    base = ExactFormat(
        "kv",
        tuple(original[0].shape),
        pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in actual),
    )
    weights = tuple(torch.ones_like(t, dtype=torch.float32) for t in original)
    return base, original, actual, weights


@pytest.mark.parametrize("budget", [0, 1, 5, 6, 17, 60, 10000])
def test_hard_physical_budget_and_deterministic_ties(budget):
    base, original, actual, w = fixture()
    packet, proof = construct(base, original, actual, w, budget_bytes=budget)
    assert proof["exact_exception_bytes"] <= budget
    assert packet.component_bytes == base.component_bytes + proof["exact_exception_bytes"]
    expected = min(budget // 6, 128)
    assert proof["restored_scalars"] == expected
    assert proof["weighted_error_removed"] == expected
    ids = torch.cat(tuple(p.indices.long() + i * 64 for i, p in enumerate(packet.patches)))
    assert torch.equal(ids, torch.arange(expected))
    assert all(
        p.values.data_ptr() != t.data_ptr() for p, t in zip(packet.patches, original, strict=True)
    )
    again, _ = construct(base, original, actual, w, budget_bytes=budget)
    assert all(
        torch.equal(a.indices, b.indices)
        for a, b in zip(packet.patches, again.patches, strict=True)
    )


def test_exact_rows_and_unselected_values_unchanged_in_both_lease_paths():
    base, original, actual, w = fixture()
    w[0][..., 0, :] = 0
    packet, _ = construct(base, original, actual, w, budget_bytes=120)
    values = packet.materialize()
    for component in range(2):
        destination = torch.full((1, 2, 7, 8), -71, dtype=torch.bfloat16)
        fill_component(packet, None, component, destination)
        assert torch.equal(
            destination[..., :4, :].view(torch.int16), values[component].view(torch.int16)
        )
        assert (destination[..., 4:, :] == -71).all()
    assert torch.equal(values[0][..., 0, :], actual[0][..., 0, :])
    assert backing_bytes(packet.tensors()) == packet.component_bytes


def test_zero_error_and_zero_weight_need_no_residual_storage():
    base, original, actual, w = fixture()
    packet, proof = construct(base, original, original, w, budget_bytes=100)
    assert proof["restored_scalars"] == 0 and packet.component_bytes == base.component_bytes
    for weight in w:
        weight.zero_()
    assert construct(base, original, actual, w, budget_bytes=100)[1]["exact_exception_bytes"] == 0


def test_invalid_scores_and_cpu_transport_fail_explicitly():
    base, original, actual, w = fixture()
    w[0].fill_(-1)
    with pytest.raises(ValueError):
        construct(base, original, actual, w, budget_bytes=100)
    with pytest.raises(NotImplementedError):
        formats(
            SimpleNamespace(device=torch.device("cpu")),
            None,
            None,
            None,
            None,
            None,
            None,
            cap_bytes=1,
        )
