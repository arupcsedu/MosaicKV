import numpy as np
import pytest
import torch

from mosaickv.evaluation.mixed_source_feasibility import (
    bounded_allocate,
    decode_mixed_cpu,
    screen_mixed,
)
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.refined_source_feasibility import table_pack


def test_allocator_finds_feasible_integral_precision_without_relaxing_rows():
    errors = np.array(
        [[[9, 4, 1, 0], [9, 4, 1, 0]], [[0, 0, 1, 0], [9, 4, 1, 0]]], dtype=np.float64
    )
    cost = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float64)
    limit = np.array([2, 2], dtype=np.float64)
    choice, info = bounded_allocate(errors, cost, limit, metadata_bytes=1)
    assert info["lp_status"] == 0
    assert info["integral_feasible"]
    assert bool(np.all(errors[:, np.arange(2), choice].sum(1) <= limit + 1e-12))
    assert not info["fractional_layout_is_measured_cache"]
    assert info["fractional_source_payload_lower_bound"] <= cost[:, 2].sum() + 1 + 1e-9


def test_constant_source_has_actual_charged_smaller_format():
    x = torch.full((1, 32, 16), 1.25, dtype=torch.bfloat16)
    part = screen_mixed(x, torch.zeros(32, dtype=torch.bool), ())
    row = part["rows"][0]
    assert row["bit_group_counts"]["1"] == 2
    assert row["candidate_source_mse"] == row["baseline_source_mse"] == 0
    assert row["numerical_screen_passed"]
    assert row["charged_code_padding_bytes"] == 64
    assert row["candidate_cpu_payload_bytes"] < row["raw_source_cpu_payload_bytes"]
    body = part["candidates"][row["family"]]
    assert body["codes"].numel() % 128 == 0
    assert body["other_bytes"] > 0
    assert not row["gpu_cache_backend_implemented"]
    assert row["serialized_cpu_roundtrip_bitwise"]


def test_parent_error_infeasibility_rejected():
    errors = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="parent layout"):
        bounded_allocate(errors, np.ones((3, 4)), np.zeros(2), metadata_bytes=0)


@pytest.mark.parametrize("seed", [31, 41])
def test_mixed_source_reconstruction_and_byte_policy_are_deterministic(seed):
    x = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(seed)).bfloat16()
    mask = torch.zeros(32, dtype=torch.bool)
    mask[[0, 31]] = True
    first = screen_mixed(x, mask, (0, 31))
    second = screen_mixed(x, mask, (0, 31))
    assert first["rows"] == second["rows"]
    assert first["rows"][0]["per_row_source_screen_passed"]


def test_serialized_decoder_exercises_all_four_bit_widths_and_partial_time_group():
    n = 137
    x = torch.randn(n, 32, generator=torch.Generator().manual_seed(21)).bfloat16()
    codes, headers, code_offsets, header_offsets, expected = [], [], [0], [0], []
    for group, bits in enumerate((1, 2, 3, 4)):
        source = x[:, group * 8 : (group + 1) * 8]
        if bits <= 2:
            from mosaickv.evaluation.refined_source_feasibility import decode_table

            body = table_pack(source, torch.ones(n, dtype=torch.bool), bits=bits)
            h = body.centers.view(torch.uint8).flatten()
            expected.append(decode_table(body))
        else:
            body = pack(source[None, None], bits=bits, axis=2)
            h = torch.cat([t.view(torch.uint8).flatten() for t in (body.minimum, body.scale)])
            expected.append(unpack_cpu(body)[0, 0])
        codes.append(body.codes)
        headers.append(h)
        code_offsets.append(code_offsets[-1] + len(body.codes))
        header_offsets.append(header_offsets[-1] + len(h))
    owners = (
        torch.cat(codes),
        torch.cat(headers),
        torch.tensor(code_offsets, dtype=torch.int32),
        torch.tensor(header_offsets, dtype=torch.int32),
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        torch.tensor([3], dtype=torch.uint8),
    )
    assert torch.equal(decode_mixed_cpu(x.shape, owners), torch.cat(expected, 1))
