import pytest
import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.refined_source_feasibility import (
    TableSource,
    apply_patch,
    decode_table,
    refine,
    screen_refined,
    serialize_codes,
    spatial_assignment,
    table_pack,
)


@pytest.mark.parametrize("bits", [1, 2])
def test_table_byte_roundtrip_stored_center_and_partial_group(bits):
    integer = torch.arange(24 * 8).reshape(24, 8) % (1 << bits)
    centers = torch.arange(1 << bits).expand(24, 1, -1).bfloat16().contiguous()
    body = TableSource((8, 24), bits, serialize_codes(integer, bits), centers)
    assert torch.equal(decode_table(body), integer.T.bfloat16())
    x = torch.ones(137, 24).bfloat16() * 1.25
    valid = torch.ones(137, dtype=torch.bool)
    packed = table_pack(x, valid, bits=bits)
    assert torch.equal(decode_table(packed), x)
    assert packed.codes.numel() == x.numel() * bits // 8
    assert packed.centers.shape == (24, 2, 1 << bits)


def test_exact_mask_and_padding_do_not_change_fitted_centers():
    x = torch.ones(131, 8).bfloat16()
    valid = torch.ones(131, dtype=torch.bool)
    valid[10] = False
    x[10] = 1000
    body = table_pack(x, valid, bits=2)
    assert bool((body.centers == 1).all())


def test_residual_selection_uses_bf16_source_addition():
    base = torch.full((136, 8), 256, dtype=torch.bfloat16)
    original = base.clone()
    original[::2] = 258
    residual = original - base
    body = table_pack(
        residual, torch.ones(136, dtype=torch.bool), bits=1, base=base, original=original
    )
    assert torch.equal(decode_table(body, base), original)


@pytest.mark.parametrize("seed", range(5))
def test_literal_refinement_enforces_each_row_and_owns_payload(seed):
    g = torch.Generator().manual_seed(seed)
    original = torch.randn(16, 64, generator=g).bfloat16()
    baseline = (original.float() + 0.05).bfloat16()
    decoded = (original.float() + torch.randn(16, 64, generator=g)).bfloat16()
    needed = torch.ones(16, dtype=torch.bool)
    needed[0] = False
    patch, reconstructed = refine(original, decoded, baseline, needed)
    before = (baseline.double() - original.double()).square().mean(-1)
    after = (reconstructed.double() - original.double()).square().mean(-1)
    assert bool((after[needed] <= before[needed] + 1e-12).all())
    assert torch.equal(reconstructed[0], decoded[0])
    assert torch.equal(apply_patch(decoded, patch), reconstructed)
    for tensor in patch.tensors():
        assert tensor.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert backing_bytes(patch.tensors()) == sum(
        t.numel() * t.element_size() for t in patch.tensors()
    )


def test_sparse_literals_and_full_rows_both_reconstruct():
    original = torch.zeros(4, 32).bfloat16()
    decoded = original.clone()
    decoded[0, 3] = 10
    decoded[1] = 1
    patch, reconstructed = refine(original, decoded, original, torch.ones(4, dtype=torch.bool))
    assert torch.equal(reconstructed, original)
    assert patch.columns.tolist() == [3]
    assert len(patch.full_rows) == 1
    assert patch.row_map.tolist() == [-1, 0, -1, -1]


def test_spatial_memberships_preserve_original_logical_coordinates():
    text = (0, 1, 578, 579)
    positions, assignment, mapping = spatial_assignment(
        text, length=580, patch_grid=(24, 24), group=(2, 2)
    )
    assert positions.tolist() == list(range(2, 578))
    assert assignment[0] == assignment[1] == assignment[24] == assignment[25] == 0
    assert assignment[2] == 1 and assignment[48] == 12
    assert int(assignment.max()) == 143
    assert mapping[list(text)].tolist() == [-1] * 4
    assert torch.equal(mapping[positions], torch.arange(576, dtype=torch.int32))
    with pytest.raises(ValueError):
        spatial_assignment(text, length=581, patch_grid=(24, 24), group=(2, 2))


def test_all_registered_source_families_run_deterministically():
    x = torch.ones(1, 580, 8).bfloat16()
    x[:, 2:578, 4:] = 2
    exact = torch.zeros(580, dtype=torch.bool)
    exact[[0, 1, 578, 579]] = True
    first = screen_refined(x, exact, (0, 1, 578, 579))
    second = screen_refined(x, exact, (0, 1, 578, 579))
    assert first["rows"] == second["rows"]
    assert [r["family"] for r in first["rows"]] == [
        "lut2",
        "spatial4_binary_residual",
        "spatial8_binary_residual",
    ]
    for row in first["rows"]:
        assert row["per_row_source_screen_passed"]
        assert row["candidate_cpu_payload_bytes"] > 0
        assert not row["gpu_cache_backend_implemented"]
