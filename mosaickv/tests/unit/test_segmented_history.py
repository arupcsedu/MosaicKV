from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.encoded_projection import EncodedFormat
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.evaluation.projection_archive import ProjectionArchive
from mosaickv.evaluation.segmented_history import make_tables, validate_formats
from mosaickv.evaluation.storage_runs import cache_from_formats


def formats(count):
    output = []
    for i in range(count):
        body = pack_cpu(torch.full((1, 1, 16, 256), 1 + i, dtype=torch.bfloat16))
        union = torch.arange(16, dtype=torch.int32)
        archive = ProjectionArchive(
            body,
            union,
            union[None].expand(2, -1),
            16,
            torch.ones(1, 16, 128, dtype=torch.bfloat16),
            torch.zeros(1, 16, 128, dtype=torch.bfloat16),
        )
        patches = tuple(
            SparseBitPatch(
                torch.tensor([0, 16 * 128 + 129], dtype=torch.int32),
                torch.tensor([1, 2], dtype=torch.bfloat16),
            )
            for _ in range(2)
        )
        output.append(
            EncodedFormat(
                "encoded_projection",
                (1, 2, 16, 128),
                archive=archive,
                patches=patches,
                output_slots=union[None].expand(2, -1),
                literal_checkpoints=torch.empty(0, dtype=torch.int16),
            )
        )
    return tuple(output)


@pytest.mark.parametrize("count", [2, 3, 7])
def test_tables_preserve_source_owners_and_translate_exact_patches(count):
    original = formats(count)
    owners = [f.archive.source.buffers[0].data_ptr() for f in original]
    t = make_tables(original, heads=2)
    assert t.pointers[:, 0].tolist() == owners
    assert t.count == count
    assert t.pointers.shape == (count, 8)
    assert t.raw.tolist() == [int(f.archive.source.raw) for f in original]
    expected = []
    for i in range(count):
        expected.extend([i * 16 * 128, count * 16 * 128 + i * 16 * 128 + 129])
    assert t.patches[0].indices.tolist() == expected
    assert t.patches[1].values.tolist() == [1, 2] * count
    assert all(
        f.archive.source.buffers[0].data_ptr() == p for f, p in zip(original, owners, strict=True)
    )


def test_incompatible_maps_or_phase_dtype_fail_closed():
    f = formats(2)
    with pytest.raises(ValueError, match="original"):
        validate_formats(f[:1], heads=2)
    f[0].output_slots[0, 0] = 15
    with pytest.raises(ValueError, match="chronological"):
        validate_formats(f, heads=2)
    f = formats(2)
    object.__setattr__(f[0].archive, "cos", f[0].archive.cos.float())
    with pytest.raises(ValueError, match="original"):
        validate_formats(f, heads=2)


def test_segmented_requires_direct_source_history():
    for kw in (
        {},
        {"source_history": True},
        {
            "source_history": True,
            "source_history_constructor": "direct",
            "consolidated_history": True,
        },
    ):
        with pytest.raises(ValueError, match="segmented"):
            cache_from_formats(
                SimpleNamespace(), (), updates=128, audit=False, segmented_history=True, **kw
            )


def test_growing_history_stride_does_not_trigger_compilation():
    from mosaickv.evaluation.segmented_history_kernel import _project

    p = next(p for p in _project.params if p.name == "HEAD_OUT")
    assert not p.is_constexpr and p.do_not_specialize
