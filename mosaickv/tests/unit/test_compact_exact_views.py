"""Execution-only source views: no selection, logical-position, or byte changes."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from mosaickv.cache_state import ExactTier, FullKVState, MosaicKVState, PrototypeTier


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_random_source_views_match_eager_reference(backend):
    rng = np.random.default_rng(516)
    for _ in range(24):
        length = int(rng.integers(2, 18))
        heads = int(rng.integers(1, 5))
        batch = int(rng.integers(1, 3))
        layers = []
        for _layer in range(2):
            # Noncontiguous source tensors, different K/V widths and dtypes.
            key = rng.normal(size=(batch, heads, length, 6)).astype(np.float32)[..., ::2]
            value = rng.normal(size=(batch, heads, length, 8)).astype(np.float64)[..., ::2]
            if backend == "torch":
                torch = pytest.importorskip("torch")
                key, value = torch.from_numpy(key), torch.from_numpy(value)
            layers.append((key, value))
        full = FullKVState.from_tensors(
            layers,
            block_size=int(rng.integers(1, 6)),
            logical_positions=tuple(3 * i for i in range(length)),
            original_logical_sequence_length=3 * length,
            next_decode_position=3 * length + 1,
        )
        selected = tuple(full.blocks[int(i)] for i in rng.permutation(len(full.blocks)))
        compact = full.gather_exact_blocks(selected, gather_mode="source_views")
        reference = full.gather_exact_blocks(selected, gather_mode="source_views_reference")
        assert compact.materialized_block_view_count == 0
        assert compact.blocks == reference.blocks
        assert compact.active_bytes == reference.active_bytes == full.active_bytes
        state = MosaicKVState.create(full, exact=compact)
        assert compact.materialized_block_view_count == 0
        assert state.logical_positions == full.logical_positions
        assert state.statistics == MosaicKVState.create(full, exact=reference).statistics
        for observed, expected in zip(
            (*compact.key_blocks, *compact.value_blocks),
            (*reference.key_blocks, *reference.value_blocks),
            strict=True,
        ):
            if backend == "torch":
                assert torch.equal(observed, expected)
                assert (
                    observed.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
                )
            else:
                np.testing.assert_array_equal(observed, expected)
                assert np.shares_memory(observed, expected)
        assert compact.materialized_block_view_count == 2 * len(selected)
        rebuilt = state.reconstruct_full_state(full)
        assert rebuilt.logical_positions == full.logical_positions


def _source():
    import torch

    key = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(1, 2, 9, 4)
    return FullKVState.from_tensors(
        ((key, key + 500), (key + 1000, key + 1500)),
        block_size=2,
        mandatory_logical_positions=(0,),
    )


def test_lazy_sequence_is_cached_indexable_and_immutable():
    full = _source()
    tier = full.gather_exact_blocks(full.blocks, gather_mode="source_views")
    assert tier.key_blocks[0] is tier.key_blocks[0]
    assert tier.value_blocks[-1] is tier.value_blocks[len(tier.blocks) - 1]
    assert tier.materialized_block_view_count == 2
    assert tier.key_blocks[:1] == (tier.key_blocks[0],)
    assert len(tier.key_blocks[::2]) == (len(tier.blocks) + 1) // 2
    with pytest.raises(IndexError):
        tier.key_blocks[len(tier.blocks)]
    with pytest.raises(FrozenInstanceError):
        tier.key_blocks.key = False
    with pytest.raises(ValueError, match="share a validated bank"):
        ExactTier(tier.blocks, tier.value_blocks, tier.key_blocks)
    with pytest.raises(ValueError, match="differ from their validated bank"):
        ExactTier(tuple(reversed(tier.blocks)), tier.key_blocks, tier.value_blocks)


@pytest.mark.parametrize("mode", ["source_views", "source_views_reference"])
def test_duplicate_and_foreign_descriptors_rejected(mode):
    full = _source()
    with pytest.raises(ValueError):
        full.gather_exact_blocks((full.blocks[0], full.blocks[0]), gather_mode=mode)
    for bad in (
        replace(full.blocks[0], byte_size=full.blocks[0].byte_size + 1),
        replace(full.blocks[0], key_dtype="float64"),
        replace(full.blocks[0], key_device="cuda:7"),
    ):
        with pytest.raises(ValueError):
            full.gather_exact_blocks((bad,), gather_mode=mode)


def test_compact_validates_materialized_view_mutation_and_cross_tier_aliases():
    full = _source()
    kept = tuple(b for b in full.blocks if b.mandatory)
    other = next(b for b in full.blocks if not b.mandatory)
    tier = full.gather_exact_blocks(kept, gather_mode="source_views")
    prototypes = PrototypeTier((other,), (tier.key_blocks[0],), (tier.value_blocks[0],), (0,))
    with pytest.raises(ValueError, match="same tensor object"):
        MosaicKVState.create(full, exact=tier, prototypes=prototypes)
    tier.key_blocks[0].resize_(1)
    with pytest.raises(ValueError, match="metadata/storage mismatch"):
        _ = tier.active_bytes
    with pytest.raises(ValueError, match="metadata/storage mismatch"):
        MosaicKVState.create(full, exact=tier)


def test_source_rebinding_does_not_redirect_preexisting_views():
    import torch

    full = _source()
    compact = full.gather_exact_blocks(full.blocks, gather_mode="source_views")
    eager = full.gather_exact_blocks(full.blocks, gather_mode="source_views_reference")
    key = full.layers[0].key
    key.set_(torch.full_like(key, -50))
    assert torch.equal(compact.key_blocks[0], eager.key_blocks[0])
    assert not torch.equal(compact.key_blocks[0], key[:, :1, :2])


def test_alternate_cache_axes_and_mandatory_policy_are_preserved():
    key = np.arange(2 * 7 * 3 * 4, dtype=np.float32).reshape(2, 7, 3, 4)
    full = FullKVState.from_tensors(
        ((key, key + 200),),
        block_size=2,
        head_dimension=2,
        sequence_dimension=1,
        mandatory_logical_positions=(0,),
    )
    selected = tuple(b for b in full.blocks if b.physical_cache_indices[0] < 4)
    first = full.gather_exact_blocks(selected, gather_mode="source_views")
    second = full.gather_exact_blocks(selected, gather_mode="source_views_reference")
    for a, b in zip(first.key_blocks, second.key_blocks, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.active_bytes == second.active_bytes
    MosaicKVState.create(full, exact=first)
    removed_mandatory = tuple(b for b in selected if not b.mandatory)
    tier = full.gather_exact_blocks(removed_mandatory, gather_mode="source_views")
    with pytest.raises(ValueError, match="mandatory"):
        MosaicKVState.create(full, exact=tier)


def test_tensor_subclasses_use_validated_eager_fallback():
    class Array(np.ndarray):
        pass

    key = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4).view(Array)
    full = FullKVState.from_tensors(((key, key.copy()),), block_size=1)
    tier = full.gather_exact_blocks(full.blocks, gather_mode="source_views")
    assert isinstance(tier.key_blocks, tuple)
    assert tier.materialized_block_view_count == 2 * len(full.blocks)


def test_dense_runtime_packing_preserves_physical_storage_and_padding_without_views():
    import torch

    from mosaickv.backends.hf_runtime import pack_runtime_cache

    full = _source()
    selected = tuple(
        b for b in full.blocks if b.mandatory or b.physical_cache_indices[0] < 4 + 2 * b.kv_head
    )
    adapter = SimpleNamespace(capabilities=SimpleNamespace(cached_key_state="post_rope"))
    tiers = [
        full.gather_exact_blocks(selected, gather_mode=mode)
        for mode in ("source_views_reference", "source_views")
    ]
    packed = [
        pack_runtime_cache(adapter, full, MosaicKVState.create(full, exact=tier), (), ())
        for tier in tiers
    ]
    assert tiers[1].materialized_block_view_count == 0
    assert packed[0].slot_records == packed[1].slot_records
    assert packed[0].prompt_capacity == packed[1].prompt_capacity == 6
    for first, second in zip(packed[0].snapshot.layers, packed[1].snapshot.layers, strict=True):
        for a, b in ((first.key, second.key), (first.value, second.value)):
            assert torch.equal(a, b)
            assert a.untyped_storage().nbytes() == b.untyped_storage().nbytes()
            assert a.untyped_storage().nbytes() == a.numel() * a.element_size()
    for a, b in zip(packed[0].validity_masks, packed[1].validity_masks, strict=True):
        assert torch.equal(a, b)
        assert not bool(a.all())  # Dense padding remains included, not silently removed.
