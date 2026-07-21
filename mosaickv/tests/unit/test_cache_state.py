from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.cache_state import (
    CompressionStatistics,
    ExactTier,
    FullKVState,
    KVBlockDescriptor,
    LogicalPositionMap,
    MediaMetadata,
    Modality,
    ModalitySpan,
    MosaicKVState,
    PrototypeTier,
    ResidualTier,
    tensor_storage_bytes,
)


def _spans(modalities: list[Modality]) -> tuple[ModalitySpan, ...]:
    spans: list[ModalitySpan] = []
    start = 0
    for index in range(1, len(modalities) + 1):
        if index == len(modalities) or modalities[index] is not modalities[start]:
            modality = modalities[start]
            spans.append(
                ModalitySpan(
                    start,
                    index,
                    modality,
                    image_index=0 if modality is Modality.IMAGE else None,
                    frame_index=0 if modality is Modality.VIDEO else None,
                    page_index=0 if modality is Modality.IMAGE else None,
                    region=(0.0, 0.0, 1.0, 1.0) if modality is Modality.IMAGE else None,
                )
            )
            start = index
    return tuple(spans)


def _simple_full(*, mandatory: tuple[int, ...] = ()) -> FullKVState:
    key = np.arange(1 * 2 * 7 * 4, dtype=np.float32).reshape(1, 2, 7, 4)
    value = (key + 1000).copy()
    return FullKVState.from_tensors(
        ((key, value),),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(
                2,
                5,
                Modality.IMAGE,
                image_index=0,
                page_index=1,
                region=(0.0, 0.0, 10.0, 10.0),
            ),
            ModalitySpan(5, 7, Modality.TEXT),
        ),
        token_ids=tuple(range(10, 17)),
        block_size=3,
        next_decode_position=9,
        mandatory_logical_positions=mandatory,
    )


def test_randomized_blockization_and_retention_one_properties() -> None:
    """Property sweep over random ranks, shapes, modalities, and block sizes."""

    rng = np.random.default_rng(202707)
    modality_values = tuple(Modality)
    for _case in range(128):
        layer_count = int(rng.integers(1, 5))
        sequence_length = int(rng.integers(1, 33))
        block_size = int(rng.integers(1, 13))
        batch = int(rng.integers(1, 3))
        layers = []
        expected_bytes = 0
        expected_memberships = 0
        for _layer in range(layer_count):
            heads = int(rng.integers(1, 6))
            key_width = int(rng.integers(1, 10))
            value_width = int(rng.integers(1, 10))
            key = rng.normal(size=(batch, heads, sequence_length, key_width)).astype(np.float32)
            value = rng.normal(size=(batch, heads, sequence_length, value_width)).astype(np.float32)
            layers.append((key, value))
            expected_bytes += key.nbytes + value.nbytes
            expected_memberships += heads * sequence_length
        modalities = [
            modality_values[int(index)]
            for index in rng.integers(0, len(modality_values), size=sequence_length)
        ]
        mandatory = tuple(
            index for index in range(sequence_length) if bool(rng.integers(0, 8) == 0)
        )
        full = FullKVState.from_tensors(
            tuple(layers),
            modality_spans=_spans(modalities),
            token_ids=tuple(int(value) for value in rng.integers(0, 32000, sequence_length)),
            block_size=block_size,
            next_decode_position=sequence_length + int(rng.integers(0, 4)),
            mandatory_logical_positions=mandatory,
        )

        assert full.active_bytes == expected_bytes
        assert sum(block.byte_size for block in full.blocks) == expected_bytes
        assert len(full.source_memberships) == expected_memberships
        assert all(1 <= block.position_count <= block_size for block in full.blocks)
        assert all(
            len({modalities[index] for index in block.physical_cache_indices}) == 1
            for block in full.blocks
        )
        assert all(
            list(block.original_logical_positions) == sorted(block.original_logical_positions)
            for block in full.blocks
        )
        for layer_index, layer in enumerate(full.layers):
            for head in range(layer.kv_heads):
                memberships = [
                    position
                    for block in full.blocks
                    if block.layer == layer_index and block.kv_head == head
                    for position in block.physical_cache_indices
                ]
                assert memberships == list(range(sequence_length))

        mosaic = MosaicKVState.retention_one(full)
        reconstructed = mosaic.reconstruct_full_state(full)
        assert mosaic.is_retention_one
        assert mosaic.statistics.byte_retention_ratio == 1.0
        assert mosaic.statistics.active_kv_bytes == expected_bytes
        assert reconstructed.original_logical_sequence_length == sequence_length
        assert reconstructed.next_decode_position == full.next_decode_position
        for source, candidate in zip(full.layers, reconstructed.layers, strict=True):
            assert np.array_equal(source.key, candidate.key)
            assert np.array_equal(source.value, candidate.value)


def test_selected_position_gather_preserves_values_and_logical_positions() -> None:
    full = _simple_full()
    exact = full.gather_selected_positions({(0, 0): (0, 2, 4, 6), (0, 1): (1, 3, 5)})
    assert exact.selected_positions(0, 0) == (0, 2, 4, 6)
    assert exact.selected_logical_positions(0, 1) == (1, 3, 5)
    assert exact.active_bytes == sum(
        tensor_storage_bytes(tensor) for tensor in (*exact.key_blocks, *exact.value_blocks)
    )
    for block, key_block, value_block in zip(
        exact.blocks, exact.key_blocks, exact.value_blocks, strict=True
    ):
        positions = list(block.physical_cache_indices)
        expected_key = full.layers[0].key[:, block.kv_head : block.kv_head + 1, positions, :]
        expected_value = full.layers[0].value[:, block.kv_head : block.kv_head + 1, positions, :]
        assert np.array_equal(key_block, expected_key)
        assert np.array_equal(value_block, expected_value)


def test_blockization_preserves_media_source_boundaries_and_metadata() -> None:
    key = np.zeros((1, 1, 6, 2), dtype=np.float32)
    value = np.ones_like(key)
    full = FullKVState.from_tensors(
        ((key, value),),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 4, Modality.IMAGE, image_index=0, page_index=2),
            ModalitySpan(4, 6, Modality.IMAGE, image_index=1, page_index=3),
        ),
        block_size=6,
    )

    assert [block.physical_cache_indices for block in full.blocks] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert full.blocks[1].image_index == 0
    assert full.blocks[1].page_index == 2
    assert full.blocks[2].image_index == 1
    assert full.blocks[2].page_index == 3
    assert not full.blocks[1].non_compressible


def test_source_partition_rejects_missing_or_duplicate_membership() -> None:
    full = _simple_full()
    with pytest.raises(ValueError, match="exactly one block"):
        replace(full, blocks=full.blocks[:-1])
    with pytest.raises(ValueError, match="duplicate source membership"):
        replace(full, blocks=(*full.blocks, full.blocks[0]))


def test_tier_memberships_are_disjoint_and_mandatory_blocks_remain_exact() -> None:
    full = _simple_full(mandatory=(0,))
    mandatory_blocks = tuple(block for block in full.blocks if block.mandatory)
    nonmandatory = tuple(block for block in full.blocks if not block.mandatory)
    exact = full.gather_exact_blocks(mandatory_blocks)
    assert mandatory_blocks
    MosaicKVState.create(full, exact=exact)
    with pytest.raises(ValueError, match="mandatory"):
        MosaicKVState.create(full, exact=full.gather_exact_blocks(nonmandatory))

    conflict_block = nonmandatory[0]
    conflict_exact = full.gather_exact_blocks((conflict_block, *mandatory_blocks))
    prototype = PrototypeTier(
        source_blocks=(conflict_block,),
        prototype_keys=(np.zeros((1, 1, 1, 4), dtype=np.float32),),
        prototype_values=(np.zeros((1, 1, 1, 4), dtype=np.float32),),
        assignments=(0,),
    )
    with pytest.raises(ValueError, match="memberships conflict"):
        MosaicKVState.create(full, exact=conflict_exact, prototypes=prototype)


def test_nonconflicting_prototype_and_residual_state_accounts_payload_storage() -> None:
    full = _simple_full()
    prototype_block, residual_block = full.blocks[:2]
    prototype = PrototypeTier(
        source_blocks=(prototype_block,),
        prototype_keys=(np.zeros((1, 1, 1, 4), dtype=np.float16),),
        prototype_values=(np.zeros((1, 1, 1, 4), dtype=np.float16),),
        assignments=(0,),
    )
    residual = ResidualTier(
        source_blocks=(residual_block,),
        key_residuals=(np.zeros((1, 1, 2), dtype=np.float32),),
        value_residuals=(np.zeros((1, 1, 2), dtype=np.float32),),
    )
    state = MosaicKVState.create(full, prototypes=prototype, residuals=residual)
    assert state.statistics.active_kv_bytes == prototype.active_bytes
    assert state.statistics.residual_kv_bytes == residual.active_bytes
    assert state.statistics.total_stored_bytes == prototype.active_bytes + residual.active_bytes
    assert state.statistics.prototype_source_blocks == 1
    assert state.statistics.residual_source_blocks == 1


def test_tier_storage_cannot_be_aliased_or_detached_from_source_metadata() -> None:
    full = _simple_full()
    exact = full.gather_exact_blocks((full.blocks[0],))
    prototype = PrototypeTier(
        source_blocks=(full.blocks[1],),
        prototype_keys=(exact.key_blocks[0],),
        prototype_values=(np.zeros((1, 1, 1, 4), dtype=np.float32),),
        assignments=(0,),
    )
    with pytest.raises(ValueError, match="same tensor object"):
        MosaicKVState.create(full, exact=exact, prototypes=prototype)

    changed = replace(full.blocks[0], modality=Modality.VIDEO)
    changed_exact = ExactTier((changed,), exact.key_blocks, exact.value_blocks)
    with pytest.raises(ValueError, match="modality differs"):
        MosaicKVState.create(full, exact=changed_exact)

    with pytest.raises(ValueError, match="require at least one source block"):
        PrototypeTier(
            prototype_keys=(np.zeros((1, 1), dtype=np.float32),),
            prototype_values=(np.zeros((1, 1), dtype=np.float32),),
        )


def test_descriptor_monotonicity_and_exact_byte_accounting_fail_closed() -> None:
    metadata = (MediaMetadata(2), MediaMetadata(1))
    with pytest.raises(ValueError, match="strictly monotonic"):
        KVBlockDescriptor(
            0,
            0,
            Modality.TEXT,
            (0, 1),
            (2, 1),
            (10, 11),
            metadata,
            "float32",
            "float32",
            "cpu",
            "cpu",
            16,
            False,
        )

    full = _simple_full()
    exact = full.gather_exact_blocks((full.blocks[0],))
    changed = replace(full.blocks[0], byte_size=full.blocks[0].byte_size + 1)
    with pytest.raises(ValueError, match="byte accounting mismatch"):
        ExactTier((changed,), exact.key_blocks, exact.value_blocks)


def test_position_map_and_compression_statistics_validation() -> None:
    position_map = LogicalPositionMap((1, 3, 5), 7, 9)
    assert position_map.gather((0, 2)) == (1, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        LogicalPositionMap((1, 1), 2, 2)
    with pytest.raises(ValueError, match="active KV bytes"):
        CompressionStatistics(100, 10, 10, 10, 31, 1, 1, 0, 0, 1, 1, 0.31)
