from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import GraphConfig
from mosaickv.graph import build_evidence_graph, builder
from mosaickv.graph.structural_templates import (
    TemplateBank,
    activate,
    cache_key,
    current,
    engine_scope,
)
from mosaickv.graph.types import BlockEvidenceMetadata, EdgeType


def fixture(seed=14):
    g = torch.Generator().manual_seed(seed)
    pair = tuple(torch.randn(1, 2, 30, 8, generator=g).bfloat16() for _ in range(2))
    state = FullKVState.from_tensors(
        (pair,),
        block_size=1,
        modality_spans=(
            ModalitySpan(0, 4, Modality.TEXT),
            ModalitySpan(4, 28, Modality.IMAGE, image_index=0),
            ModalitySpan(28, 30, Modality.TEXT),
        ),
    )
    metadata = {
        i: BlockEvidenceMetadata(
            image_index=0,
            normalized_box=(0.1, 0.1, 0.2, 0.2),
            frame_index=i % 3,
            row_index=i % 2,
            evidence_region=f"group{i % 3}",
        )
        for i in range(len(state.blocks))
    }
    cfg = GraphConfig(max_neighbors=3, edge_accumulation_mode="segmented_tensor_topk")
    return state, metadata, cfg


def equal_graph(a, b):
    for key in ("row_indices", "column_indices", "weights", "edge_types", "diagnostics"):
        assert getattr(a, key) == getattr(b, key)


@pytest.mark.parametrize("seed", range(5))
def test_cached_structural_edges_are_exact_but_dynamic_semantics_are_recomputed(seed):
    state, metadata, cfg = fixture(seed)
    other, _, _ = fixture(seed + 50)
    a = build_evidence_graph(state, cfg, metadata_by_node=metadata)
    b = build_evidence_graph(other, cfg, metadata_by_node=metadata)
    with engine_scope() as bank, activate(True):
        with patch.object(
            builder, "_add_structural_edges", wraps=builder._add_structural_edges
        ) as f:
            cached_a = build_evidence_graph(state, cfg, metadata_by_node=metadata)
            cached_b = build_evidence_graph(other, cfg, metadata_by_node=metadata)
            cached_a2 = build_evidence_graph(state, cfg, metadata_by_node=metadata)
            assert f.call_count == 1
        equal_graph(a, cached_a)
        equal_graph(b, cached_b)
        equal_graph(a, cached_a2)
        assert bank.hits == 2 and bank.misses == 1
        assert bank.resident_bytes <= bank.maximum_bytes
    assert not bank.entries and current() is None


@pytest.mark.parametrize("changed", ("position", "modality", "metadata", "config", "eligibility"))
def test_every_structural_dependency_invalidates_the_template(changed):
    state, metadata, cfg = fixture()
    args = dict(metadata_by_node=metadata)
    if changed == "eligibility":
        cfg = replace(
            cfg,
            candidate_graph_mode="shortlist_rerank",
            similarity_shortlist_multiplier=1.0,
            candidate_descriptor_pooling=True,
        )
        args["similarity_candidate_node_ids"] = [0]
    with engine_scope() as bank, activate(True):
        build_evidence_graph(state, cfg, **args)
        if changed == "position":
            state = FullKVState.from_tensors(
                tuple((layer.key, layer.value) for layer in state.layers),
                block_size=1,
                logical_positions=tuple(range(100, 130)),
                original_logical_sequence_length=130,
                next_decode_position=130,
                modality_spans=tuple(
                    replace(span, start=span.start + 100, end=span.end + 100)
                    for span in state.modality_spans
                ),
            )
        if changed == "modality":
            blocks = list(state.blocks)
            blocks[0] = replace(blocks[0], modality=Modality.IMAGE)
            state = replace(state, blocks=tuple(blocks))
        if changed == "metadata":
            args["metadata_by_node"] = dict(metadata, **{})
            args["metadata_by_node"][0] = replace(metadata[0], frame_index=17)
        if changed == "config":
            cfg = replace(cfg, spatial_weight=0.5)
        if changed == "eligibility":
            args["similarity_candidate_node_ids"] = [1]
        build_evidence_graph(state, cfg, **args)
        assert bank.misses == 2


def test_bounded_cache_evicts_without_retaining_model_or_kv_owners():
    bank = TemplateBank(2048)
    packet = ((0, 1, 2), (1, 2, 3), (0.7, 1.4, 0.5), (EdgeType.SPATIAL_ADJACENCY,) * 3)
    for i in range(20):
        bank.put(str(i).zfill(64), packet, 4, 1)
        assert bank.resident_bytes <= 2048
    assert bank.evictions > 0
    assert bank.get("0".zfill(64)) is None
    last = bank.get("19".zfill(64))
    assert last[0][2].tolist() == [0.7, 1.4, 0.5]  # preweighted >1 is not clipped
    assert not last[0][0].flags.writeable
    assert bank.report()["retained_KV_or_query_tensors"] == 0
    bank.close()
    assert not bank.entries


def test_dynamic_edges_and_malformed_packets_cannot_be_cached():
    bank = TemplateBank()
    with pytest.raises(ValueError, match="dynamic"):
        bank.put("key", ((0,), (1,), (1.0,), (EdgeType.ATTENTION_COACTIVATION,)), 3, 1)
    with pytest.raises(ValueError, match="bounded-degree"):
        bank.put("key", ((0,), (0,), (1.0,), (EdgeType.SPATIAL_ADJACENCY,)), 3, 1)
    with pytest.raises(ValueError, match="engine scope"), activate(True):
        pass
    with engine_scope(), pytest.raises(RuntimeError, match="cannot nest"), engine_scope():
        pass


def test_oversized_entry_is_not_retained():
    bank = TemplateBank(1024)
    n = 100
    edges = (tuple(range(n)), tuple(range(1, n + 1)), (1.0,) * n, (EdgeType.SPATIAL_ADJACENCY,) * n)
    bank.put("key", edges, n + 1, 1)
    assert not bank.entries and bank.skipped == 1


@pytest.mark.parametrize("seed", range(7))
def test_direct_metadata_key_matches_generic_deepcopy_for_every_field(seed):
    state, metadata, cfg = fixture(seed)
    rich = BlockEvidenceMetadata(
        image_index=seed,
        frame_index=2,
        page_index=3,
        clip_index=4,
        normalized_box=(0.1, 0.2, 0.8, 0.9),
        ocr_text="A\u03a9\nquestion",
        row_index=6,
        column_index=7,
        page_region="page",
        evidence_region="region",
        alignment_ids=("a", "b"),
    )
    graph = build_evidence_graph(state, cfg, metadata_by_node=metadata)
    nodes = [replace(n, evidence=rich) for n in graph.nodes]

    def key(items, direct):
        return cache_key(items, cfg, {0, 1}, ("indexed", "zero_overlap"), direct_metadata=direct)

    base = key(nodes, False)
    assert key(nodes, True) == base
    assert len(base) == 64
    for f in fields(BlockEvidenceMetadata):
        changed = list(nodes)
        value = getattr(rich, f.name)
        new = None if value is not None and f.name != "alignment_ids" else ("other",)
        changed[0] = replace(nodes[0], evidence=replace(rich, **{f.name: new}))
        assert key(changed, True) == key(changed, False)
        assert key(changed, True) != base


def test_direct_policy_preserves_complete_graph_and_restores_context():
    state, metadata, cfg = fixture()
    reference = build_evidence_graph(state, cfg, metadata_by_node=metadata)
    with engine_scope() as bank:
        with activate(True, direct_metadata=True):
            direct = build_evidence_graph(state, cfg, metadata_by_node=metadata)
        with activate(True):
            legacy = build_evidence_graph(state, cfg, metadata_by_node=metadata)
        assert bank.hits == 1 and bank.misses == 1
        equal_graph(reference, direct)
        equal_graph(reference, legacy)
    with pytest.raises(ValueError), activate(False, direct_metadata=True):
        pass


def test_direct_key_is_explicitly_gated_by_overlay_policy():
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    with pytest.raises(ValueError, match="require structural"):
        OverlayPolicy(graph_direct_metadata_key=True)
    assert OverlayPolicy(graph_structural_templates=True, graph_direct_metadata_key=True)
