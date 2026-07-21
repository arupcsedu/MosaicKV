"""Sparse cross-modal evidence graphs over MosaicKV cache blocks."""

from mosaickv.graph.builder import build_evidence_graph
from mosaickv.graph.pooling import pool_block_descriptors, pool_prompt_attention_coactivation
from mosaickv.graph.types import (
    BlockEvidenceMetadata,
    EdgeType,
    GraphDiagnostics,
    PooledBlockDescriptor,
    SparseCSRGraph,
    SparseEvidenceGraph,
)

__all__ = [
    "BlockEvidenceMetadata",
    "EdgeType",
    "GraphDiagnostics",
    "PooledBlockDescriptor",
    "SparseCSRGraph",
    "SparseEvidenceGraph",
    "build_evidence_graph",
    "pool_block_descriptors",
    "pool_prompt_attention_coactivation",
]
