#!/usr/bin/env python3
"""CPU-only exact hash/graph feasibility, not pretrained-model latency evidence."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import GraphConfig
from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.graph import build_evidence_graph
from mosaickv.graph.pooling import pool_block_descriptors
from mosaickv.graph.structural_templates import activate, cache_key, engine_scope
from mosaickv.graph.types import BlockEvidenceMetadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = clean_source(Path(__file__).resolve().parents[2])
    if a.output.exists():
        raise FileExistsError("new immutable CPU-only feasibility output required")
    torch.set_num_threads(1)
    torch.manual_seed(20261107)
    pair = tuple(torch.randn(1, 1, 603, 8).bfloat16() for _ in range(2))
    state = FullKVState.from_tensors(
        (pair,), block_size=1, modality_spans=(ModalitySpan(0, 603, Modality.IMAGE),)
    )
    metadata = {
        i: BlockEvidenceMetadata(
            image_index=0,
            normalized_box=((i % 23) / 23, (i // 23) / 27, (i % 23 + 1) / 23, (i // 23 + 1) / 27),
            alignment_ids=("source",),
        )
        for i in range(603)
    }
    nodes = pool_block_descriptors(state, metadata_by_node=metadata)
    cfg = GraphConfig(max_neighbors=8, edge_accumulation_mode="segmented_tensor_topk")
    execution = ("indexed", "zero_overlap")
    expected = cache_key(nodes, cfg, None, execution, direct_metadata=False)
    times = {"reference": [], "direct": []}
    for step in range(55):
        order = (False, True) if step % 2 == 0 else (True, False)
        for direct in order:
            start = perf_counter()
            key = cache_key(nodes, cfg, None, execution, direct_metadata=direct)
            elapsed = perf_counter() - start
            if key != expected:
                raise ValueError("direct metadata serialization changed SHA256")
            if step >= 5:
                times["direct" if direct else "reference"].append(elapsed)
    with engine_scope() as bank:
        with activate(True):
            reference = build_evidence_graph(state, cfg, metadata_by_node=metadata)
        with activate(True, direct_metadata=True):
            direct = build_evidence_graph(state, cfg, metadata_by_node=metadata)
        for field in ("row_indices", "column_indices", "weights", "edge_types", "diagnostics"):
            if getattr(reference, field) != getattr(direct, field):
                raise ValueError("direct metadata serialization changed complete graph")
        if bank.hits != 1 or bank.misses != 1:
            raise ValueError("identical structural key must reuse the same bank entry")
    ratio = statistics.median(times["direct"]) / statistics.median(times["reference"])
    a.output.mkdir(parents=True)
    write_once(
        a.output / "decision.json",
        dict(
            source_sha=source,
            measurement_type="synthetic_CPU_hash_microbenchmark_not_model_latency_or_quality",
            nodes=603,
            five_warmups_fifty_alternating_pairs=True,
            actual_paired_seconds=times,
            median_ratio=ratio,
            all_hashes_identical=True,
            complete_graph_identical=True,
            CPU_operator_gate_passed=ratio <= 0.8,
            preregistered_operator_maximum_ratio=0.8,
            model_weights_loaded=False,
            benchmark_labels_accessed=False,
            persistent_GPU_bytes=0,
            independent_quality_or_latency_superiority_established=False,
        ),
    )
    print(json.dumps(dict(operator_ratio=ratio, CPU_operator_gate_passed=ratio <= 0.8)), flush=True)


if __name__ == "__main__":
    main()
