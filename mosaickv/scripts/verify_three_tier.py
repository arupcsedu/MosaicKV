#!/usr/bin/env python3
"""No-download CUDA smoke for prototype and pinned residual construction."""

from __future__ import annotations

import json

import torch

from mosaickv.adapters.huggingface import (
    InternVL25Adapter,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Qwen25VLAdapter,
)
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import (
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    UtilityConfig,
)
from mosaickv.graph import EdgeType, GraphDiagnostics, SparseEvidenceGraph
from mosaickv.graph.pooling import pool_block_descriptors
from mosaickv.prototypes import TierConstructionMode, construct_three_tier_cache
from mosaickv.repair import RepairCacheState, repair_decode_step
from mosaickv.residual import build_residual_storage, restore_residual_payload
from mosaickv.selection import (
    BudgetedObjective,
    SelectionBudget,
    SelectionResult,
    compute_block_utilities,
    lazy_greedy_select,
)
from mosaickv.types import BudgetUnit, RepairPolicy, ResidualStorageDType


def _graph(full: FullKVState) -> SparseEvidenceGraph:
    nodes = pool_block_descriptors(full)
    edges = ((1, 0, 0.25), (2, 0, 0.75))
    return SparseEvidenceGraph(
        nodes=nodes,
        row_indices=tuple(edge[0] for edge in edges),
        column_indices=tuple(edge[1] for edge in edges),
        weights=tuple(edge[2] for edge in edges),
        edge_types=(EdgeType.SEMANTIC_SIMILARITY,) * len(edges),
        diagnostics=GraphDiagnostics(
            node_count=len(nodes),
            edge_count=len(edges),
            connected_components=1,
            modality_mixing=0.0,
            average_degree=len(edges) / len(nodes),
            evidence_cluster_coverage=None,
            edge_counts=((EdgeType.SEMANTIC_SIMILARITY, len(edges)),),
            maximum_out_degree=1,
            fallback_used=False,
        ),
    )


def _selection(
    graph: SparseEvidenceGraph, budget: int, probabilities: tuple[float, ...]
) -> SelectionResult:
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=probabilities,
        attention_provenance="cuda_validation_smoke",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )
    return lazy_greedy_select(objective, SelectionBudget(budget, BudgetUnit.BLOCKS))


def _safe_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic_rope_free",
        architectures=("SyntheticVLM",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=True,
        supports_residual_repair=True,
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the three-tier GPU smoke")
    device = torch.device("cuda:0")
    key = torch.tensor([10.0, 20.0, 30.0], device=device, dtype=torch.float16).reshape(1, 1, 3, 1)
    value = torch.tensor([1.0, 3.0, 7.0], device=device, dtype=torch.float16).reshape(1, 1, 3, 1)
    full = FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = _graph(full)
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))
    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(group_size=2),
        residual_config=ResidualConfig(require_pinned_memory=True),
    )
    if result.mode is not TierConstructionMode.THREE_TIER:
        raise AssertionError(f"expected a three-tier result, received {result.mode.value}")
    prototype_key = result.state.prototypes.prototype_keys[0]
    prototype_value = result.state.prototypes.prototype_values[0]
    if prototype_key.device.type != "cuda" or prototype_value.device.type != "cuda":
        raise AssertionError("active prototypes did not remain on the source CUDA device")
    if float(prototype_key.item()) != 27.5 or float(prototype_value.item()) != 6.0:
        raise AssertionError("CUDA weighted prototype differs from the manual fixture")
    if not result.residual_storage.all_payloads_pinned:
        raise AssertionError("lossless residual payloads are not pinned")
    if result.residual_storage.lookup(0, 0, 0, 2).payload_index != 1:
        raise AssertionError("residual original-position lookup returned the wrong payload")
    if any(tensor.device.type != "cpu" for tensor in result.state.residuals.key_residuals):
        raise AssertionError("residual K payload is not on CPU")
    if any(tensor.device.type != "cpu" for tensor in result.state.residuals.value_residuals):
        raise AssertionError("residual V payload is not on CPU")

    repair_state = RepairCacheState.from_construction(full, result)
    re_decode_calls = 0

    def re_decode(repaired: RepairCacheState) -> torch.Tensor:
        nonlocal re_decode_calls
        re_decode_calls += 1
        if repaired.promoted_node_ids != (2,):
            raise AssertionError("repair did not promote the highest-risk residual block")
        return torch.tensor([1.0, 4.0], device=device, dtype=torch.float32)

    repair = repair_decode_step(
        repair_state,
        RepairConfig(
            policy=RepairPolicy.PROTOTYPE_RISK,
            prototype_risk_threshold=0.1,
            max_blocks_per_step=1,
        ),
        step_index=0,
        provisional_logits=torch.tensor([4.0, 1.0], device=device, dtype=torch.float32),
        prototype_attention_mass={0: 0.9},
        re_decode=re_decode,
    )
    if re_decode_calls != 1 or repair.event.re_decode_count != 1:
        raise AssertionError("repair did not re-decode the current step exactly once")
    if not repair.event.transfer_was_asynchronous:
        raise AssertionError("pinned residual promotion did not use an asynchronous CUDA copy")
    if repair.state.active_cost > repair.state.active_budget.value:
        raise AssertionError("repair exceeded the active cache budget")

    storage_formats = [
        ResidualStorageDType.LOSSLESS,
        ResidualStorageDType.FP16,
        ResidualStorageDType.BF16,
        ResidualStorageDType.INT8,
    ]
    if hasattr(torch, "float8_e4m3fn"):
        storage_formats.append(ResidualStorageDType.FP8)
    expected = full.gather_exact_blocks((full.blocks[1],))
    for storage_dtype in storage_formats:
        encoded = build_residual_storage(
            full,
            {1: 0},
            ResidualConfig(
                storage_dtype=storage_dtype,
                require_pinned_memory=True,
            ),
        )
        if not encoded.all_payloads_pinned:
            raise AssertionError(f"{storage_dtype.value} residual payload is not pinned")
        restored_key, restored_value = restore_residual_payload(encoded, 0, full)
        if storage_dtype is ResidualStorageDType.LOSSLESS and (
            not torch.equal(restored_key, expected.key_blocks[0])
            or not torch.equal(restored_value, expected.value_blocks[0])
        ):
            raise AssertionError("lossless residual restoration changed source values")
        if not torch.allclose(restored_key, expected.key_blocks[0], atol=0.25, rtol=0):
            raise AssertionError(f"{storage_dtype.value} residual K restoration failed")
        if not torch.allclose(restored_value, expected.value_blocks[0], atol=0.25, rtol=0):
            raise AssertionError(f"{storage_dtype.value} residual V restoration failed")

    post_rope_full = FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        cached_key_state=CachedKeyState.POST_ROPE,
    )
    post_rope_graph = _graph(post_rope_full)
    post_rope_selection = _selection(post_rope_graph, 1, (0.98, 0.01, 0.01))
    adapters = (Llava15Adapter, Qwen25VLAdapter, LlavaOneVisionAdapter, InternVL25Adapter)
    for adapter in adapters:
        fallback = construct_three_tier_cache(
            post_rope_full,
            post_rope_graph,
            post_rope_selection,
            adapter.capabilities,
        )
        if fallback.mode is not TierConstructionMode.EXACT_ONLY_UNSAFE:
            raise AssertionError(f"{adapter.__name__} did not take the exact-only safety path")

    retention_selection = _selection(graph, 3, (1.0, 1.0, 1.0))
    retention = construct_three_tier_cache(
        full,
        graph,
        retention_selection,
        _safe_capabilities(),
        retention_ratio=1.0,
    )
    if (
        retention.mode is not TierConstructionMode.RETENTION_ONE
        or not retention.state.is_retention_one
    ):
        raise AssertionError("retention one did not bypass all transformed tiers")

    print(
        json.dumps(
            {
                "measurement_type": "validation_smoke",
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "prototype_device": str(prototype_key.device),
                "prototype_dtype": str(prototype_key.dtype),
                "residual_payloads": len(result.residual_storage.payloads),
                "residual_storage_formats": [item.value for item in storage_formats],
                "all_residuals_pinned": result.residual_storage.all_payloads_pinned,
                "adapter_exact_only_count": len(adapters),
                "repair_async_transfer": repair.event.transfer_was_asynchronous,
                "repair_redecode_count": repair.event.re_decode_count,
                "repair_restored_blocks": list(repair.event.restored_block_ids),
                "retention_one": retention.state.is_retention_one,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
