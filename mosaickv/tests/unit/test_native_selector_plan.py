from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mosaickv.backends.native_selector_plan import (
    NativeSelectorPlanEntry,
    NativeSelectorPlanError,
    catalog_payload,
    fit_projection_to_competitor_budget,
    load_native_selector_plan_catalog,
    merge_native_selector_plan_catalogs,
    project_common_selector_to_request_blocks,
    retarget_request_wide_projection,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan


def _full_state() -> FullKVState:
    layers = tuple(
        (
            np.zeros((1, 1, 16, 2), dtype=np.float32),
            np.zeros((1, 1, 16, 2), dtype=np.float32),
        )
        for _ in range(2)
    )
    return FullKVState.from_tensors(
        layers,
        block_size=4,
        mandatory_logical_positions=(0, 15),
    )


def test_request_wide_projection_uses_votes_scores_and_mandatory_boundaries() -> None:
    full = _full_state()
    # Both layers retain request block 2; only one retains request block 1.
    selected = (
        (full.blocks[1], 0.2),
        (full.blocks[2], 0.5),
        (full.blocks[6], 0.5),
    )

    projection = project_common_selector_to_request_blocks(
        full,
        selected,
        method="mosaickv_exact",
        retention_ratio=0.75,
    )

    assert projection.selected_blocks == (0, 2, 3)
    assert projection.mandatory_blocks == (0, 3)
    assert projection.selected_position_votes == (0, 4, 8, 0)
    assert projection.selection.selected_positions == 12
    assert projection.to_json_object()["policy"].endswith("projection_v2")


def test_request_wide_projection_preserves_internal_mandatory_tokens() -> None:
    layers = tuple(
        (
            np.zeros((1, 1, 32, 2), dtype=np.float32),
            np.zeros((1, 1, 32, 2), dtype=np.float32),
        )
        for _ in range(2)
    )
    full = FullKVState.from_tensors(
        layers,
        block_size=4,
        mandatory_logical_positions=(0, 17, 31),
    )
    projection = project_common_selector_to_request_blocks(
        full,
        ((full.blocks[0], 1.0),),
        method="mosaickv_exact",
        retention_ratio=0.25,
    )
    assert projection.mandatory_blocks == (0, 4, 7)
    assert set(projection.mandatory_blocks).issubset(projection.selected_blocks)


def test_competitor_relative_budget_fills_largest_safe_block_capacity() -> None:
    layers = tuple(
        (
            np.zeros((1, 1, 32, 2), dtype=np.float32),
            np.zeros((1, 1, 32, 2), dtype=np.float32),
        )
        for _ in range(2)
    )
    full = FullKVState.from_tensors(
        layers,
        block_size=4,
        mandatory_logical_positions=(0, 31),
    )
    prompt = project_common_selector_to_request_blocks(
        full,
        tuple((block, 1.0) for block in full.blocks[:6]),
        method="prompt_attention_topk",
        retention_ratio=0.75,
    )
    candidate = project_common_selector_to_request_blocks(
        full,
        ((full.blocks[2], 2.0),),
        method="mosaickv_exact",
        retention_ratio=0.25,
        fallback_candidates=((full.blocks[4], 1.0),),
    )

    fitted = fit_projection_to_competitor_budget(
        candidate,
        prompt,
        minimum_reduction_fraction=0.30,
    )

    # Six prompt blocks permit at most floor(0.7 * 6) == four candidate blocks.
    assert len(fitted.selected_blocks) == 4
    assert fitted.selected_blocks == (0, 2, 4, 7)
    assert fitted.policy == "common_selector_competitor_relative_evidence_budget_v2"


def test_evidence_reservation_beats_universal_suffix_votes_inside_same_budget() -> None:
    layers = tuple(
        (
            np.zeros((1, 1, 32, 2), dtype=np.float32),
            np.zeros((1, 1, 32, 2), dtype=np.float32),
        )
        for _ in range(2)
    )
    full = FullKVState.from_tensors(
        layers,
        block_size=4,
        modality_spans=(
            ModalitySpan(0, 8, Modality.TEXT),
            ModalitySpan(8, 24, Modality.IMAGE),
            ModalitySpan(24, 32, Modality.TEXT),
        ),
        mandatory_logical_positions=(0, 31),
    )
    prompt = project_common_selector_to_request_blocks(
        full,
        tuple((block, 1.0) for block in full.blocks[:5]),
        method="prompt_attention_topk",
        retention_ratio=0.625,
    )
    candidate = project_common_selector_to_request_blocks(
        full,
        # Suffix block 6 has stronger cross-route support than image block 4.
        ((full.blocks[6], 10.0), (full.blocks[14], 10.0), (full.blocks[4], 1.0)),
        method="mosaickv_exact",
        retention_ratio=0.25,
    )
    fitted = fit_projection_to_competitor_budget(
        candidate,
        prompt,
        minimum_reduction_fraction=0.30,
        minimum_evidence_blocks=1,
        maximum_evidence_blocks=1,
    )

    assert len(fitted.selected_blocks) == 3
    assert fitted.selected_blocks == (0, 4, 7)
    assert 4 in fitted.evidence_blocks


def test_hash_bound_catalog_rejects_provenance_mismatch(tmp_path: Path) -> None:
    projection = project_common_selector_to_request_blocks(
        _full_state(),
        ((_full_state().blocks[0], 1.0),),
        method="prompt_attention_topk",
        retention_ratio=0.5,
    )
    entry = NativeSelectorPlanEntry(
        sample_id="mmstar:1",
        projection=projection,
        selector_planning_seconds=0.25,
        source_trace="/scratch/djy8hg/source.json",
        source_trace_sha256="a" * 64,
    )
    payload = catalog_payload(
        model_id="model",
        model_revision="revision",
        task="mmstar",
        dataset_revision="dataset-revision",
        method="prompt_attention_topk",
        entries=(entry,),
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    catalog = load_native_selector_plan_catalog(
        path,
        digest,
        model_id="model",
        model_revision="revision",
        task="mmstar",
        dataset_revision="dataset-revision",
        method="prompt_attention_topk",
    )
    assert catalog.entry("mmstar:1").selector_planning_seconds == 0.25
    with pytest.raises(NativeSelectorPlanError, match="method"):
        load_native_selector_plan_catalog(
            path,
            digest,
            model_id="model",
            model_revision="revision",
            task="mmstar",
            dataset_revision="dataset-revision",
            method="mosaickv_exact",
        )


def test_merge_catalogs_produces_disjoint_two_request_plan(tmp_path: Path) -> None:
    catalogs = []
    for index in range(2):
        projection = project_common_selector_to_request_blocks(
            _full_state(),
            ((_full_state().blocks[0], 1.0),),
            method="mosaickv_exact",
            retention_ratio=0.5 if index == 0 else 0.75,
        )
        entry = NativeSelectorPlanEntry(
            sample_id=f"textvqa:{index}",
            projection=projection,
            selector_planning_seconds=0.25,
            source_trace=f"/scratch/djy8hg/source-{index}.json",
            source_trace_sha256=str(index) * 64,
        )
        payload = catalog_payload(
            model_id="model",
            model_revision="revision",
            task="textvqa",
            dataset_revision="dataset-revision",
            method="mosaickv_exact",
            entries=(entry,),
        )
        path = tmp_path / f"plan-{index}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        catalogs.append(
            load_native_selector_plan_catalog(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                model_id="model",
                model_revision="revision",
                task="textvqa",
                dataset_revision="dataset-revision",
                method="mosaickv_exact",
            )
        )

    merged = merge_native_selector_plan_catalogs(catalogs)

    assert [entry["sample_id"] for entry in merged["entries"]] == [
        "textvqa:0",
        "textvqa:1",
    ]
    assert [entry["projection"]["retention_ratio"] for entry in merged["entries"]] == [
        0.5,
        0.75,
    ]


def test_retarget_projection_aggregates_fine_votes_before_budgeting() -> None:
    layers = tuple(
        (
            np.zeros((1, 1, 64, 2), dtype=np.float32),
            np.zeros((1, 1, 64, 2), dtype=np.float32),
        )
        for _ in range(2)
    )
    full = FullKVState.from_tensors(
        layers,
        block_size=4,
        mandatory_logical_positions=(0, 63),
    )
    # Fine blocks 8 and 9 both vote for serving block 2.  Their combined
    # evidence must beat the single vote in serving block 1.
    fine = project_common_selector_to_request_blocks(
        full,
        (
            (full.blocks[5], 0.9),
            (full.blocks[8], 0.4),
            (full.blocks[9], 0.4),
        ),
        method="mosaickv_exact",
        retention_ratio=0.5,
    )

    serving = retarget_request_wide_projection(
        fine,
        target_block_size=16,
        target_retention_ratio=0.75,
    )

    assert serving.block_size == 16
    assert serving.selected_blocks == (0, 2, 3)
    assert serving.selected_position_votes == (0, 4, 8, 0)
    assert serving.selection.selected_positions == 48


def test_retarget_projection_rejects_unaligned_or_finer_target() -> None:
    fine = project_common_selector_to_request_blocks(
        _full_state(),
        ((_full_state().blocks[0], 1.0),),
        method="mosaickv_exact",
        retention_ratio=0.5,
    )
    with pytest.raises(NativeSelectorPlanError, match="integer multiple"):
        retarget_request_wide_projection(
            fine,
            target_block_size=6,
            target_retention_ratio=0.5,
        )
