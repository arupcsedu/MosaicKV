#!/usr/bin/env python3
"""Build a hashable vLLM selector catalog from common HF runtime traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from mosaickv.backends.native_selector_plan import (
    NativeSelectorPlanEntry,
    RequestWideSelectorProjection,
    catalog_payload,
    fit_projection_to_competitor_budget,
    load_native_selector_plan_catalog,
    retarget_request_wide_projection,
)
from mosaickv.types import JsonObject


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast("dict[str, Any]", value)


def _projection(value: object) -> RequestWideSelectorProjection:
    payload = _object(value, "request_wide_native_projection")
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported projection schema")
    return RequestWideSelectorProjection(
        method=str(payload["method"]),
        logical_positions=int(payload["logical_positions"]),
        block_size=int(payload["block_size"]),
        retention_ratio=float(payload["retention_ratio"]),
        selected_blocks=tuple(int(value) for value in payload["selected_blocks"]),
        mandatory_blocks=tuple(int(value) for value in payload["mandatory_blocks"]),
        selected_position_votes=tuple(
            int(value) for value in payload["selected_position_votes"]
        ),
        selected_score_sums=tuple(float(value) for value in payload["selected_score_sums"]),
        source_selected_positions=int(payload["source_selected_positions"]),
        source_total_positions=int(payload["source_total_positions"]),
        selected_route_votes=tuple(
            int(value) for value in payload.get("selected_route_votes", [])
        ),
        normalized_utility_scores=tuple(
            float(value) for value in payload.get("normalized_utility_scores", [])
        ),
        evidence_route_votes=tuple(
            int(value) for value in payload.get("evidence_route_votes", [])
        ),
        evidence_utility_scores=tuple(
            float(value) for value in payload.get("evidence_utility_scores", [])
        ),
        evidence_blocks=tuple(int(value) for value in payload.get("evidence_blocks", [])),
        policy=str(payload["policy"]),
    )


def _planning_seconds(trace: Mapping[str, object]) -> float:
    timing = _object(trace.get("timing_breakdown"), "timing_breakdown")
    keys = ("forecast", "graph", "utility_and_selection")
    values = tuple(float(timing.get(key, 0.0)) for key in keys)
    if any(value < 0 for value in values):
        raise ValueError("selector planning timings must be nonnegative")
    return sum(values)


def build_catalog(
    manifest_path: Path,
    trace_root: Path,
    *,
    task: str,
    target_block_size: int | None = None,
    target_retention_ratio: float | None = None,
    competitor_catalog: Path | None = None,
    competitor_catalog_sha256: str | None = None,
    minimum_reduction_fraction: float | None = None,
    minimum_competitor_anchor_blocks: int = 0,
    competitor_tail_dominance_threshold: float = 0.0,
    coherent_evidence_minimum_route_ratio: float = 0.0,
) -> JsonObject:
    manifest = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    model = _object(manifest.get("model"), "manifest.model")
    dataset = _object(manifest.get("dataset"), "manifest.dataset")
    config = _object(manifest.get("resolved_config"), "manifest.resolved_config")
    method = str(config.get("method", ""))
    traces = sorted(trace_root.glob("*.json"))
    if not traces:
        raise ValueError(f"no traces found under {trace_root}")
    entries: list[NativeSelectorPlanEntry] = []
    for path in traces:
        payload_bytes = path.read_bytes()
        trace = _object(json.loads(payload_bytes), str(path))
        trace_method = str(trace.get("method", ""))
        if trace_method != method:
            raise ValueError(
                f"trace method {trace_method!r} does not match manifest method {method!r}"
            )
        projection = _projection(trace.get("request_wide_native_projection"))
        if projection.method != method:
            raise ValueError("projection method differs from its trace")
        if target_block_size is not None or target_retention_ratio is not None:
            projection = retarget_request_wide_projection(
                projection,
                target_block_size=(
                    projection.block_size
                    if target_block_size is None
                    else target_block_size
                ),
                target_retention_ratio=(
                    projection.retention_ratio
                    if target_retention_ratio is None
                    else target_retention_ratio
                ),
            )
        entries.append(
            NativeSelectorPlanEntry(
                sample_id=str(trace["sample_id"]),
                projection=projection,
                selector_planning_seconds=_planning_seconds(trace),
                source_trace=str(path.resolve()),
                source_trace_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            )
        )
    if any(
        value is not None
        for value in (
            competitor_catalog,
            competitor_catalog_sha256,
            minimum_reduction_fraction,
        )
    ):
        if (
            competitor_catalog is None
            or competitor_catalog_sha256 is None
            or minimum_reduction_fraction is None
        ):
            raise ValueError(
                "competitor catalog, SHA-256, and minimum reduction are required together"
            )
        competitor = load_native_selector_plan_catalog(
            competitor_catalog,
            competitor_catalog_sha256,
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            task=task,
            dataset_revision=str(dataset["revision"]),
            method="prompt_attention_topk",
        )
        entries = [
            NativeSelectorPlanEntry(
                sample_id=entry.sample_id,
                projection=fit_projection_to_competitor_budget(
                    entry.projection,
                    competitor.entry(entry.sample_id).projection,
                    minimum_reduction_fraction=minimum_reduction_fraction,
                    minimum_competitor_anchor_blocks=(
                        minimum_competitor_anchor_blocks
                    ),
                    competitor_tail_dominance_threshold=(
                        competitor_tail_dominance_threshold
                    ),
                    coherent_evidence_minimum_route_ratio=(
                        coherent_evidence_minimum_route_ratio
                    ),
                ),
                selector_planning_seconds=entry.selector_planning_seconds,
                source_trace=entry.source_trace,
                source_trace_sha256=entry.source_trace_sha256,
            )
            for entry in entries
        ]
    return catalog_payload(
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
        task=task,
        dataset_revision=str(dataset["revision"]),
        method=method,
        entries=entries,
    )


def _write_once(path: Path, payload: JsonObject) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite selector plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-block-size", type=int)
    parser.add_argument("--target-retention-ratio", type=float)
    parser.add_argument("--competitor-catalog", type=Path)
    parser.add_argument("--competitor-catalog-sha256")
    parser.add_argument("--minimum-reduction-fraction", type=float)
    parser.add_argument("--minimum-competitor-anchor-blocks", type=int, default=0)
    parser.add_argument("--competitor-tail-dominance-threshold", type=float, default=0.0)
    parser.add_argument(
        "--coherent-evidence-minimum-route-ratio", type=float, default=0.0
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_once(
        args.output.resolve(),
        build_catalog(
            args.manifest.resolve(),
            args.trace_root.resolve(),
            task=args.task,
            target_block_size=args.target_block_size,
            target_retention_ratio=args.target_retention_ratio,
            competitor_catalog=(
                None
                if args.competitor_catalog is None
                else args.competitor_catalog.resolve()
            ),
            competitor_catalog_sha256=args.competitor_catalog_sha256,
            minimum_reduction_fraction=args.minimum_reduction_fraction,
            minimum_competitor_anchor_blocks=args.minimum_competitor_anchor_blocks,
            competitor_tail_dominance_threshold=(
                args.competitor_tail_dominance_threshold
            ),
            coherent_evidence_minimum_route_ratio=(
                args.coherent_evidence_minimum_route_ratio
            ),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
