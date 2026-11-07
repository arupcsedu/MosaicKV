#!/usr/bin/env python3
"""Prepare the V5 layer-capacity canary/calibration without reading outcomes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import prepare_output_preserving_route_logit_validation as shared

_v4_candidate = shared._candidate


def _candidate(base):
    candidate = _v4_candidate(base)
    assert candidate.output_preserving is not None
    return replace(
        candidate,
        output_preserving=replace(
            candidate.output_preserving,
            logit_candidate_strategy="layer_capacity_beam",
            logit_route_single_candidates=8,
            logit_route_candidate_limit=9,
            logit_route_max_combination_width=2,
            logit_probe_batch_size=8,
            logit_stability_horizon=2,
            logit_extended_horizon=8,
            logit_extended_candidate_limit=3,
            logit_layer_transfer_limit=8,
        ),
    )


def _candidate_metadata(candidate) -> dict[str, object]:
    options = candidate.output_preserving
    assert options is not None
    return {
        "effective_method": options.effective_method,
        "cap_mib": 25.5,
        "draft_ranking_mode": "joint_normalized",
        "short_horizon_logit_reranking": True,
        "logit_candidate_strategy": "layer_capacity_beam",
        "logit_route_candidate_limit": 9,
        "logit_stability_horizon": 2,
        "logit_extended_horizon": 8,
        "logit_extended_candidate_limit": 3,
        "logit_layer_transfer_limit": 8,
        "maximum_fail_closed_fraction": 0.20,
    }


def prepare(args: argparse.Namespace) -> None:
    shared.STAGES = {"canary": (16, 20271019), "calibration": (128, 20271020)}
    shared.PROTOCOL_VERSION = "output_preserving_layer_capacity_validation_v5"
    shared.PROTOCOL_FILENAME = "output_preserving_layer_capacity_validation_v5.yaml"
    shared.COHORT_NAMESPACE = "output_preserving_layer_capacity_v5"
    shared.CANDIDATE_ANALYSIS_LABEL = "output_preserving_layer_capacity"
    shared.CANDIDATE_JOB_LABEL = "output_preserving_layer_capacity_25p5"
    shared._candidate = _candidate
    shared._candidate_metadata = _candidate_metadata
    shared.ALLOWED_OVERLAY_PATHS = shared.ALLOWED_OVERLAY_PATHS | {
        "mosaickv/src/mosaickv/adapters/huggingface/layerwise_exact_cache.py",
        "mosaickv/scripts/prepare_output_preserving_layer_capacity_validation.py",
        "mosaickv/configs/experiments/output_preserving_layer_capacity_validation_v5.yaml",
        "mosaickv/docs/output_preserving_route_logit_v4_16_development_seal.md",
    }
    shared.prepare(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--pool-sha256", required=True)
    parser.add_argument("--excluded-plan", type=Path, action="append", required=True)
    parser.add_argument("--stage", choices=("canary", "calibration"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path)
    parser.add_argument("--canary-plan", type=Path)
    prepare(parser.parse_args())
