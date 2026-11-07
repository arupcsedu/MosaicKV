#!/usr/bin/env python3
"""Prepare the V9 fixed-shape within-route development validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import prepare_output_preserving_route_logit_validation as shared
from prepare_output_preserving_directional_single_validation import (
    _candidate as _v8_candidate,
)


def _candidate(base):
    candidate = _v8_candidate(base)
    assert candidate.output_preserving is not None
    return replace(
        candidate,
        output_preserving=replace(
            candidate.output_preserving,
            logit_candidate_strategy="within_route_position_beam",
            logit_route_candidate_limit=16,
            logit_route_max_combination_width=1,
            logit_probe_batch_size=4,
            logit_extended_candidate_limit=2,
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
        "logit_candidate_strategy": options.logit_candidate_strategy,
        "transaction": "one_block_equal_byte_substitution_within_route",
        "route_lengths": "fixed_to_prompt_topk_anchor",
        "physical_padding_bytes": 0,
        "directional_ranking": ("normalized_prompt_draft_gain_minus_directional_perturbation"),
        "logit_route_candidate_limit": 16,
        "logit_stability_horizon": 2,
        "logit_extended_horizon": 8,
        "logit_extended_candidate_limit": 2,
        "model_probed_candidate_count": 4,
        "maximum_transaction_width": 1,
        "maximum_fail_closed_fraction": 0.20,
    }


def prepare(args: argparse.Namespace) -> None:
    shared.STAGES = {"canary": (16, 20271027), "calibration": (128, 20271028)}
    shared.PROTOCOL_VERSION = "output_preserving_within_route_validation_v9"
    shared.PROTOCOL_FILENAME = "output_preserving_within_route_validation_v9.yaml"
    shared.COHORT_NAMESPACE = "output_preserving_within_route_v9"
    shared.CANDIDATE_ANALYSIS_LABEL = "output_preserving_within_route"
    shared.CANDIDATE_JOB_LABEL = "output_preserving_within_route_25p5"
    shared._candidate = _candidate
    shared._candidate_metadata = _candidate_metadata
    shared.ALLOWED_OVERLAY_PATHS = shared.ALLOWED_OVERLAY_PATHS | {
        "mosaickv/src/mosaickv/adapters/huggingface/layerwise_exact_cache.py",
        "mosaickv/scripts/prepare_output_preserving_layer_capacity_validation.py",
        "mosaickv/scripts/prepare_output_preserving_surrogate_validation.py",
        "mosaickv/scripts/prepare_output_preserving_vectorized_surrogate_validation.py",
        "mosaickv/scripts/prepare_output_preserving_directional_single_validation.py",
        "mosaickv/scripts/prepare_output_preserving_within_route_validation.py",
        "mosaickv/scripts/prepare_output_preserving_probe_microbenchmark.py",
        "mosaickv/scripts/analyze_output_preserving_probe_microbenchmark.py",
        "mosaickv/scripts/submit_output_preserving_probe_microbenchmark.py",
        "mosaickv/configs/experiments/output_preserving_layer_capacity_validation_v5.yaml",
        "mosaickv/configs/experiments/output_preserving_surrogate_validation_v6.yaml",
        "mosaickv/configs/experiments/output_preserving_vectorized_surrogate_validation_v7.yaml",
        "mosaickv/configs/experiments/output_preserving_directional_single_validation_v8.yaml",
        "mosaickv/configs/experiments/output_preserving_within_route_validation_v9.yaml",
        "mosaickv/docs/output_preserving_route_logit_v4_16_development_seal.md",
        "mosaickv/docs/output_preserving_layer_capacity_v5_16_development_seal.md",
        "mosaickv/docs/output_preserving_surrogate_v6_timing_seal.md",
        "mosaickv/docs/output_preserving_vectorized_surrogate_v7_16_development_seal.md",
        "mosaickv/docs/output_preserving_directional_single_v8_16_development_seal.md",
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
