#!/usr/bin/env python3
"""Export a CPU-only, fail-closed Fox review packet from registered capsules."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
from pathlib import Path

from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.evaluation.paper_readiness_v63 import (
    ANCHOR,
    AUTHORIZATION_KEYS,
    CERTIFICATES,
    CapsuleReader,
    build_decision,
)

SOURCE = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ANCHOR / "fox-review-v63"
ASSETS = {
    "protocol": "mosaickv/configs/experiments/evicastkv_fox/review_completion_v63.json",
    "baselines": "mosaickv/docs/evicastkv_fox/metric_baseline_readiness_v63.json",
    "claims": "mosaickv/docs/evicastkv_fox/review_claims_v63.json",
}


def prepare(output):
    revision = clean_source(SOURCE)
    output = Path(output)
    if (
        output.parent.parent.resolve() != OUTPUT_BASE.resolve()
        or output.parent.name != f"run-20261114-{revision[:7]}-1"
        or output.name != "packet"
        or output.parent.is_symlink()
        or not output.parent.is_dir()
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("new registered CPU packet output required")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    reader = CapsuleReader()
    result = None
    try:
        current_cpu = cpu_gate(output.parent / "cpu", revision)
        if current_cpu["counts"]["tests"] < 10761:
            raise ValueError("complete current-source CPU suite required")
        evidence = {key: reader.read(*spec) for key, spec in CERTIFICATES.items()}
        assets = {
            key: reader.read(str(SOURCE.relative_to(ANCHOR) / path), None)
            for key, path in ASSETS.items()
        }
        for path in ASSETS.values():
            expected_oid = subprocess.check_output(
                ["git", "rev-parse", "--verify", f"{revision}:{path}"],
                cwd=SOURCE,
                text=True,
                timeout=15,
            ).strip()
            binding = next(b for b in reader.bindings if b["path"] == str(SOURCE / path))
            if binding["computed_git_blob_oid"] != expected_oid:
                raise ValueError("source asset is not the registered committed Git blob")
        result = build_decision(evidence, **assets)
        result["current_source_CPU"] = current_cpu
        clean_source(SOURCE, revision)
        for key, value in assets.items():
            write_once(output / f"{key}.json", value)
    except (
        OSError,
        ValueError,
        TimeoutError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result = dict(
            schema="fox_paper_readiness_packet_v63",
            status="BLOCKED_EVIDENCE_INVALID",
            authorization={key: False for key in AUTHORIZATION_KEYS},
            error_type=type(error).__name__,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            paper_campaign_authorized=False,
            superiority_established=False,
            GPU_jobs=0,
            credential_fixture_read=False,
            completed_proofs_replayed=0,
        )
    result.update(
        source_sha=revision,
        input_bindings=reader.bindings,
        metadata_input_bytes=reader.input_bytes,
        metadata_nodes=reader.nodes,
        projected_metadata_bytes=reader.projected_bytes,
        raw_links_followed=0,
        limits_changed=False,
    )
    write_once(output / "readiness_decision.json", result)
    digest = hashlib.sha256((output / "readiness_decision.json").read_bytes()).hexdigest()
    print(json.dumps(dict(output=str(output), status=result["status"], sha256=digest)))
    return result["status"] != "BLOCKED_EVIDENCE_INVALID"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    signal.alarm(210)
    return 0 if prepare(args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
