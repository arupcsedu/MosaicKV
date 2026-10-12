#!/usr/bin/env python3
"""Fail-closed reducer for sharded official PrefixKV profile reproductions.

The official PrefixKV class uses an upstream-distributed profile whose
calibration identities were not released.  Individual two-hour workers are
therefore intentionally not paper eligible.  This reducer admits an
official-profile reproduction only when the exact 1,000 registered
LLaVA-Description identities occur once, every worker is clean and pinned to
the same upstream profile, and all additive PPL statistics are present.

It does *not* turn the result into evidence that ``prefixkv_reimpl`` is
equivalent.  That separate claim requires a locally recorded, disjoint
calibration source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DATASET_ID = "PrefixKV/LLaVA-Description"
DATASET_REVISION = "407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a"
EXPECTED_SAMPLE_COUNT = 1000
DEFAULT_PPL_TARGET = 3.41
DEFAULT_PPL_TOLERANCE = 0.25


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any] | list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{label} {path} must contain an object or array")
    return payload


def _expected_ids(data_path: Path) -> set[str]:
    payload = _load_json(data_path, label="evaluation data")
    if not isinstance(payload, list) or len(payload) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"evaluation data must contain exactly {EXPECTED_SAMPLE_COUNT} registered records"
        )
    ids = [str(item.get("id")) for item in payload if isinstance(item, dict)]
    if len(ids) != EXPECTED_SAMPLE_COUNT or len(set(ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("evaluation data must have one unique id per registered record")
    return set(ids)


def aggregate_official_profile_shards(
    *,
    shard_root: Path,
    data_path: Path,
    official_profile_path: Path,
    published_ppl_target: float = DEFAULT_PPL_TARGET,
    published_ppl_tolerance: float = DEFAULT_PPL_TOLERANCE,
) -> dict[str, Any]:
    """Validate and aggregate a complete, duplicate-free official shard union."""

    expected_ids = _expected_ids(data_path)
    expected_profile_sha = _sha256_file(official_profile_path)
    manifest_paths = sorted(shard_root.glob("*/manifest.json"))
    if not manifest_paths:
        raise ValueError(f"no shard manifests found below {shard_root}")

    observations: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest_raw = _load_json(manifest_path, label="shard manifest")
        if not isinstance(manifest_raw, dict):
            raise ValueError(f"shard manifest must be an object: {manifest_path}")
        if bool(manifest_raw.get("git_dirty")):
            raise ValueError(f"dirty worktree in shard {manifest_path.parent.name}")
        protocol = manifest_raw.get("protocol")
        if not isinstance(protocol, dict) or not protocol.get("official_profile_reproduction"):
            raise ValueError(f"not an official-profile shard: {manifest_path.parent.name}")
        dataset = manifest_raw.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("id") != DATASET_ID:
            raise ValueError(f"wrong dataset in shard {manifest_path.parent.name}")
        if dataset.get("revision") != DATASET_REVISION:
            raise ValueError(f"wrong dataset revision in shard {manifest_path.parent.name}")
        profile = manifest_raw.get("profile")
        if not isinstance(profile, dict) or profile.get("source_sha256") != expected_profile_sha:
            raise ValueError(f"upstream profile hash mismatch in shard {manifest_path.parent.name}")
        if manifest_raw.get("paper_eligible"):
            raise ValueError(
                f"a partial shard cannot be paper eligible: {manifest_path.parent.name}"
            )
        if "incomplete_official_profile_shard" not in str(
            manifest_raw.get("paper_ineligibility_reason", "")
        ):
            raise ValueError(f"shard is not marked incomplete: {manifest_path.parent.name}")

        artifact_path = manifest_path.parent / "official.json"
        artifact_raw = _load_json(artifact_path, label="official PrefixKV artifact")
        if (
            not isinstance(artifact_raw, dict)
            or artifact_raw.get("implementation") != "official_prefixkv"
        ):
            raise ValueError(f"missing official PrefixKV artifact in {manifest_path.parent.name}")
        samples = artifact_raw.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"empty official PrefixKV shard {manifest_path.parent.name}")
        manifest_ids = {str(value) for value in dataset.get("evaluation_sample_ids", [])}
        artifact_ids = {str(row.get("sample_id")) for row in samples if isinstance(row, dict)}
        if manifest_ids != artifact_ids:
            raise ValueError(f"manifest/artifact sample mismatch in {manifest_path.parent.name}")
        for row in samples:
            if not isinstance(row, dict):
                raise ValueError(f"malformed sample row in {manifest_path.parent.name}")
            sample_id = str(row.get("sample_id"))
            if sample_id not in expected_ids:
                raise ValueError(f"unregistered sample id {sample_id}")
            if sample_id in observations:
                raise ValueError(f"duplicate sample id across shards: {sample_id}")
            nll_sum = row.get("nll_sum")
            token_count = row.get("evaluated_token_count")
            rouge_l = row.get("rouge_l_f1")
            if (
                not isinstance(nll_sum, (float, int))
                or not isinstance(token_count, int)
                or token_count < 1
                or not isinstance(rouge_l, (float, int))
                or not math.isfinite(float(nll_sum))
                or not math.isfinite(float(rouge_l))
            ):
                raise ValueError(f"incomplete metrics for sample {sample_id}")
            observations[sample_id] = row
        shard_records.append(
            {
                "run_id": manifest_raw.get("run_id"),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "artifact_sha256": _sha256_file(artifact_path),
                "sample_count": len(samples),
                "git_sha": manifest_raw.get("git_sha"),
            }
        )

    missing = sorted(expected_ids.difference(observations))
    if missing:
        raise ValueError(f"incomplete official reproduction: {len(missing)} sample IDs missing")
    if len(observations) != EXPECTED_SAMPLE_COUNT:
        raise AssertionError("validated union has unexpected sample count")

    rows = [observations[sample_id] for sample_id in sorted(observations)]
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    token_count = sum(int(row["evaluated_token_count"]) for row in rows)
    corpus_ppl = math.exp(nll_sum / token_count)
    mean_rouge_l = sum(float(row["rouge_l_f1"]) for row in rows) / len(rows)
    return {
        "schema_version": 1,
        "result_category": "paper_reproduction",
        "measurement_origin": "locally_reproduced",
        "implementation": "official_prefixkv",
        "official_profile_provenance": "unverifiable_upstream_profile_ids_not_released",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "file_sha256": _sha256_file(data_path),
            "sample_count": len(rows),
        },
        "profile": {
            "path": str(official_profile_path),
            "sha256": expected_profile_sha,
        },
        "metrics": {
            "teacher_forced_ppl": {
                "value": corpus_ppl,
                "nll_sum": nll_sum,
                "evaluated_token_count": token_count,
                "published_target": published_ppl_target,
                "absolute_delta": corpus_ppl - published_ppl_target,
                "tolerance": published_ppl_tolerance,
                "within_tolerance": abs(corpus_ppl - published_ppl_target)
                <= published_ppl_tolerance,
            },
            "rouge_l_f1": {
                "value": mean_rouge_l,
                "scope": "mean_per_sample_against_frozen_fullkv",
                "published_target": None,
                "comparison_status": "reported_without_unregistered_numeric_target",
            },
        },
        "shards": shard_records,
        "paper_eligible": abs(corpus_ppl - published_ppl_target) <= published_ppl_tolerance,
        "eligibility_scope": (
            "official PrefixKV code/profile and registered LLaVA-Description metrics only; "
            "not local reimplementation parity"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--official-profile-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-ppl-target", type=float, default=DEFAULT_PPL_TARGET)
    parser.add_argument("--published-ppl-tolerance", type=float, default=DEFAULT_PPL_TOLERANCE)
    args = parser.parse_args()
    if args.published_ppl_tolerance < 0:
        raise ValueError("--published-ppl-tolerance must be nonnegative")
    result = aggregate_official_profile_shards(
        shard_root=args.shard_root,
        data_path=args.data_path,
        official_profile_path=args.official_profile_path,
        published_ppl_target=args.published_ppl_target,
        published_ppl_tolerance=args.published_ppl_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
