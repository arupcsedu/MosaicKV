"""Read-only post-run checks for the registered consumed precision ablations.

This is not an independent benchmark-quality or cross-method superiority test.
It never changes the producer, exceptions, fractions, or numerical thresholds.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

FIELDS = tuple(f"{name}_active_kv_storage_bytes" for name in ("initial", "final", "peak"))


def check(
    row: dict[str, Any],
    corrections: dict[str, Any],
    ceiling: dict[str, Any],
    quality: dict[int, dict[str, float]],
) -> dict[str, Any]:
    control = row["controls"]["3"]
    reference = row["reference"]["storage"]
    actual = control["candidate"]["storage"]
    layers = corrections["layers"]
    if not layers or len({r["layer"] for r in layers}) != len(layers):
        raise ValueError("unique nonempty correction-layer records required")
    funded = all(
        0 <= r["exception_bytes"] <= r["saved_text_and_map_bytes"]
        and r["corrected_component_bytes"] <= r["raw_text_equivalent_bytes"]
        and r["fraction"] == 0.005
        and all(c["squared_error_after"] <= c["squared_error_before"] for c in r["components"])
        for r in layers
    )
    ratios = []
    for trial in range(20):
        pairs = [t for t in row["trials"] if not t["warmup"] and t["trial"] == trial]
        if len(pairs) != 2 or {t["variant"] for t in pairs} != {"reference", "3"}:
            raise ValueError("all 20 randomized paired repetitions required")
        times = {t["variant"]: t["decode_wall_seconds"] for t in pairs}
        if any(not math.isfinite(t) or t <= 0 for t in times.values()):
            raise ValueError("finite positive paired times required")
        ratios.append(times["3"] / times["reference"])
    values = np.asarray(ratios)
    boot = np.median(values[np.random.default_rng(20261030).integers(0, 20, (10000, 20))], axis=1)
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    differences = {}
    for cap in (32, 64, 128):
        pair = quality[cap]
        if set(pair) != {"reference", "3"} or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in pair.values()
        ):
            raise ValueError("both EOS-aware diagnostic scores required for every prefix")
        differences[str(cap)] = pair["3"] - pair["reference"]
    passed = dict(
        funded=funded,
        below_raw_text_precision_storage=all(actual[k] <= ceiling[k] for k in FIELDS),
        smaller_than_own_lossless_source=all(actual[k] < reference[k] for k in FIELDS),
        clone_and_codec_bitwise=control["clone_repack_bitwise"]
        and control["lossless_exact_codec_128_bitwise"],
        unchanged_registered_32_token_guard=control["numerical_gate"],
        nonnegative_diagnostic_reference_rouge_all_prefixes=all(
            v >= 0 for v in differences.values()
        ),
        paired_decode_ratio_upper_at_most_one=ci[1] <= 1,
    )
    return dict(
        checks=passed,
        diagnostic_all_checks_passed=all(passed.values()),
        memory_bytes=actual,
        own_source_reference_bytes=reference,
        funded_exception_bytes=sum(r["exception_bytes"] for r in layers),
        rouge_l_delta_from_own_source_by_prefix=differences,
        paired_decode_ratio_p50=float(np.median(values)),
        paired_decode_ratio_technical_ci95=ci,
    )


def derive(root: Path, output: Path, source: Path) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    if output.exists():
        raise ValueError("new write-once derived directory required")
    plan = json.loads((root / "plan.json").read_text())
    observations = root / "analysis/quality_prefixes/observations.parquet"
    manifest = root / "analysis/quality_prefixes/manifest.json"
    quality_manifest = json.loads(manifest.read_text())
    if quality_manifest["outputs"][observations.name] != sha(observations):
        raise ValueError("quality observations changed after derivation")
    inputs = {str(root / "plan.json"): sha(root / "plan.json"), str(manifest): sha(manifest)}
    inputs.update(quality_manifest["inputs"])
    inputs[str(observations)] = sha(observations)
    frame = pd.read_parquet(observations)
    records = []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        if (
            result["status"] != "completed"
            or result["failures"]
            or result["source_sha"] != plan["source_sha"]
            or result["plan_sha256"] != sha(root / "plan.json")
        ):
            raise ValueError("complete source/plan-bound producer required")
        inputs[str(result_path)] = sha(result_path)
        inputs.update({str(folder / name): digest for name, digest in result["outputs"].items()})
        selected = frame.loc[
            frame.task.eq(job["task"])
            & frame.sample_id.eq(job["sample_id"])
            & frame.correction_mode.eq(job["correction_mode"])
        ]
        scores = {}
        for cap in (32, 64, 128):
            subset = selected.loc[selected.prefix_token_cap.eq(cap)]
            if len(subset) != 2 or not subset.execution_valid.all():
                raise ValueError("two transport-valid rows required per prefix and mode")
            scores[cap] = dict(zip(subset.variant, subset.rouge_l_fullkv_reference, strict=True))
        outcome = check(
            json.loads((folder / "precision_controls.json").read_text()),
            json.loads((folder / "precision_3_corrections.json").read_text()),
            job["uncorrected_raw_text_storage"],
            scores,
        )
        records.append(dict(task=job["task"], mode=job["correction_mode"], **outcome))
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("bound producer or reference evidence changed")
    output.mkdir(parents=True)
    write_once(
        output / "decision.json",
        dict(
            source_sha=clean_source(source),
            inputs=inputs,
            records=records,
            measurement_type="consumed_two_input_posthoc_diagnostic_not_independent_quality",
            whole_request_peak_certified=False,
            superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    derive(args.root, args.output, Path(__file__).resolve().parents[4])
