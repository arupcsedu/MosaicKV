#!/usr/bin/env python3
"""Recheck existing mirror artifacts on CPU; never rerun inference or edit raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

RUN = runpy.run_path(str(Path(__file__).with_name("run_vllm_byte_capped_mirror.py")))


def analyze(index: Path, expected_sha: str) -> dict:
    entries = json.loads(RUN["bound"](str(index), expected_sha).read_text())["jobs"]
    lanes = []
    for entry in entries:
        job_path = RUN["bound"](entry["path"], entry["sha256"])
        job = json.loads(job_path.read_text())
        terminal = Path(job["output_root"]) / "validation.json"
        record = {
            "lane": entry["lane"], "job_path": str(job_path),
            "job_sha256": entry["sha256"], "terminal_path": str(terminal),
        }
        try:
            RUN["verify_source"](job)
            original = json.loads(terminal.read_text())
            if original["job_sha256"] != entry["sha256"] or original["status"] != "passed":
                raise ValueError("worker terminal record did not pass this immutable job")
            outputs = {
                arm["name"]: (
                    Path(job["output_root"]) / "outputs" / arm["name"],
                    f"vllm-bytecap-{entry['sha256'][:12]}-{arm['name']}",
                )
                for arm in job["arms"]
            }
            record.update(RUN["validate_outputs"](job, outputs))
            record["terminal_sha256"] = RUN["sha"](terminal)
            record["canonical_preflight"] = (
                job["classification"] == "canonical_vllm_projection"
                and job["stage"] == "preflight"
                and original.get("gate_binding") == job.get("gate_binding")
                and job.get("gate_binding") is not None
                and record["telemetry_checks_passed"]
            )
        except (ValueError, KeyError, OSError) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        lanes.append(record)
    return {
        "schema_version": 1,
        "index": str(index), "index_sha256": expected_sha,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validator_sha256": RUN["sha"](
            Path(__file__).with_name("run_vllm_byte_capped_mirror.py")
        ),
        "execution_checks_passed": bool(lanes) and all(
            lane["status"] == "passed" for lane in lanes
        ),
        "all_canonical_preflights": bool(lanes) and all(
            lane.get("canonical_preflight", False) for lane in lanes
        ),
        "telemetry_checks_passed": bool(lanes) and all(
            lane.get("telemetry_checks_passed", False) for lane in lanes
        ),
        "paper_quality_claim_established": False,
        "full_sweep_submitted": False,
        "lanes": lanes,
    }


def render(report: dict) -> str:
    lines = [
        "# vLLM byte-cap preflight audit", "",
        "This is a derived CPU audit of immutable GPU outputs. Execution success is not "
        "quality non-inferiority or paper admissibility. No GPU work is launched here.", "",
        "| Lane | n | Execution | FullKV score | Prompt Top-K | MosaicKV | Other baseline |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for lane in report["lanes"]:
        if lane["status"] != "passed":
            lines.append(f"| {lane['lane']} | — | failed: {lane['error']} | — | — | — | — |")
            continue
        scores = lane["mean_task_scores"]
        others = ", ".join(
            f"{name}: {100 * value:.2f}%" for name, value in scores.items()
            if name not in {"full_kv", "native_identity", "prompt_attention_topk", "mosaickv_exact"}
        ) or "—"
        full = f"{100 * scores['full_kv']:.2f}%" if "full_kv" in scores else "not repeated"
        lines.append(
            f"| {lane['lane']} | {lane['sample_count']} | passed | {full} | "
            f"{100 * scores['prompt_attention_topk']:.2f}% | "
            f"{100 * scores['mosaickv_exact']:.2f}% | {others} |"
        )
    lines += [
        "", "Scores are task-specific: MMStar accuracy, TextVQA VQA consensus, "
        "and DocVQA ANLS. They are not averaged across tasks.", "",
        "All methods with a serving projection are variants, not official implementations "
        "or identical HF per-head selectors. FullKV is excluded from latency comparisons.", "",
        f"Execution checks passed: `{report['execution_checks_passed']}`. "
        f"All canonical preflights: `{report['all_canonical_preflights']}`.", "",
        f"Native telemetry checks passed: `{report['telemetry_checks_passed']}`. "
        "A telemetry mismatch blocks campaign promotion even when execution checks pass.", "",
        "## Per-lane evidence", "",
    ]
    for lane in report["lanes"]:
        lines += [f"### {lane['lane']}", "", f"Artifact: `{lane['terminal_path']}`", ""]
        if lane["status"] != "passed":
            continue
        lines += [
            f"Retention-1 parity: `{lane['retention_1_parity']}`; "
            f"A-B-A isolation: `{lane['request_isolation']}`. "
            f"Classification: `{lane['classification']}`.", "",
            f"Telemetry discrepancies: {len(lane['telemetry_issues'])}; "
            "see audit.json for each sample's recorded and planned retention ratios.", "",
            "| Method | Active MiB (range) | Engine request s | Decode s | "
            "Offline planner + transaction s | Reconstructed total s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for method, timing in lane["compressed_timing"].items():
            sizes = [check["active_kv_bytes"][method] / 2**20 for check in lane["byte_checks"]]
            lines.append(
                f"| {method} | {min(sizes):.2f}-{max(sizes):.2f} | "
                f"{timing['engine_request_mean_seconds']:.4f} | "
                f"{timing['decode_mean_seconds']:.4f} | "
                f"{timing['offline_planner_plus_transaction_mean_seconds']:.4f} | "
                f"{timing['reconstructed_total_mean_seconds']:.4f} |"
            )
        lines += ["", "Reconstructed totals are not online native-serving end-to-end latency.", ""]
        for baseline, comparison in lane["quality_comparisons"].items():
            lines += [
                f"MosaicKV minus {baseline}: "
                f"{100 * comparison['mosaickv_minus_baseline_score']:+.2f} percentage points; "
                f"exact answer agreement {100 * comparison['exact_answer_agreement']:.1f}%. "
                "No statistical non-inferiority claim is established.", "",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("derived report must use a new output directory")
    report = analyze(args.index, args.index_sha256)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "audit.md").write_text(render(report))
    print(json.dumps({
        "output": str(args.output_dir),
        "execution_checks_passed": report["execution_checks_passed"],
        "telemetry_checks_passed": report["telemetry_checks_passed"],
        "all_canonical_preflights": report["all_canonical_preflights"],
    }))
    return 0 if report["execution_checks_passed"] and report["telemetry_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
