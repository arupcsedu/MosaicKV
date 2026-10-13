#!/usr/bin/env python3
"""Execute hash-bound vLLM mirror arms in isolated engine processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path: str, expected: str) -> Path:
    result = Path(path)
    if sha(result) != expected:
        raise ValueError(f"immutable input changed: {result}")
    return result


def verify_source(job: dict) -> None:
    root = Path(job["source_root"])
    actual_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != job["source_git_sha"]:
        raise ValueError("source base commit changed")
    attestation = job["source_attestation"]
    if attestation is not None:
        if job["stage"] != "preflight" or job["classification"] != "development_only":
            raise ValueError("dirty snapshots are allowed only for development preflight")
        inventory = json.loads(bound(attestation, job["source_attestation_sha256"]).read_text())
        if Path(inventory["source_root"]).resolve() != root.resolve():
            raise ValueError("source attestation belongs to another snapshot")
        for relative, expected in inventory["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("source attestation escapes snapshot")
            bound(str(path), expected)
    else:
        subprocess.run([str(root / "mosaickv/scripts/assert_clean_worktree.sh")], check=True)
    if job["stage"] == "campaign":
        gate = json.loads(
            bound(job["preflight_report"], job["preflight_report_sha256"]).read_text()
        )
        if (
            gate.get("status") != "passed"
            or gate.get("stage") != "preflight"
            or gate.get("classification") != "canonical_vllm_projection"
            or gate.get("source_git_sha") != actual_sha
            or gate.get("task") != job["task"]
            or gate.get("gate_binding") != job["gate_binding"]
            or gate.get("telemetry_checks_passed") is not True
        ):
            raise ValueError("campaign preflight gate is absent or incompatible")
    bound(job["hf_completion"], job["hf_completion_sha256"])
    bound(job["ledger"], job["ledger_sha256"])


def validate_outputs(job: dict, outputs: dict) -> dict:
    expected = set(job["sample_ids"])
    if len(expected) != len(job["sample_ids"]) or len(expected) < 2:
        raise ValueError("validation requires at least two unique sample identities")
    pairs = {}
    traces = {}
    hf_comparisons = {}
    timings = {}
    for arm in job["arms"]:
        name = arm["name"]
        root, run_id = outputs[name]
        manifest_path = root / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        config = yaml.safe_load(bound(arm["config"], arm["config_sha256"]).read_text())
        if manifest["resolved_config"] != config:
            raise ValueError(f"executed configuration differs from immutable input for {name}")
        if manifest["source"]["git_sha"] != job["source_git_sha"]:
            raise ValueError("output source commit differs from job")
        if job["classification"] != "development_only" and (
            manifest["source"]["git_dirty"] or not manifest["source"]["canonical_eligible"]
        ):
            raise ValueError("canonical job produced a noncanonical manifest")
        raw_path = root / run_id / "results.jsonl"
        if sha(raw_path) != manifest["artifacts"]["raw_output_sha"]:
            raise ValueError("vLLM raw artifact hash changed")
        if not manifest["execution"]["request_isolation_verified"]:
            raise ValueError(f"A-B-A isolation failed for {name}")
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(expected) or {row["sample_id"] for row in rows} != expected:
            raise ValueError(f"missing/duplicate samples for {name}")
        if any(row["status"] != "completed" for row in rows):
            raise ValueError(f"failed samples for {name}; preserved in raw output")
        for row in rows:
            for field in (
                "task_score",
                "ttft",
                "prefill_time",
                "compression_time",
                "decode_time",
                "end_to_end_time",
                "active_kv_bytes",
            ):
                if row.get(field) is None or not math.isfinite(float(row[field])):
                    raise ValueError(f"missing/nonfinite {field} for {name}")
            if row["generated_tokens"] != job["shared_controls"]["generation"]["max_new_tokens"]:
                raise ValueError("generated output was silently shortened")
        pairs[name] = {row["sample_id"]: row for row in rows}
        digest = hashlib.sha256()
        trace_list = []
        for path in sorted((root / "traces" / run_id).glob("*.json")):
            payload = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(payload)
            trace_list.append(json.loads(payload))
        if digest.hexdigest() != manifest["artifacts"]["log_sha"]:
            raise ValueError(f"vLLM trace artifact hash changed for {name}")
        traces[name] = {trace["sample_id"]: trace for trace in trace_list}
        if len(trace_list) != len(expected) or set(traces[name]) != expected:
            raise ValueError("trace identities incomplete")
        isolation = [trace for trace in trace_list if trace.get("request_isolation")]
        if len(isolation) != 1:
            raise ValueError("missing or ambiguous A-B-A trace evidence")
        anchor = isolation[0]
        probe = anchor["request_isolation"]
        if not (
            probe["performed"] is True
            and probe["token_ids_match_anchor"] is True
            and probe["intervening_distinct_input_fingerprints"] >= 1
            and probe["anchor_token_ids"] == probe["probe_token_ids"]
            == anchor["trials"][0]["token_ids"]
        ):
            raise ValueError("A-B-A trace evidence does not match manifest assertion")
        for trace in trace_list:
            if trace["status"] != "completed" or any(
                len(trial["token_ids"]) != job["shared_controls"]["generation"]["max_new_tokens"]
                for trial in trace["trials"]
            ) or not trace["trials"]:
                raise ValueError("incomplete token trace")
            if trace["kv_cache_accounting"]["active_kv_bytes"] != (
                pairs[name][trace["sample_id"]]["active_kv_bytes"]
            ):
                raise ValueError("trace/raw active-byte accounting differs")
        hf_manifest = json.loads(bound(arm["hf_manifest"], arm["hf_manifest_sha256"]).read_text())
        hf_raw = Path(arm["hf_manifest"]).with_name("results.jsonl")
        bound(str(hf_raw), hf_manifest["artifacts"]["raw_output_sha"])
        hf_rows = [json.loads(line) for line in hf_raw.read_text().splitlines() if line.strip()]
        selected_hf = [row for row in hf_rows if row["sample_id"] in expected]
        if len(selected_hf) != len(expected) or any(
            row["status"] != "completed" for row in selected_hf
        ) or {row["sample_id"] for row in selected_hf} != expected:
            raise ValueError("HF reference rows are missing, duplicated, or failed")
        for row in selected_hf:
            if row["reference"] != pairs[name][row["sample_id"]]["reference"]:
                raise ValueError("HF/vLLM reference annotations differ")
        if name not in {"full_kv", "native_identity"}:
            catalog = json.loads(bound(arm["catalog"], arm["catalog_sha256"]).read_text())
            source_tokens = {}
            for entry in catalog["entries"]:
                if entry["sample_id"] not in expected:
                    continue
                source_trace = json.loads(
                    bound(entry["source_trace"], entry["source_trace_sha256"]).read_text()
                )
                if source_trace["sample_id"] != entry["sample_id"]:
                    raise ValueError("HF source trace identity differs from selector catalog")
                source_tokens[entry["sample_id"]] = source_trace["generated_token_ids"]
            if set(source_tokens) != expected:
                raise ValueError("HF source token traces incomplete")
            token_count = sum(len(tokens) for tokens in source_tokens.values())
            if token_count == 0 or any(
                len(tokens) != len(traces[name][sid]["trials"][0]["token_ids"])
                for sid, tokens in source_tokens.items()
            ):
                raise ValueError("HF/vLLM diagnostic token lengths differ")
            hf_comparisons[name] = {
                "sample_count": len(expected),
                "hf_score_mean": sum(row["task_score"] for row in selected_hf) / len(expected),
                "vllm_score_mean": sum(row["task_score"] for row in pairs[name].values())
                / len(expected),
                "exact_answer_agreement": sum(
                    row["answer"] == pairs[name][row["sample_id"]]["answer"]
                    for row in selected_hf
                ) / len(expected),
                "fixed_length_token_agreement": sum(
                    left == right for sid, tokens in source_tokens.items()
                    for left, right in zip(tokens, traces[name][sid]["trials"][0]["token_ids"],
                                           strict=True)
                ) / token_count,
                "token_scope": "all_fixed_length_tokens_including_after_eos",
                "interpretation": "different_cache_layouts_not_a_bitwise_parity_gate",
            }
            timings[name] = {
                "engine_request_mean_seconds": sum(
                    trace["trials"][0]["request_latency_seconds"] for trace in trace_list
                ) / len(expected),
                "decode_mean_seconds": sum(row["decode_time"] for row in pairs[name].values())
                / len(expected),
                "offline_planner_plus_transaction_mean_seconds": sum(
                    row["compression_time"] for row in pairs[name].values()
                ) / len(expected),
                "reconstructed_total_mean_seconds": sum(
                    row["end_to_end_time"] for row in pairs[name].values()
                ) / len(expected),
                "not_online_native_end_to_end": True,
            }
    compressed = [arm for arm in job["arms"] if arm["name"] not in {"full_kv", "native_identity"}]
    checks = []
    telemetry_issues = []
    for sample_id in job["sample_ids"]:
        sizes = {
            arm["name"]: pairs[arm["name"]][sample_id]["active_kv_bytes"] for arm in compressed
        }
        fingerprints = {
            (traces[name][sample_id]["prompt_sha256"], traces[name][sample_id]["media_sha256"])
            for name in traces
        }
        if len(fingerprints) != 1:
            raise ValueError("prompt/media differ across vLLM arms")
        if len({pairs[name][sample_id]["reference"] for name in pairs}) != 1:
            raise ValueError("reference annotations differ across vLLM arms")
        if len(set(sizes.values())) != 1 or max(sizes.values()) > job["physical_byte_cap"]:
            raise ValueError(f"per-sample realized bytes differ or exceed shared cap: {sizes}")
        for arm in compressed:
            catalog = json.loads(bound(arm["catalog"], arm["catalog_sha256"]).read_text())
            entry = next(entry for entry in catalog["entries"] if entry["sample_id"] == sample_id)
            block_bytes = entry["projection"]["block_size"] * entry["bytes_per_position"]
            planned = len(entry["projection"]["selected_blocks"]) * block_bytes
            if sizes[arm["name"]] != planned:
                raise ValueError("committed active bytes differ from maximal feasible plan")
            audited = traces[arm["name"]][sample_id]["trials"][0].get("native_audit", {})
            ratio = audited.get("selection", {}).get("retention_ratio")
            expected_ratio = entry["projection"]["retention_ratio"]
            if not isinstance(ratio, (float, int)) or not math.isclose(
                ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12
            ):
                telemetry_issues.append({
                    "sample_id": sample_id, "method": arm["name"],
                    "issue": "native_audit_ratio_differs_from_byte_cap_plan",
                    "audit_ratio": ratio, "plan_ratio": expected_ratio,
                })
        if job["stage"] == "preflight":
            ordinary = traces["full_kv"][sample_id]["trials"][0]["token_ids"]
            identity = traces["native_identity"][sample_id]["trials"][0]["token_ids"]
            if ordinary != identity:
                raise ValueError("retention-1 native/ordinary vLLM token parity failed")
        checks.append(
            {
                "sample_id": sample_id,
                "active_kv_bytes": sizes,
                "byte_cap": job["physical_byte_cap"],
                "exact_byte_match": True,
            }
        )
    quality = {}
    mosaic = pairs["mosaickv_exact"]
    for arm in compressed:
        name = arm["name"]
        if name == "mosaickv_exact":
            continue
        delta = sum(
            mosaic[sid]["task_score"] - pairs[name][sid]["task_score"] for sid in expected
        ) / len(expected)
        quality[name] = {
            "mosaickv_minus_baseline_score": delta,
            "observed_mean_no_worse": delta >= 0,
            "exact_answer_agreement": sum(
                mosaic[sid]["answer"] == pairs[name][sid]["answer"] for sid in expected
            ) / len(expected),
            "statistical_noninferiority_established": False,
        }
    return {
        "status": "passed",
        "status_scope": "execution_integrity_only_not_quality_or_paper_admissibility",
        "telemetry_checks_passed": not telemetry_issues,
        "telemetry_issues": telemetry_issues,
        "classification": job["classification"],
        "stage": job["stage"],
        "task": job["task"],
        "retention_1_parity": True if job["stage"] == "preflight" else "not_repeated",
        "request_isolation": True,
        "byte_checks": checks,
        "sample_count": len(expected),
        "fullkv_latency_excluded": True,
        "hf_vs_vllm_compressed_quality": "report_separately_not_bitwise_identity",
        "full_campaign_authorized_by_this_report": False,
        "quality_comparisons": quality,
        "hf_comparisons": hf_comparisons,
        "compressed_timing": timings,
        "mean_task_scores": {
            name: sum(row["task_score"] for row in values.values()) / len(values)
            for name, values in pairs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    job = json.loads(bound(args.job, args.job_sha256).read_text())
    verify_source(job)
    root = Path(job["output_root"])
    finished = root / "validation.json"
    if finished.exists():
        previous = json.loads(finished.read_text())
        if previous.get("job_sha256") != args.job_sha256 or previous.get("status") != "passed":
            raise ValueError("existing terminal record differs; retry in a new output root")
        return 0
    outputs = {}
    try:
        for arm in job["arms"]:
            verify_source(job)
            bound(arm["hf_manifest"], arm["hf_manifest_sha256"])
            config = bound(arm["config"], arm["config_sha256"])
            out = root / "outputs" / arm["name"]
            out.mkdir(parents=True, exist_ok=True)
            run_id = f"vllm-bytecap-{args.job_sha256[:12]}-{arm['name']}"
            command = [
                sys.executable,
                "-m",
                "mosaickv.cli",
                "evaluate",
                "--config",
                str(config),
                "--task",
                job["task"],
                "--sample-id-file",
                job["ledger"],
                "--sample-id-file-sha",
                job["ledger_sha256"],
                "--local-files-only",
                "--cache-probe-repeats",
                "1",
                "--gpu-memory-utilization",
                "0.65",
                "--vllm-max-model-len",
                "4096",
                "--tensor-parallel-size",
                "1",
                "--output-dir",
                str(out),
                "--trace-directory",
                str(out / "traces"),
                "--run-id",
                run_id,
                "--json",
            ]
            if arm["catalog"]:
                bound(arm["catalog"], arm["catalog_sha256"])
                command += [
                    "--enable-mosaickv",
                    "--vllm-selector-plan",
                    arm["catalog"],
                    "--vllm-selector-plan-sha",
                    arm["catalog_sha256"],
                ]
            with (out / "stdout.log").open("a") as stdout, (out / "stderr.log").open("a") as stderr:
                subprocess.run(
                    command,
                    cwd=Path(job["source_root"]) / "mosaickv",
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                )
            outputs[arm["name"]] = (out, run_id)
        result = validate_outputs(job, outputs)
    except Exception as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "planned_sample_ids": job["sample_ids"],
            "classification": job["classification"],
            "full_campaign_authorized_by_this_report": False,
        }
    result.update(
        job_sha256=args.job_sha256,
        source_git_sha=job["source_git_sha"],
        gate_binding=job.get("gate_binding"),
        job_path=args.job,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    with finished.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
