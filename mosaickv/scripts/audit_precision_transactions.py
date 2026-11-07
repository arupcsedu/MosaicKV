#!/usr/bin/env python3
"""Read only consumed, hash-bound layer captures; never load a model or answers."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import digest, sha, write_once
from mosaickv.evaluation.precision_transaction_study import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable derived-analysis directory required")
    source = Path(__file__).resolve().parents[2]
    torch.set_num_threads(2)
    results, inputs = {}, {}
    for worker in (0, 1):
        paths = sorted((args.input_root / f"workers/{worker}").glob("[0-9][0-9]-full.json"))
        if len(paths) != 1:
            raise ValueError("one completed fixed-reference arm per consumed input required")
        row = json.loads(paths[0].read_text())
        capture = row["evaluation_capture"]
        if row["status"] != "completed" or sha(Path(capture["path"])) != capture["sha256"]:
            raise ValueError("capture/producer row is incomplete or hash mismatch")
        results[row["sample_id"]] = study(
            torch.load(capture["path"], map_location="cpu", weights_only=True)
        )
        inputs[str(paths[0])] = sha(paths[0])
        inputs[capture["path"]] = capture["sha256"]
    code = source / "mosaickv/src/mosaickv/evaluation/precision_transaction_study.py"
    report = dict(
        results=results,
        all_proxy_gates_passed=all(r["proxy_gate_passed"] for r in results.values()),
        all_guarded_transactions_accepted=all(
            r["accepted_meaningful_change"] and r["final_joint_guard_passed"]
            for r in results.values()
        ),
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source)),
        constructor_sha=sha(code),
        inputs=inputs,
        measurement_type="cpu_consumed_capture_diagnostic",
        no_gpu_inference=True,
        no_benchmark_answers_used=True,
        superiority_established=False,
    )
    report["config_sha"] = digest({"input_hashes": inputs, "constructor_sha": sha(code)})
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output / "decision.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
