#!/usr/bin/env python3
"""Bounded exact history coalescing, CPU-gated admission, finite GPU stages."""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import evicastkv_source_history_runtime as runtime

from mosaickv.evaluation.fox_campaign import clean_source, sha

base = runtime.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/consolidated-history-consumed-20261030-v1")
VALIDATED = Path(
    "/scratch/djy8hg/runs/evicastkv/source-history-direct-consumed-20261030-v1/comparison"
)
original_prepare = runtime.prepare


def prepare(root, phase):
    if not base.common.read(VALIDATED / "analysis/source_history_decision.json")[
        "runtime_canary_passed"
    ]:
        raise ValueError("exact direct source history must already be validated")
    original = runtime.encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            for name in [
                "plan.json",
                "analysis/GATE.json",
                "analysis/manifest.json",
                "analysis/source_history_decision.json",
            ]:
                bound = VALIDATED / name
                value["bound_files"][str(bound)] = sha(bound)
            value.update(
                protocol="consolidated_exact_source_history_independent_layer_transactions_v1",
                source_history_comparison_kind="consolidated",
                candidate_constructor="direct",
                reference_constructor="direct",
                maximum_consolidation_final_ratio=1.02,
                maximum_consolidation_peak_ratio=1.05,
                original_bit_checks_per_consolidated_layer=3,
                graph_replay_enabled=False,
                attention_arithmetic_unchanged=True,
                source_packet_merge_uses_original_rope=True,
                source_packet_copy_overlap_charged=True,
                strict_bitwise_original_logits_required=True,
                full_retention_controls_128_tokens=root.name == "canary",
                no_new_selection_or_benchmark_labels=True,
                whole_request_peak_certified=False,
            )
        original(path, value)

    runtime.encoded.history.original_writer = register
    try:
        return original_prepare(root, phase)
    finally:
        runtime.encoded.history.original_writer = original


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import torch

    from mosaickv.evaluation.consolidated_history import gpu_controls
    from mosaickv.evaluation.storage_runs import run

    runtime.write_once(out / "consolidation_gpu_controls.json", gpu_controls())
    if plan["stage"] == "canary":
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        full = tuple(tuple(tuple(range(k.shape[-2])) for _ in range(k.shape[1])) for k, _ in source)
        variants, _ = runtime.encoded.fused_choices(adapter, prefill, capture, full, compact=True)
        formats = next(iter(variants.values()))
        outputs, records = {}, {}
        for name, policy in [("legacy", {}), ("consolidated", dict(consolidated_history=True))]:
            outputs[name], records[name] = run(
                adapter,
                prefill,
                formats,
                tokens=128,
                source_history=True,
                source_history_constructor="direct",
                audit=True,
                **policy,
            )
        passed = torch.equal(outputs["legacy"], outputs["consolidated"])
        passed = passed and torch.equal(outputs["legacy"][:32], full_reference)
        runtime.write_once(
            out / "full_retention_consolidated_controls.json",
            dict(
                original_128_logits_bitwise=passed,
                records=records,
                maximum_logit_difference=float(
                    (outputs["legacy"] - outputs["consolidated"]).abs().max()
                ),
            ),
        )
        torch.save(outputs, out / "full_retention_consolidated_logits.pt")
        if not passed:
            raise RuntimeError("history consolidation changed original retention-one logits")
    return runtime.study(
        adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
    )


def admit(root: Path, expected: str, cpu_xml: Path) -> None:
    plan = base.common.inputs(root, expected)
    record = base.common.read(root / "cpu_record.json")
    if (
        record["source_sha"] != plan["source_sha"]
        or record["returncode"] != 0
        or record["cpu_xml_sha256"] != sha(cpu_xml)
        or Path(record["cpu_xml"]).resolve() != cpu_xml.resolve()
    ):
        raise ValueError("CPU proof must match the immutable source and exact XML")
    document = ET.parse(cpu_xml).getroot()
    suites = list(document.iter("testsuite"))
    if (
        not suites
        or sum(int(s.get("tests", "0")) for s in suites) < 2674
        or sum(int(s.get("tests", "0")) for s in suites) != len(list(document.iter("testcase")))
        or any(int(s.get(k, "0")) for s in suites for k in ["failures", "errors", "skipped"])
    ):
        raise ValueError("complete clean full CPU suite must pass before admission")
    runtime.write_once(
        root / "cpu_admission.json",
        dict(
            cpu_xml=str(cpu_xml),
            cpu_xml_sha256=sha(cpu_xml),
            source_sha=base.common.read(root / "plan.json")["source_sha"],
            scope="same_two_consumed_inputs_only_no_fresh_or_paper_campaign",
        ),
    )
    base.launch.submit(
        root,
        expected,
        script_name="evicastkv_consolidated_history.sbatch",
        worker_minutes=30,
        memory_gb=24,
    )


def cpu(root: Path, expected: str) -> None:
    plan = base.common.inputs(root, expected)
    xml = root / "cpu-regression.xml"
    if xml.exists() or (root / "cpu_record.json").exists():
        raise FileExistsError("CPU proof is write-once")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "mosaickv/tests/unit",
        "mosaickv/tests/integration",
        f"--junitxml={xml}",
    ]
    result = subprocess.run(command, cwd=base.SOURCE, check=False)
    if clean_source(base.SOURCE) != plan["source_sha"]:
        raise ValueError("CPU source changed during execution")
    runtime.write_once(
        root / "cpu_record.json",
        dict(
            source_sha=plan["source_sha"],
            cpu_xml=str(xml),
            cpu_xml_sha256=sha(xml) if xml.exists() else "missing",
            returncode=result.returncode,
            command=command,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    runtime.encoded.history.paired.install()
    runtime.prepare = prepare
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_consolidated_history.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, runtime.analyze
    if len(sys.argv) > 1 and sys.argv[1] in {"admit", "cpu"}:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=["admit", "cpu"])
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha", required=True)
        parser.add_argument("--cpu-xml", type=Path)
        args = parser.parse_args()
        if args.mode == "cpu":
            cpu(args.root, args.plan_sha)
        else:
            admit(args.root, args.plan_sha, args.cpu_xml or args.root / "cpu-regression.xml")
    else:
        base.main()
