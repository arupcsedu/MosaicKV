#!/usr/bin/env python3
"""Finite CPU -> one A100 synthetic operator -> terminal analysis, no models."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/eager-core-operator-20261106-import-retry1")


def cpu(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    parent = json.loads((PROFILE / "analysis/decision.json").read_text())
    if not parent["complete"] or parent["failures"]:
        raise ValueError("complete warmed attribution required before this operator")
    write_once(
        root / "config.json",
        dict(
            source_sha=revision,
            warmups=5,
            repetitions=20,
            seed=20261106,
            source_lengths=[600, 604],
            minimum_layout_opportunities=16,
            uniform_only_capability=True,
            reuse_views=True,
            uncached_uniform_ablation=True,
            adaptive_acceleration_supported=False,
            previous_universal_operator_gate_remains_negative=True,
            attention_workspace_bound_bytes=393216,
            attention_math="same_native_eager_operators_no_SDPA",
            synthetic_operator_only=True,
            dataset_access=False,
            model_weights_loaded=False,
            benchmark_labels_used=False,
            independent_quality_N=0,
            parent_profile=str(PROFILE / "analysis/decision.json"),
            parent_profile_sha256=sha(PROFILE / "analysis/decision.json"),
            unchanged_attention_probabilities_and_BF16_words=True,
            one_shared_K_to_V_lease_required=True,
            cold_capture_input_copies_sync_and_retirement_included=True,
            no_model_integration_or_quality_campaign_from_this_result=True,
        ),
    )
    xml = root / "cpu.xml"
    with (root / "cpu.stdout").open("w") as out, (root / "cpu.stderr").open("w") as err:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "mosaickv/tests/unit",
                "mosaickv/tests/integration",
                f"--junitxml={xml}",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=err,
            check=False,
        )
    suites = ET.parse(xml).getroot().iter("testsuite") if xml.exists() else ()
    values = list(suites)
    failures = sum(int(s.get(k, 0)) for s in values for k in ("failures", "errors", "skipped"))
    count = sum(int(s.get("tests", 0)) for s in values)
    passed = not result.returncode and not failures and count >= 4451
    write_once(
        root / "cpu.json",
        dict(
            source_sha=revision,
            returncode=result.returncode,
            failed_or_skipped=failures,
            test_count=count,
            xml_sha256=sha(xml) if xml.exists() else None,
            passed=passed,
        ),
    )
    if not passed:
        raise RuntimeError("complete same-SHA CPU suite failed; no GPU promotion")


def checked(root, revision):
    config = json.loads((root / "config.json").read_text())
    if clean_source(SOURCE) != revision or config["source_sha"] != revision:
        raise ValueError("unchanged clean registered source required")
    if sha(Path(config["parent_profile"])) != config["parent_profile_sha256"]:
        raise ValueError("changed prior attribution")
    return config


def run(root, revision):
    folder = root / "operator"
    folder.mkdir(parents=True, exist_ok=False)
    try:
        checked(root, revision)
        proof = json.loads((root / "cpu.json").read_text())
        if (
            not proof["passed"]
            or proof["source_sha"] != revision
            or proof["xml_sha256"] != sha(root / "cpu.xml")
        ):
            raise ValueError("successful same-SHA CPU proof required")
        import torch

        versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "transformers", "accelerate", "datasets", "lmms-eval")
        }
        hardware = folder / "nvidia-smi.txt"
        hardware.write_text(subprocess.check_output(["nvidia-smi", "-q"], text=True))
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=revision,
                source_dirty=False,
                config_sha256=sha(root / "config.json"),
                model_id="not_used",
                model_revision="not_used",
                dataset="synthetic_operator_fixture",
                dataset_revision=revision,
                measurement_type="synthetic_operator_diagnostic_measured",
                result_category="unified_comparison",
                result_origin="locally_measured",
                not_a_measured_benchmark_or_paper_table=True,
                software=versions,
                vllm="not_used",
                sglang="not_used",
                cuda=torch.version.cuda,
                driver=subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                ).strip(),
                gpu_type=torch.cuda.get_device_name(),
                gpu_count=torch.cuda.device_count(),
                backend="torch_triton_synthetic_operator",
                attention_implementation="not_used",
                seed=20261106,
                precision="bf16_words_no_arithmetic_changes",
                hardware_snapshot_sha256=sha(hardware),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                python_executable=sys.executable,
                cache_paths={
                    k: os.environ.get(k)
                    for k in (
                        "MOSAICKV_CACHE_ROOT",
                        "HF_HOME",
                        "PIP_CACHE_DIR",
                        "TMPDIR",
                        "TRITON_CACHE_DIR",
                    )
                },
                job_id=os.environ.get("SLURM_JOB_ID", "not_slurm"),
                independent_quality_N=0,
            ),
        )
        from mosaickv.evaluation.uniform_eager_core import study

        result = study()
    except Exception as error:
        result = dict(
            complete=False,
            failures=[f"{type(error).__name__}: {error}"],
            traceback=traceback.format_exc(),
            independent_quality_N=0,
        )
    write_once(folder / "result.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "failures")}))
    if not result["complete"]:
        raise SystemExit(1)


def analyze(root, revision):
    path = root / "operator/result.json"
    try:
        checked(root, revision)
        result = (
            json.loads(path.read_text())
            if path.exists()
            else dict(complete=False, failures=["operator result missing"])
        )
    except Exception as error:
        result = dict(complete=False, failures=[f"{type(error).__name__}: {error}"])
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=result["complete"],
            failures=result["failures"],
            operator_result_sha256=sha(path) if path.exists() else None,
            source_sha=revision,
            config_sha256=sha(root / "config.json") if (root / "config.json").exists() else None,
            uniform_feature_screen_passed=result["complete"]
            and result["uniform_feature_operator_feasible"],
            uniform_only_capability=True,
            adaptive_acceleration_supported=False,
            previous_universal_operator_gate_remains_negative=True,
            records=result.get("records", []),
            synthetic_operator_only=True,
            independent_quality_N=0,
            no_model_latency_or_quality_acceptance=True,
            no_automatic_model_or_quality_submission=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    {"cpu": cpu, "run": run, "analyze": analyze}[args.action](args.root, args.source_sha)


if __name__ == "__main__":
    main()
