#!/usr/bin/env python3
"""CPU/resources -> two native-tile operators -> terminal afterany analysis."""

import argparse
import os
import re
import subprocess
import traceback
from pathlib import Path

import evicastkv_row4_operator as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = common.SOURCE, common.driver
common.SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_repaired_tile_operator.sbatch"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/graph-arena-consumed-20261104-retry2")
NEGATIVE = Path("/scratch/djy8hg/runs/evicastkv/column-byte-operator-20261104-v1")


def compile_proof(root):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from mosaickv.evaluation.repaired_source_tile import _project

    rows = []
    for columns in (False, True):
        types = [
            "*u8",
            "*bf16",
            "*fp32",
            "*u8",
            "*bf16" if columns else "*u8",
            "*bf16",
            "*bf16",
            "i32",
            "i32",
        ]
        compiled = triton.compile(
            ASTSource(
                _project,
                dict(zip(_project.arg_names[:9], types, strict=True)),
                constexprs=dict(WIDTH=4096, HAS_COLUMNS=columns),
                attrs={(i,): [("tt.divisibility", 16)] for i in range(7)},
            ),
            target=GPUTarget("cuda", 80, 32),
            options=dict(num_warps=4, num_stages=3, enable_fp_fusion=False),
        )
        binary = root / f"native-tile-columns-{columns}.cubin"
        with binary.open("xb") as stream:
            stream.write(compiled.asm["cubin"])
        resources = subprocess.check_output(
            [
                str(Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"),
                "--dump-resource-usage",
                str(binary),
            ],
            text=True,
        )
        match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
        if not match:
            raise ValueError("missing explicit native-tile compiler resource witness")
        rows.append(
            dict(
                columns=columns,
                registers=int(match[1]),
                stack_bytes=int(match[2]),
                local_bytes=int(match[4]),
                resources=resources,
                shared_bytes=compiled.metadata.shared,
                scratch_bytes=compiled.metadata.global_scratch_size,
                binary_sha256=sha(binary),
            )
        )
    return dict(rows=rows, CPU_only=True, no_GPU_execution=True)


def validate_proof(proof):
    if (
        len(proof["rows"]) != 2
        or {r["columns"] for r in proof["rows"]} != {False, True}
        or any(r["stack_bytes"] or r["scratch_bytes"] or r["local_bytes"] for r in proof["rows"])
    ):
        raise ValueError("both native-tile column paths must have zero scratch/stack/local storage")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    negative = driver.read(NEGATIVE / "analysis/decision.json")
    timing = driver.read(PARENT / "paired-timing/analysis/decision.json")
    if (
        not negative["complete"]
        or negative["operator_gate"]
        or not timing["complete"]
        or timing["four_percent_latency_gate"]
    ):
        raise ValueError("preserved complete staging and full-request negatives required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (
        NEGATIVE / "analysis/decision.json",
        PARENT / "paired-timing/analysis/decision.json",
        Path(
            "/scratch/djy8hg/runs/evicastkv/native-operator-consumed-20261103-v1/analysis/decision.json"
        ),
        SOURCE / "mosaickv/docs/repaired_tile_operator_20261104.md",
    ):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="no_workspace_repaired_native_tile_operator",
            no_checkpoint_loaded=True,
            independent_quality_N=0,
            operator_upper_ratio=0.8,
            per_layer_numerical_patch_cap_bytes=32768,
            automatic_model_submission=False,
            automatic_quality_submission=False,
            changed_matmul_tiling_but_original_KV_after_fixed_exact_repair=True,
        ),
    )
    common.stage(root, "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "cpu_analysis", "operator", "operator_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
        driver.cpu_proof(args.root, driver.checked(args.root, args.plan_sha))
        proof = None
        try:
            proof = compile_proof(args.root)
            validate_proof(proof)
            write_once(args.root / "compiler.json", dict(complete=True, proof=proof))
        except Exception as error:
            write_once(
                args.root / "compiler.json",
                dict(
                    complete=False,
                    proof=proof,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                ),
            )
            raise
    elif args.action == "cpu_analysis":
        try:
            control = driver.read(args.root / "compiler.json")
            if not control["complete"]:
                raise ValueError(control)
            validate_proof(control["proof"])
        except Exception as error:
            write_once(
                args.root / "cpu-analysis/decision.json",
                dict(
                    complete=False,
                    failures=[str(error)],
                    no_GPU_submitted=True,
                ),
            )
        else:
            common.analyze(args.root, args.plan_sha, args.action)
    elif args.action == "operator":
        from mosaickv.evaluation import repaired_tile_operator, row_major_source_operator

        row_major_source_operator.run = repaired_tile_operator.run
        common.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        common.analyze(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
