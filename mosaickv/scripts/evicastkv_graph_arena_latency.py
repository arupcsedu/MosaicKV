#!/usr/bin/env python3
"""Same-input graph/arena controls -> paired PrefixKV gate -> all baselines."""

import argparse
import os
import subprocess
import traceback
from pathlib import Path

import evicastkv_prompt_graph_latency as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = previous.SOURCE, previous.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_graph_arena_latency.sbatch"
OWNED = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-owned-consumed-20261104-v1")
RESUME = Path("/scratch/djy8hg/runs/evicastkv/graph-arena-consumed-20261104-retry1")
RESUME_SHA = "fc94667435fce1e116f889339fa3f17c04d40195"
APPROVED_GUARD_PATHS = {
    "mosaickv/scripts/evicastkv_graph_arena_latency.py",
    "mosaickv/docs/graph_arena_latency_20261104.md",
    "mosaickv/tests/unit/test_graph_arena_latency_controller.py",
}
POLICY = dict(history_arena=True, captured_code_replay=True, row_major_codes=False)
ORIGINAL_BUILDER = previous.make_records
ORIGINAL_EXECUTE = previous.direct.execute


def make_records(parent, methods):
    result = ORIGINAL_BUILDER(parent, methods)
    for method, arms in result.items():
        arms["graph-original"] = arms["graph"]
        arms["graph"] = dict(arms["graph"], prompt_graph_policy=dict(POLICY))
        if method == "prefixkv_reimpl":
            arms["whole-graph-original"] = arms["whole-graph"]
            arms["whole-graph"] = dict(arms["whole-graph"], prompt_graph_policy=dict(POLICY))
    return result


def required(final):
    methods = previous.METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
    return {(m, a) for m in methods for a in ("direct", "graph-original", "graph")} | {
        ("prefixkv_reimpl", a) for a in ("whole-6291456", "whole-graph-original", "whole-graph")
    }


def expected_arena(capacity):
    """Frozen LLaVA: 32 layers, 32 MHA heads, BF16, three independent fills."""
    if type(capacity) is not int or capacity < 8192 or capacity % 8192:
        raise ValueError("whole-token BF16 MHA workspace backing required")
    requirements = [4 * n * 32 * 128 * 2 for n in (32, 48, 64, 80, 96, 112)]
    fits = [n for n in requirements if n <= capacity]
    return dict(borrows=32 * len(fits), misses=32 * (6 - len(fits)), maximum=max(fits, default=0))


def validate_engagement(measured, plan, method, record):
    if record.get("prompt_graph_policy") is not None:
        info = measured["prompt_graph_reader"]
        expected = expected_arena(info["maximum_workspace_bytes"])
        if (
            record["prompt_graph_policy"] != POLICY
            or info["optimized_policy"] != POLICY
            or info["history_borrows"] != expected["borrows"]
            or info["history_borrow_misses"] != expected["misses"]
            or info["history_borrow_maximum_bytes"] != expected["maximum"]
            or info["row4_workspace_peak_bytes"] != 0
            or info["row4_replays"] != 0
            or not info["retired"]
        ):
            raise RuntimeError(
                f"geometry-derived arena engagement required: {info!r}; expected={expected!r}"
            )
        if method in plan.get("code_delta_methods", []) and (
            info["captured_code_replays"] < 1 or info["captured_code_kernel_dispatches"] < 1
        ):
            raise RuntimeError("ordered code replay must genuinely execute in captured kernels")


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    measured = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    validate_engagement(measured, plan, method, record)
    return measured


def compatible_source(revision, *, changed=None):
    if changed is None:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", RESUME_SHA, revision], cwd=SOURCE, text=True
        ).splitlines()
    if not set(changed) <= APPROVED_GUARD_PATHS:
        raise ValueError("control reuse forbids every runtime/algorithm source change")


def reused_controls(index, expected):
    worker = RESUME / f"workers/{index}"
    result = driver.read(worker / "result.json")
    if (
        result["source_sha"] != RESUME_SHA
        or result["status"] != "failed"
        or result["failures"]
        != ["RuntimeError: complete charged graph/arena policy engagement required"]
    ):
        raise ValueError("only the recorded guard false-positive may reuse positive controls")
    controls = {}
    for method, arm in required(True):
        name = f"control-{method}-{arm}.json"
        if name not in result["outputs"]:
            continue
        path = worker / name
        if sha(path) != result["outputs"][name]:
            raise ValueError("changed positive control artifact")
        record = driver.read(path)

        def policy(value):
            return {k: v for k, v in value.items() if k not in ("audited", "packet_sha256")}

        if policy(record) != policy(expected[method][arm]):
            raise ValueError("reused control changed the frozen method/reader policy")
        audited = record["audited"]
        if audited["logits_bitwise"] is not True or audited["no_FullKV_fallback"] is not True:
            raise ValueError("only fully positive parent-logit/fallback controls may be reused")
        controls[method, arm] = (
            record,
            dict(
                source_sha=RESUME_SHA,
                path=str(path),
                sha256=sha(path),
                measurement_reused=True,
                derived_copy_not_new_GPU_measurement=True,
            ),
        )
    return controls


def summarize(rows, seed, *, final=False):
    import numpy as np

    expected = {(m, a, t) for m, a in required(final) for t in range(-5, 20)}
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("all original AND optimized readers, five warmups/twenty pairs required")
    arrays, summaries = {}, []
    for m, a in sorted(required(final)):
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (m, a)), key=lambda r: r["trial"]
        )
        for r in selected:
            measured = r["measurement"]
            if (
                r["warmup"] is not (r["trial"] < 0)
                or measured["logits_bitwise"] is not True
                or measured["no_FullKV_fallback"] is not True
            ):
                raise ValueError(
                    "bitwise parent logits, exact warmup identity, no fallback required"
                )
            value = measured["total_wall_seconds"]
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError("finite positive nonboolean full-request latency required")
        values = np.array([r["measurement"]["total_wall_seconds"] for r in selected[5:]])
        arrays[m, a] = values
        summaries.append(
            dict(
                method=m,
                arm=a,
                mean=float(values.mean()),
                p50=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
            )
        )
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    candidate, comparisons = arrays["mosaickv_exact", "graph"], []
    tail = 0.05 / (2 * 64)
    for (m, a), values in arrays.items():
        if (m, a) == ("mosaickv_exact", "graph"):
            continue
        ratios = candidate[draws].mean(1) / values[draws].mean(1)
        ci = np.quantile(ratios, [tail, 1 - tail]).tolist()
        comparisons.append(
            dict(
                method=m,
                arm=a,
                mean_ratio=float(candidate.mean() / values.mean()),
                adjusted_ratio_CI=ci,
                equal_or_faster=bool(ci[1] <= 1),
                four_percent_faster=bool(ci[1] <= 0.96),
            )
        )
    baselines = [r for r in comparisons if r["method"] != "mosaickv_exact"]
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        four_percent_latency_gate=all(r["four_percent_faster"] for r in baselines),
        equal_or_faster_gate=all(r["equal_or_faster"] for r in baselines),
        acceptance_ratio_upper_bound=0.96,
        bootstrap_replicates=10000,
        family_maximum_engineering_contrasts=64,
        intervals_over_repetitions_not_independent_examples=True,
    )


def submit(root, action, *, dependency=None):
    plan = driver.checked(root, sha(root / "plan.json"))
    gpu = action in ("study", "timing", "all_baselines")
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if gpu:
        driver.cpu_proof(root, plan)
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-1%2",
            "--time="
            + {"study": "00:35:00", "timing": "01:15:00", "all_baselines": "01:55:00"}[action],
        ]
    else:
        args += ["--partition=bii", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-garena-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if gpu:
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
        )
        write_once(root / "scheduler/policy.json", policy)
        write_once(
            root / "scheduler/registration.json",
            tick(policy, {}, root / "scheduler/state.json", False),
        )
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    compatible_source(revision)
    old = driver.read(OWNED / "plan.json")
    bound = dict(old["bound_files"])
    for path in (
        OWNED / "plan.json",
        OWNED / "analysis/decision.json",
        SOURCE / "mosaickv/docs/graph_arena_latency_20261104.md",
        Path(
            "/scratch/djy8hg/runs/evicastkv/graph-arena-consumed-20261104-v1/analysis/decision.json"
        ),
        Path(
            "/scratch/djy8hg/runs/evicastkv/captured-code-retirement-20261104-retry1/diagnostic.py"
        ),
        Path("/scratch/djy8hg/runs/evicastkv/captured-code-retirement-20261104-retry1/result.json"),
    ):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        path = OWNED / f"workers/{index}/study.json"
        bound[str(path)] = sha(path)
        for path in (RESUME / f"workers/{index}").glob("*.json"):
            bound[str(path)] = sha(path)
    for path in (RESUME / "plan.json", RESUME / "cpu.json", RESUME / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_captured_code_and_verification_arena",
            prompt_graph_root=str(root),
            optimized_graph_policy=POLICY,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
            registered_request_ratio_upper=0.96,
            acceptance_ratio_upper_bound=0.96,
            positive_control_reuse_root=str(RESUME),
            positive_control_producer_sha=RESUME_SHA,
            runtime_source_unchanged_except_guard_paths=sorted(APPROVED_GUARD_PATHS),
        ),
    )
    previous.stage(root, "cpu")


def install():
    previous.make_records, previous.summarize, previous.submit = make_records, summarize, submit
    previous.direct.execute = execute


def study(root, expected, index):
    from mosaickv.evaluation import captured_code_gpu_control, prompt_graph_reader

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan = driver.checked(root, expected)
        compatible_source(plan["source_sha"])
        write_once(
            out / "primitive.json",
            dict(
                ordinary_prompt_graph=prompt_graph_reader.gpu_controls(),
                captured_codes=captured_code_gpu_control.run(),
            ),
        )
        parent = driver.read(previous.PARENT / f"workers/{index}/study.json")
        records = make_records(parent, previous.METHODS)
        reuse = reused_controls(index, records)
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_guard_corrected_selective_control_retry",
                methods=list(previous.METHODS),
                actual_timing_protocol=dict(
                    output_predictions=128,
                    not_latency_evidence=True,
                    reused_controls=[
                        dict(method=m, arm=a, **p) for (m, a), (_, p) in reuse.items()
                    ],
                    runtime_source_compatible_with=RESUME_SHA,
                ),
            ),
        )
        for method, arms in records.items():
            for arm, record in arms.items():
                name = f"control-{method}-{arm}.json"
                if (method, arm) in reuse:
                    record, provenance = reuse[method, arm]
                    validate_engagement(record["audited"], plan, method, record)
                    record = dict(record, control_provenance=provenance)
                else:
                    family = "whole" if arm.startswith("whole") else "direct"
                    if family == "direct":
                        digest = parent["records"][method]["packet_sha256"]
                    else:
                        digest = records[method]["whole-6291456"]["packet_sha256"]

                    def verify(packets, digest=digest):
                        if previous.direct.packet_signature(packets)["digest"] != digest:
                            raise RuntimeError("reader changed frozen parent packet words")

                    measured = execute(
                        adapter,
                        refs,
                        messages,
                        profile,
                        plan,
                        method,
                        record,
                        audit=True,
                        final_validator=verify,
                    )
                    anchor = parent["records"][method][
                        "whole-6291456" if family == "whole" else "direct"
                    ]["audited"]
                    if any(measured[k] != anchor[k] for k in driver.STORAGE[:2]):
                        raise RuntimeError("reader changed initial or final encoded backing")
                    if record["prompt_graph_reader"]:
                        info = measured["prompt_graph_reader"]
                        if (
                            info["captures"] != 64
                            or info["replays"] != 8128
                            or info["graph_private_tensor_bytes"] != 1024
                            or info["graph_private_live_bytes"] != 0
                            or info["unowned_tensor_bytes"] != 0
                            or not info["retired"]
                            or info["active_workspace_bytes"]
                        ):
                            raise RuntimeError(
                                "complete private fully charged graph replay required"
                            )
                    record = dict(
                        record,
                        audited=measured,
                        packet_sha256=digest,
                        control_provenance=dict(
                            source_sha=plan["source_sha"], measurement_reused=False
                        ),
                    )
                records[method][arm] = record
                write_once(out / name, record)
        write_once(
            out / "study.json",
            dict(
                records=records,
                independent_quality_N=0,
                old_measurements_retain_exact_producer_provenance=True,
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "cpu_analysis",
            "study",
            "study_analysis",
            "timing",
            "timing_analysis",
            "all_baselines",
            "all_baselines_analysis",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    install()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action.endswith("_analysis"):
        previous.analyze(args.root, args.plan_sha, args.action)
    elif args.action == "study":
        study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        previous.timing(
            args.root,
            args.plan_sha,
            int(os.environ["SLURM_ARRAY_TASK_ID"]),
            all_methods=args.action == "all_baselines",
        )


if __name__ == "__main__":
    main()
