#!/usr/bin/env python3
"""Numerical-first same-consumed full-request uniform eager replay screen."""

import os
import subprocess
from pathlib import Path

import evicastkv_async_text_native as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base, SOURCE, parent, driver = prior.base, prior.SOURCE, prior.parent, prior.driver
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/fused-predicate-screen-consumed-20261106-v1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/uniform-core-operator-20261106-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_uniform_core_screen.sbatch"
DOC = SOURCE / "mosaickv/docs/uniform_core_screen.md"
ARMS = (
    ("mosaickv_exact", "joint-async-handoff"),
    ("mosaickv_exact", "joint-async-uniform-core"),
    ("prefixkv_reimpl", "graph-history-graph"),
    ("prefixkv_reimpl", "graph-original-history-graph"),
)
OLD_OVERRIDES, OLD_EXECUTE = prior.overrides, prior.execute
_checked = set()
GPU_MINUTES = 20


def submit(root, action, dependency=None):
    plan = driver.checked(root, sha(root / "plan.json"))
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    gpu = action == "profile"
    if gpu:
        driver.cpu_proof(root, plan)
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-1%2",
            f"--time=00:{GPU_MINUTES:02d}:00",
        ]
    else:
        args += [
            "--partition=bii",
            "--nodelist=udc-aj36-2c0",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
        ]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-uniform-{action}",
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


def overrides(method, arm):
    canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
    return dict(
        OLD_OVERRIDES(method, canonical),
        fused_transport_predicate=False,
        uniform_attention_core=arm == "joint-async-uniform-core" or method != "mosaickv_exact",
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for folder in (PREVIOUS, OPERATOR):
        gate = driver.read(folder / "analysis/decision.json")
        if not gate["complete"] or gate["failures"]:
            raise ValueError("complete numerical/timing predecessor required")
        if folder == OPERATOR and not gate["uniform_feature_screen_passed"]:
            raise ValueError("uniform-only operator gate must pass")
        path = folder / "analysis/decision.json"
        bound[str(path)] = sha(path)
    for path in (DOC, PREVIOUS / "plan.json", OPERATOR / "config.json"):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        for method, arm in ARMS:
            canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
            path = PREVIOUS / f"workers/{index}/audited-{method}-{canonical}.json"
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            arms=ARMS,
            stage="consumed_uniform_split_eager_core_request_screen",
            uniform_core_root=str(root),
            resolved_arm_overrides=[
                dict(method=m, arm=a, overrides=overrides(m, a)) for m, a in ARMS
            ],
            warmups=5,
            repetitions=3,
            correctness_only=False,
            explicit_softmax_cast_pre_request_word_controls_required=True,
            graph_private_pool_reserved_bytes_allowed=0,
            attention_workspace_bound_bytes=384 * 1024,
            common_uniform_optimization_offered_to_every_method=True,
            adaptive_layouts_use_unchanged_original_eager=True,
            exact_same_preconversion_packet_and_128_parent_logits=True,
            initial_and_final_storage_identical_peak_includes_attention_workspace=True,
            no_selection_precision_history_or_threshold_changes=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def pre_request_controls(plan):
    key = plan["uniform_core_root"]
    if key in _checked:
        return
    from mosaickv.evaluation.eager_core_operator import controls

    records = dict(
        source=controls(reuse_views=True, explicit_cast=True),
        shorter_baseline=controls(
            lengths=(301, 305, 365, 369, 427, 431, 301),
            capacity=431,
            reuse_views=True,
            explicit_cast=True,
        ),
    )
    if any(
        case["graph_pool_reserved_bytes_NOT_active_tensor_storage"]
        for value in records.values()
        for case in value["cases"]
    ):
        raise RuntimeError("pre-request explicit cast retained private graph pool storage")
    write_once(
        Path(key) / f"pre_request/{os.environ['SLURM_ARRAY_TASK_ID']}.json",
        dict(
            complete=True,
            records=records,
            before_first_comparison_request=True,
            model_weights_already_loaded=True,
            independent_quality_N=0,
        ),
    )
    _checked.add(key)


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    pre_request_controls(plan)
    canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
    result = OLD_EXECUTE(
        adapter, refs, messages, profile, plan, method, canonical, record, **kwargs
    )
    info = result["uniform_attention_core"]
    if info["requested"] is not record["uniform_attention_core"]:
        raise RuntimeError("declared uniform core did not engage")
    if info["requested"]:
        if not info["graphs_retired"] or info["retained_tensor_owners"] or info["aborted"]:
            raise RuntimeError("complete attention graph retirement required")
        if info["eligible"]:
            if (
                info["calls"] != 4064
                or info["captures"] != 254
                or info["replays"] != 8128
                or info["warm_calls"] != 508
                or info["view_builds"] != 127
                or info["view_reuses"] != 8001
                or not 0 < info["workspace_bytes"] <= 384 * 1024
                or info["extra_KV_buffer_bytes"]
                or info["capture_owner_checks"] != (254 if kwargs.get("audit", False) else 0)
            ):
                raise RuntimeError(
                    "complete uniform attention replay and charged workspace required"
                )
        elif info["calls"] or info["captures"] or info["workspace_bytes"]:
            raise RuntimeError("ineligible adaptive layout must use original zero-graph eager")
    return result


def summary(rows, audited):
    length = audited["mosaickv_exact", "joint-async-handoff"]["prefill_storage"]["logical_lengths"][
        0
    ]
    if length not in (600, 604):
        raise ValueError("only two same consumed inputs permitted")
    index = 0 if length == 600 else 1
    for (method, arm), value in audited.items():
        canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
        anchor = driver.read(PREVIOUS / f"workers/{index}/audited-{method}-{canonical}.json")
        workspace = value["uniform_attention_core"].get("workspace_bytes", 0)
        if any(value[k] != anchor[k] for k in driver.STORAGE[:2]):
            raise RuntimeError("attention core changed persistent cache storage")
        if value[driver.STORAGE[2]] > anchor[driver.STORAGE[2]] + workspace:
            raise RuntimeError("decode-owned peak increase exceeds explicit attention workspace")
    return dict(
        base.ORIGINAL_SUMMARY(rows, audited),
        physical_storage={
            f"{m}:{a}": {k: v[k] for k in driver.STORAGE} for (m, a), v in audited.items()
        },
        attention_core={f"{m}:{a}": v["uniform_attention_core"] for (m, a), v in audited.items()},
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
        no_fresh_or_larger_campaign_authorized=True,
    )


def finish_configuration():
    base.control.CANDIDATE_ARM = "joint-async-uniform-core"
    base.control.VARIANT = "consumed_uniform_split_eager_core_request_screen"
    base.control.JOB_NAME = "mk-uniform-core"
    prior.prior.OLD_MAIN()


def main():
    parent.submit = submit
    prior.prepare, prior.execute, prior.summary, prior.overrides = (
        prepare,
        execute,
        summary,
        overrides,
    )
    prior.SCRIPT, prior.DOC, prior.ARMS = SCRIPT, DOC, ARMS
    prior.finish_configuration = finish_configuration
    prior.main()


if __name__ == "__main__":
    main()
