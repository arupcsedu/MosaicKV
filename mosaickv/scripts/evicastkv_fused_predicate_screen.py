#!/usr/bin/env python3
"""Same-packet request check only after the exact-predicate operator gate."""

import ast
import subprocess
from pathlib import Path

import evicastkv_async_text_native as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base, SOURCE, parent, driver = prior.base, prior.SOURCE, prior.parent, prior.driver
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/async-text-native-consumed-20261106-launch-retry1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/fused-predicate-operator-20261106-cpu-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_fused_predicate_screen.sbatch"
DOC = SOURCE / "mosaickv/docs/fused_predicate_screen.md"
ARMS = (
    ("mosaickv_exact", "joint-async-handoff"),
    ("mosaickv_exact", "joint-async-fused-predicate"),
    ("prefixkv_reimpl", "graph-history-graph"),
    ("prefixkv_reimpl", "graph-original-history-graph"),
)
OLD_OVERRIDES, OLD_EXECUTE = prior.overrides, prior.execute


def arithmetic_contract(source):
    nodes = [
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name in ("_mask", "validate"))
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "LIMIT" for t in node.targets)
        )
    ]
    if len(nodes) != 3:
        raise ValueError("complete bounded operator arithmetic contract required")
    return tuple(ast.dump(node, include_attributes=False) for node in nodes)


def overrides(method, arm):
    canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
    return dict(
        OLD_OVERRIDES(method, canonical),
        fused_transport_predicate=arm == "joint-async-fused-predicate"
        or method != "mosaickv_exact",
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for folder in (PREVIOUS, OPERATOR):
        gate = driver.read(folder / "analysis/decision.json")
        if not gate["complete"] or gate["failures"]:
            raise ValueError(
                "complete numerical/timing and exact-predicate operator proofs required"
            )
        if folder == OPERATOR and not gate["operator_screen_passed"]:
            raise ValueError("positive preregistered operator performance gate required")
        if folder == OPERATOR:
            relative = "mosaickv/src/mosaickv/evaluation/fused_transport_predicate.py"
            frozen = subprocess.check_output(
                ["git", "show", f"{gate['source_sha']}:{relative}"], cwd=SOURCE, text=True
            )
            if arithmetic_contract(frozen) != arithmetic_contract((SOURCE / relative).read_text()):
                raise ValueError("operator kernel, input validation or bound changed after proof")
        path = folder / "analysis/decision.json"
        bound[str(path)] = sha(path)
    for path in (DOC, PREVIOUS / "plan.json", OPERATOR / "config.json"):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        for method, arm in ARMS:
            old_arm = "joint-async-handoff" if method == "mosaickv_exact" else arm
            path = PREVIOUS / f"workers/{index}/audited-{method}-{old_arm}.json"
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            arms=ARMS,
            stage="consumed_exact_fused_correction_predicate_screen",
            resolved_arm_overrides=[
                dict(method=m, arm=a, overrides=overrides(m, a)) for m, a in ARMS
            ],
            correctness_only=False,
            warmups=5,
            repetitions=3,
            exact_same_preconversion_packet_and_128_parent_logits=True,
            exact_same_initial_final_decode_owned_peak_bytes=True,
            original_canonical_dynamic_nonzero_and_BF16_words=True,
            unchanged_finite_protected_and_close_commit=True,
            common_optimization_available_to_ineligible_baselines_as_truthful_noop=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    canonical = "joint-async-handoff" if method == "mosaickv_exact" else arm
    value = OLD_EXECUTE(adapter, refs, messages, profile, plan, method, canonical, record, **kwargs)
    info = value["fused_transport_predicate"]
    enabled = record["fused_transport_predicate"]
    expected = 64 if enabled and method == "mosaickv_exact" else 0
    if info["enabled"] is not enabled or info["calls"] != expected:
        raise RuntimeError("declared exact predicate did not engage truthfully")
    if enabled and (
        not info["closed_before_decode"]
        or info["retained_tensor_owners"]
        or not info["original_canonical_nonzero_and_BF16_words"]
        or not info["unchanged_finite_protected_and_close_commit"]
    ):
        raise RuntimeError("private predicate must retire without changing any original checks")
    return value


def summary(rows, audited):
    length = audited["mosaickv_exact", "joint-async-handoff"]["prefill_storage"]["logical_lengths"][
        0
    ]
    if length not in (600, 604):
        raise ValueError("only the two same consumed inputs permitted")
    index = 0 if length == 600 else 1
    for (method, arm), value in audited.items():
        old_arm = "joint-async-handoff" if method == "mosaickv_exact" else arm
        anchor = driver.read(PREVIOUS / f"workers/{index}/audited-{method}-{old_arm}.json")
        if any(value[key] != anchor[key] for key in driver.STORAGE):
            raise RuntimeError("exact predicate changed frozen physical cache storage")
    return dict(
        base.ORIGINAL_SUMMARY(rows, audited),
        exact_same_initial_final_decode_owned_peak_bytes=True,
        physical_storage={
            f"{m}:{a}": {key: value[key] for key in driver.STORAGE}
            for (m, a), value in audited.items()
        },
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
        no_fresh_or_larger_campaign_authorized=True,
    )


def finish_configuration():
    base.control.CANDIDATE_ARM = "joint-async-fused-predicate"
    base.control.VARIANT = "consumed_exact_fused_correction_predicate_screen"
    base.control.JOB_NAME = "mk-fused-predicate"
    prior.prior.OLD_MAIN()


def main():
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
