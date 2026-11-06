#!/usr/bin/env python3
"""Twenty paired repetitions on consumed inputs, not independent task quality."""

import ast
import math
import subprocess
from pathlib import Path

import evicastkv_uniform_core_screen as prior
import numpy as np

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver, parent = prior.SOURCE, prior.driver, prior.parent
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/uniform-core-request-consumed-20261106-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_uniform_core_paired.sbatch"
DOC = SOURCE / "mosaickv/docs/uniform_core_paired.md"
RUNTIME = (
    "common_streaming_timing.py",
    "eager_core_operator.py",
    "uniform_core_transport.py",
    "sequential_kv_lease.py",
    "storage_runs.py",
)
OLD_SUMMARY, OLD_FINISH = prior.summary, prior.finish_configuration


def semantic(source):
    # Import sorting does not change the numerical implementation. All other
    # AST nodes, including validation, ownership and cleanup, remain frozen.
    class ImportOrder(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for field, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                if not any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in value):
                    continue
                normalized, group = [], []
                for item in [*value, None]:
                    if isinstance(item, (ast.Import, ast.ImportFrom)):
                        group.append(item)
                    else:
                        normalized.extend(sorted(group, key=lambda n: ast.dump(n)))
                        group.clear()
                        if item is not None:
                            normalized.append(item)
                setattr(node, field, normalized)
            return node

    tree = ImportOrder().visit(ast.parse(source))
    return ast.dump(tree, include_attributes=False)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable paired root and clean source required")
    gate = driver.read(PREVIOUS / "analysis/decision.json")
    if not gate["complete"] or gate["failures"] or not gate["four_percent_POINT_only"]:
        raise ValueError("both numerical and four-percent point screens must pass")
    old = driver.read(PREVIOUS / "plan.json")
    for filename in RUNTIME:
        path = "mosaickv/src/mosaickv/evaluation/" + filename
        frozen = subprocess.check_output(
            ["git", "show", f"{old['source_sha']}:{path}"], cwd=SOURCE, text=True
        )
        if semantic(frozen) != semantic((SOURCE / path).read_text()):
            raise ValueError("paired confirmation cannot change frozen runtime semantics")
    bound = dict(old["bound_files"])
    for path in (
        DOC,
        PREVIOUS / "plan.json",
        PREVIOUS / "analysis/decision.json",
        *(PREVIOUS / f"pre_request/{i}.json" for i in (0, 1)),
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            uniform_core_root=str(root),
            stage="consumed_uniform_core_twenty_paired_repetitions",
            warmups=5,
            repetitions=20,
            bootstrap_replicates=10000,
            family_maximum_engineering_contrasts=64,
            acceptance_adjusted_latency_ratio_upper=0.96,
            intervals_over_repetitions_NOT_independent_examples=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = prior.submit(root, "cpu")
    release = prior.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def paired_bounds(rows, seed):
    arms = prior.ARMS
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in arms for t in range(-5, 20)
    }:
        raise ValueError("all five warmups and twenty paired trials required")
    arrays = {}
    for method, arm in arms:
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (method, arm) and not r["warmup"]),
            key=lambda r: r["trial"],
        )
        values = [r["measurement"]["total_wall_seconds"] for r in selected]
        if len(values) != 20 or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("twenty finite paired request times required")
        arrays[method, arm] = np.asarray(values)
    candidate = arrays["mosaickv_exact", "joint-async-uniform-core"]
    draws = np.random.default_rng(seed).integers(0, 20, size=(10000, 20))
    tail = 0.05 / (2 * 64)
    comparisons = []
    for (method, arm), values in arrays.items():
        if arm == "joint-async-uniform-core":
            continue
        ratios = candidate[draws].mean(1) / values[draws].mean(1)
        interval = np.quantile(ratios, [tail, 1 - tail]).tolist()
        differences = candidate[draws].mean(1) - values[draws].mean(1)
        comparisons.append(
            dict(
                method=method,
                arm=arm,
                mean_ratio=float(candidate.mean() / values.mean()),
                adjusted_ratio_CI=interval,
                adjusted_difference_seconds_CI=np.quantile(differences, [tail, 1 - tail]).tolist(),
                equal_or_faster=bool(interval[1] <= 1),
                four_percent_faster=bool(interval[1] <= 0.96),
            )
        )
    prefix = [r for r in comparisons if r["method"] == "prefixkv_reimpl"]
    return dict(
        comparisons=comparisons,
        bootstrap_replicates=10000,
        family_maximum_engineering_contrasts=64,
        formal_latency_acceptance_against_both_Prefix_controls=all(
            r["four_percent_faster"] for r in prefix
        ),
        equal_or_faster_against_both_Prefix_controls=all(r["equal_or_faster"] for r in prefix),
        intervals_over_repetitions_NOT_independent_examples=True,
        does_NOT_certify_all_baselines_or_benchmark_quality=True,
    )


def summary(rows, audited):
    result = OLD_SUMMARY(rows, audited)
    length = audited["mosaickv_exact", "joint-async-handoff"]["prefill_storage"]["logical_lengths"][
        0
    ]
    result.update(paired_bounds(rows, 20261106 + (length == 604)))
    return result


def finish():
    prior.base.control.SCREEN_REPETITIONS = 20
    OLD_FINISH()


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    for index in (0, 1):
        proof = root / f"pre_request/{index}.json"
        if not proof.exists() or not driver.read(proof)["complete"]:
            failures.append(f"missing complete pre-request numerical proof {index}")
    complete = not failures and len(rows) == 2
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=complete,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            formal_latency_acceptance_against_both_Prefix_controls=complete
            and all(
                r["formal_latency_acceptance_against_both_Prefix_controls"]
                and r["smaller_at_all_three_decode_owned_boundaries"]
                for r in rows
            ),
            all_baseline_superiority_established=False,
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            no_fresh_or_larger_campaign_authorized=True,
        ),
    )


def main():
    prior.base.base.analyze = analyze
    prior.prepare, prior.summary, prior.finish_configuration = prepare, summary, finish
    prior.SCRIPT, prior.DOC, prior.GPU_MINUTES = SCRIPT, DOC, 40
    prior.main()


if __name__ == "__main__":
    main()
