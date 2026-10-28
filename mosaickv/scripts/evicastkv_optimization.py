#!/usr/bin/env python3
"""One profiler-justified common-runtime optimization, independently gated."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import subprocess
from pathlib import Path

from evicastkv_analyze import collect, export_descriptive, pairing
from evicastkv_campaign import SOURCE, submit
from evicastkv_campaign import prepare as prepare_campaign
from evicastkv_evidence import group_manifest
from evicastkv_finalize import schedule as schedule_final
from evicastkv_packet import csv_once, profile_summary, read, schedule_watch, text_once

from mosaickv.evaluation.fox_campaign import clean_source, digest, gate_payload, sha, write_once
from mosaickv.evaluation.fox_optimization import budget_function_tree, without_unit_reuse
from mosaickv.evaluation.fox_statistics import paired_scalar_statistics

SEMANTIC_FIELDS = (
    "selected_blocks",
    "packed_slots",
    "packed_padding_kv_bytes",
    "packed_cache_all_slots_valid",
    "packed_prompt_capacity",
    "candidate_graph_routing",
    "selection_objective",
    "byte_budget_accounting",
    "graph_edges",
    "source_kv_layout",
    "tier_active_cache_lengths",
    "effective_method",
    "residual_kv_bytes",
)


def semantic_check(left, right):
    for field in (
        "input_sha",
        "processor_sha",
        "scorer_sha",
        "config_sha",
        "token_ids",
        "forward_steps",
        "initial_kv_bytes",
        "final_kv_bytes",
        "peak_kv_bytes",
    ):
        if left[field] != right[field]:
            raise ValueError(f"optimization changed {field}")
    lt, rt = read(left["trace_path"]), read(right["trace_path"])
    for field in SEMANTIC_FIELDS:
        if lt.get(field) != rt.get(field):
            raise ValueError(f"optimization changed trace/{field}")
    for field in ("heads", "provenance"):
        if lt.get("forecast_statistics", {}).get(field) != rt.get("forecast_statistics", {}).get(
            field
        ):
            raise ValueError(f"optimization changed forecast/{field}")
    return dict(
        input_sha=left["input_sha"],
        selected_blocks_sha256=digest(lt["selected_blocks"]),
        packed_slots_sha256=digest(lt["packed_slots"]),
        token_agreement=1.0,
        physical_bytes=left["initial_kv_bytes"],
        semantic_fields=list(SEMANTIC_FIELDS),
    )


def prepare(parent, root):
    clean_source(SOURCE)
    lock = read(parent / "protocol_lock.json")
    reference = Path(lock["source_root"])
    clean_source(reference, lock["source_sha"])
    gates = {}
    for stage in ("correctness", "scoring", "profiling"):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        if not path.exists() or read(path)["status"] != "passed":
            raise ValueError(f"measured optimization requires passed {stage}")
        gates[str(path)] = sha(path)
    root.mkdir(parents=True, exist_ok=False)
    evidence = root / "preparation"
    evidence.mkdir()
    ranges = profile_summary(parent, evidence)
    target = next(
        (r for r in ranges if r["name"] == "evicastkv/_dense_physical_cache_budget"), None
    )
    if target is None or target["observations"] < 1 or target["mean_inclusive_cpu_ms"] <= 0:
        raise ValueError("target has no completed measured hotspot evidence")
    path = reference / "mosaickv/src/mosaickv/backends/hf_runtime.py"
    optimized_path = SOURCE / "mosaickv/src/mosaickv/backends/hf_runtime.py"
    if ast.dump(budget_function_tree(path), include_attributes=False) != ast.dump(
        without_unit_reuse(budget_function_tree(optimized_path)), include_attributes=False
    ):
        raise ValueError("the slow reference branch was changed, not just optimized")
    contract = dict(
        changes=1,
        parent_root=str(parent),
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        prerequisite_gates=gates,
        reference_source_root=str(reference),
        reference_source_sha=lock["source_sha"],
        reference_runtime_sha256=sha(path),
        reference_function_sha256=digest(
            ast.dump(budget_function_tree(path), include_attributes=False)
        ),
        optimized_runtime_sha256=sha(SOURCE / "mosaickv/src/mosaickv/backends/hf_runtime.py"),
        target=target,
        profiler_report_sha256=sha(evidence / "profiler_hotspots.json"),
        classification="candidate_semantics_preserving_requires_equivalence",
        rationale="Skip only identical re-blockization of a validated immutable FullKVState; "
        "unchanged budget computation, mandatory policy, layout lattice and source tensors.",
        nonselected_hotspot="Exact-tier source-view bookkeeping is more invasive; "
        "no speculative rewrite.",
        statistical_family="secondary_optimization_four",
        family_size=4,
        method_order="rotating positions with alternating reference/optimized pair orientation",
        controls_changed=False,
        confirmation_authorized=False,
    )
    prepare_campaign(
        root,
        workloads=("coco",),
        scope_registration=lock["scope_registration"]["path"],
        optimization_protocol=contract,
    )
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "--no-ext-diff",
            lock["source_sha"],
            clean_source(SOURCE),
            "--",
            "mosaickv/src/mosaickv/backends/hf_runtime.py",
        ],
        cwd=SOURCE,
        text=True,
    )
    if not patch:
        raise ValueError("no production optimization is present")
    text_once(root / "optimization_diff.patch", patch)
    if (parent / "history-inventory.json").exists():
        write_once(root / "history-inventory.json", read(parent / "history-inventory.json"))
    submit(root, "smoke")
    submission = read(root / "jobs/smoke-submission.json")
    schedule_watch(root, "smoke", submission["analysis"])
    schedule_final(root, root, "smoke", read(root / "jobs/smoke-packet-submission.json")["job"])


def correctness_equivalence(root, rows, contract):
    parent = Path(contract["parent_root"])
    if sha(parent / "protocol_lock.json") != contract["parent_protocol_sha256"]:
        raise ValueError("parent changed")
    _spec, _workers, reference, errors = collect(parent, "correctness")
    if errors:
        raise ValueError("reference correctness is incomplete")
    lookup = {
        (r["sample_id"], r["method"], r["budget_bytes"]): r
        for r in reference
        if r["workload"] == "coco"
    }
    candidate = {(r["sample_id"], r["method"], r["budget_bytes"]): r for r in rows}
    if set(lookup) != set(candidate):
        raise ValueError("optimization changed validation input/arm cohort")
    reports = [
        dict(
            sample_id=k[0],
            method=k[1],
            budget_bytes=k[2],
            **semantic_check(lookup[k], candidate[k]),
        )
        for k in sorted(candidate)
    ]
    return reports


def factorial(root, rows):
    lock = read(root / "protocol_lock.json")
    groups = group_manifest(root)["groups"]["coco"]
    inputs = set(lock["cohorts"]["coco"]["timing"]["sample_ids"])
    indexed = {
        (r["sample_id"], r["method"], r["budget_bytes"], r["variant"], r["repeat"]): r for r in rows
    }
    summaries, contrasts, equivalence = [], [], []
    for method in ("prompt_attention_topk", "mosaickv_exact"):
        for mib in lock["budgets_mib"]["coco"]:
            cap = int(mib * 2**20)
            reference, candidate = {}, {}
            for sid in sorted(inputs):
                rrows, orows = [], []
                for repeat in range(lock["repeats"]):
                    a = indexed[(sid, method, cap, "reference-budget", repeat)]
                    b = indexed[(sid, method, cap, "reuse-budget", repeat)]
                    equivalence.append(
                        dict(
                            sample_id=sid,
                            method=method,
                            cap=cap,
                            repeat=repeat,
                            **semantic_check(a, b),
                        )
                    )
                    rrows.append(a["e2e_wall_seconds"])
                    orows.append(b["e2e_wall_seconds"])
                reference[sid], candidate[sid] = statistics.mean(rrows), statistics.mean(orows)
                summaries.append(
                    dict(
                        sample_id=sid,
                        method=method,
                        cap_mib=mib,
                        reference_mean=reference[sid],
                        optimized_mean=candidate[sid],
                        reference_repeat_stdev=statistics.stdev(rrows),
                        optimized_repeat_stdev=statistics.stdev(orows),
                        repeats=lock["repeats"],
                        initial_bytes=a["initial_kv_bytes"],
                        final_bytes=a["final_kv_bytes"],
                        peak_bytes=a["peak_kv_bytes"],
                    )
                )
            contrast = paired_scalar_statistics(
                reference,
                candidate,
                {sid: groups[sid] for sid in inputs},
                dict.fromkeys(inputs, "COCO"),
                resamples=10000,
                seed=lock["seed"],
                family_size=4,
            )
            contrasts.append(dict(method=method, cap_mib=mib, **contrast))
    return summaries, contrasts, equivalence


def analyze_optimization(root, stage):
    lock = read(root / "protocol_lock.json")
    contract = lock["optimization_protocol"]
    output = root / "analysis" / stage
    output.mkdir(parents=True, exist_ok=False)
    spec, workers, rows, errors = collect(root, stage)
    expected = sum(len(j["indices"]) * len(j["arms"]) for j in spec["jobs"]) * (
        lock["repeats"] if stage == "optimization" else 1
    )
    if len(rows) != expected:
        errors.append(
            dict(error="incomplete expected denominator", expected=expected, actual=len(rows))
        )
    try:
        paired = pairing(rows)
        write_once(output / "pairing.json", paired)
        if stage == "correctness" and not errors:
            write_once(
                output / "equivalence_report.json", correctness_equivalence(root, rows, contract)
            )
        elif stage == "optimization" and not errors:
            summaries, contrasts, equivalence = factorial(root, rows)
            csv_once(output / "before_after_runtime.csv", summaries)
            write_once(output / "paired_factorial.json", contrasts)
            write_once(output / "equivalence_report.json", equivalence)
            report = [
                "# Dense-budget state reuse: measured implementation factorial",
                "",
                "No objective, routing, precision, allocation or output-policy change. "
                "Five warmups and three counterbalanced repeats; 16 consumed timing inputs. "
                "FullKV is an uncompressed reference, not a matched-budget latency competitor.",
                "",
            ]
            report += [
                f"- {r['method']} / {r['cap_mib']} MiB: reference/optimized mean ratio "
                f"{r['speedup_ratio_of_means']:.6f}; adjusted CI {r['speedup_simultaneous_ci']}."
                for r in contrasts
            ]
            report += [
                "",
                "A null/regressive speedup is retained. This is not a graph-quality claim "
                "or independent confirmation. The unchanged reference campaign remains separate.",
            ]
            text_once(output / "optimization_report.md", "\n".join(report) + "\n")
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    write_once(output / "failures.json", errors)
    if rows:
        export_descriptive(root, stage, rows)
    gate = gate_payload(
        stage,
        source_sha=lock["source_sha"],
        protocol_sha=sha(root / "protocol_lock.json"),
        passed=not errors,
        tests=dict(
            expected=expected,
            actual=len(rows),
            workers_expected=len(spec["jobs"]),
            workers_completed=len(workers),
        ),
        blockers=errors,
    )
    write_once(output / f"GATE_{stage}.json", gate)
    successor = {
        "smoke": "representation_canary",
        "representation_canary": "correctness",
        "correctness": "optimization",
    }.get(stage)
    if not errors and successor:
        submit(root, successor)
        decision = dict(action="submitted_registered_successor", stage=successor)
    else:
        decision = dict(
            action="stop_failed_gate" if errors else "optimization_complete_no_confirmation"
        )
    write_once(output / "completion_decision.json", decision)
    print(json.dumps(dict(gate=gate, action=decision)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.root)


if __name__ == "__main__":
    main()
