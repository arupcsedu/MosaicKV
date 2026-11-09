#!/usr/bin/env python3
"""Verify the same lossless scale transport on all four frozen local baselines."""

import json
from pathlib import Path
from unittest.mock import patch

import exact_scale_replay_study as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/early-exact-headers-20261108-v1")
METHODS = ("prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl")
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/common_baseline_headers.sbatch"
base.EXTRA_TESTS = ("test_streaming_baselines.py", "test_common_baseline_headers.py")


def arms():
    return [method + ":" + mode for method in METHODS for mode in ("control", "headers")]


def prepare(root):
    import evicastkv_uniform_all_baselines as legacy

    if root.exists():
        raise FileExistsError("new immutable root required")
    gate = base.common.read(PARENT / "analysis/decision.json")
    if not gate["complete"] or not gate["all_initial_final_peak_savings"]:
        raise ValueError("complete common-source header evidence required")
    p = base.common.read(PARENT / "plan.json")
    frozen = base.common.read(legacy.PARENT / "plan.json")
    options_by_input = {}
    for i in (0, 1):
        path = legacy.screen.parent.PARENT / f"workers/{i}/study.json"
        records = base.common.read(path)["records"]
        p["bound_files"][str(path)] = sha(path)
        with patch.object(legacy, "METHODS", METHODS):
            local = {
                method: legacy.configurations(records, method, frozen)[method, "graph"]
                for method in METHODS
            }
        options_by_input[str(i)] = local
    for path in (
        legacy.PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PARENT / "plan.json",
    ):
        p["bound_files"][str(path)] = sha(path)
    p["controls"]["options"] = options_by_input["0"]
    p.update(
        source_sha=clean_source(SOURCE),
        study="common_exact_headers_four_baseline_compatibility_v1",
        measurement_type="consumed_local_baseline_lossless_transport_parity",
        arms=arms(),
        execution_methods={arm: arm.split(":")[0] for arm in arms()},
        baseline_labels="Prompt Top-K or local precision-variant reimplementations; NOT official",
        comparison="within each baseline: identical selections/generation, common headers only",
        baseline_options_by_input=options_by_input,
        per_input_policy_provenance="unchanged historical cost-only record, identical within pair",
        cross_method_byte_matched_claim=False,
        clean_latency_comparison=False,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = base.common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=base.common.submit(root, "after_cpu", cpu))
    )
    print(json.dumps(base.common.read(root / "submission.json")), flush=True)


def model(root, p, folder, i):
    import copy

    from compact_hybrid_study import fidelity

    from mosaickv.evaluation.exact_scale_replay import use

    references = {}

    def context(stack, arm):
        return stack.enter_context(use()) if arm.endswith(":headers") else None

    def compare(actual, unused, plan, arm):
        method, mode = arm.split(":")
        if mode == "control":
            references[method] = actual.clone()
        if method not in references:
            raise ValueError("missing identical-baseline unoptimized reference")
        proof = fidelity(actual, references[method], strict=True, p=plan)
        proof.update(
            within_same_baseline_transport_parity=True,
            control_is_reference_measurement_NOT_FullKV_parity=mode == "control",
            reference_path=str(folder / f"{method}:control.logits.pt"),
        )
        return proof

    base.common.extra_model_context = context
    base.common.assess_fidelity = compare
    p = copy.deepcopy(p)
    p["controls"]["options"] = p["baseline_options_by_input"][str(i)]
    base.common.model(root, p, folder, i)


def analyze(root, p):
    import gzip

    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    errors, pairs = [], []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if [r["arm"] for r in rows] != arms():
            errors.append(dict(input=i, error="incomplete/duplicate baseline arms"))
        by_arm = {r["arm"]: r for r in rows}
        for method in METHODS:
            local = [by_arm.get(method + ":" + mode) for mode in ("control", "headers")]
            if any(
                not r or r["status"] != "completed" or not r.get("numerically_admissible")
                for r in local
            ):
                errors.append(
                    dict(
                        input=i,
                        method=method,
                        error="failed baseline transport row",
                        details=[r.get("error") if r else "missing" for r in local],
                    )
                )
                continue
            a, b = local
            for r in local:
                path = Path(r["ledger_path"])
                with gzip.open(path, "rt") as stream:
                    if sha(path) != r["ledger_sha"] or not valid_whole_request_certificate(
                        json.load(stream)
                    ):
                        errors.append(
                            dict(input=i, method=method, error="ownership evidence invalid")
                        )
            report = b.get("common_header")
            engaged = bool(report and any(r.get("enabled") for r in report))
            pairs.append(
                dict(
                    input=i,
                    method=method,
                    headers_engaged=engaged,
                    all_128_logits_bitwise=b["fidelity"]["all_128_logit_words_bitwise"],
                    initial_before_bytes=a["initial_GPU_KV_bytes"],
                    initial_after_bytes=b["initial_GPU_KV_bytes"],
                    final_before_bytes=a["final_GPU_KV_bytes"],
                    final_after_bytes=b["final_GPU_KV_bytes"],
                    peak_before_bytes=a["whole_request_KV_plus_workspace_peak_bytes"],
                    peak_after_bytes=b["whole_request_KV_plus_workspace_peak_bytes"],
                    common_header=report,
                )
            )
            if (
                b["initial_GPU_KV_bytes"] > a["initial_GPU_KV_bytes"]
                or b["final_GPU_KV_bytes"] > a["final_GPU_KV_bytes"]
            ):
                errors.append(
                    dict(input=i, method=method, error="header path expanded live persistent KV")
                )
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    write_once(
        folder / "decision.json",
        dict(
        complete=len(pairs) == 2 * len(METHODS) and not errors,
            errors=errors,
            pairs=pairs,
            comparative_quality_or_latency_superiority_established=False,
            cross_method_byte_matching_established=False,
            fresh_cohort_authorized=False,
        ),
    )
    lines = [
        "# Common lossless headers on frozen local baseline precision variants",
        "",
        "Same consumed inputs; within-method before/after transport check, not a paper comparison.",
        "",
        "| Input | Method | Initial before/after MiB | Final before/after MiB | "
        "128 logits identical |",
        "|---|---|---|---|---|",
    ]
    for r in pairs:
        lines.append(
            f"| {r['input']} | {r['method']} | "
            f"{r['initial_before_bytes'] / 2**20:.6f} / {r['initial_after_bytes'] / 2**20:.6f} | "
            f"{r['final_before_bytes'] / 2**20:.6f} / {r['final_after_bytes'] / 2**20:.6f} | "
            f"{r['all_128_logits_bitwise']} |"
        )
    lines += [
        "",
        "All reimplementations remain local and their frozen precision variants are "
        "not official baselines. No early source-header path is applied to incompatible "
        "baseline constructors; their compatible affine headers get the same bit-exact "
        "codec and charged replay buffers. Negative/unsupported cases stay explicit.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    base.prepare, base.model, base.analyze = prepare, model, analyze
    base.main()
