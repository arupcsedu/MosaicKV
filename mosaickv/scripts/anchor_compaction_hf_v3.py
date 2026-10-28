#!/usr/bin/env python3
"""Frozen V3, two consumed inputs, unchanged HF guard; no automatic promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import attention_compaction_hf_gate as base
import yaml
from bounded_history_pilot import SOURCE, clean_source, digest, write_once

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.anchor_compaction_diagnostic import (
    FIT_PREFIXES,
    METHOD,
    VALUE_POLICY,
    VARIANT,
    construction_policy,
)
from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    plan_compaction_capacity,
)
from mosaickv.prototypes.constrained_compaction import fit_component

PROTOCOL = SOURCE / "mosaickv/configs/experiments/anchor_compaction_hf_v3.yaml"
BLOCKED = base.BLOCKED


def validate_protocol(protocol):
    base.validate_protocol(protocol, method=METHOD, dictionary="frozen_parent_39_positions")
    if (
        protocol["construction_variant"] != VARIANT
        or protocol["construction"] != construction_policy()
        or protocol["prompt_positions"] != 39
        or protocol["cap_bytes"] != 28 * 2**20
        or protocol["maximum_cpu_construction_seconds"] != 600
        or protocol["on_negative"] != "stop_no_automatic_successor"
        or protocol["on_positive"] != "design_review_only_not_fresh_cohort_or_paper_authorization"
    ):
        raise ValueError("frozen V3 construction, capacity or follow-up scope changed")


def prepare(root):
    import torch

    sha = clean_source()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.resolve().parents:
        raise ValueError("require a new immutable scratch GPU diagnostic root")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(protocol)
    bound = {str(PROTOCOL): digest(PROTOCOL)}
    cpu = Path(protocol["cpu_root"]).resolve()
    old_gpu = Path(protocol["consumed_gpu_parent"]).resolve()
    for path, expected in (
        (cpu / "plan.json", protocol["cpu_plan_sha256"]),
        (cpu / "analysis/report.json", protocol["cpu_report_sha256"]),
        (cpu / "analysis/decision.json", protocol["cpu_decision_sha256"]),
        (old_gpu / "plan.json", protocol["consumed_gpu_plan_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError("frozen CPU evidence or consumed parent changed")
        bound[str(path)] = expected
    cpu_plan = json.loads((cpu / "plan.json").read_text())
    evidence = json.loads((cpu / "analysis/report.json").read_text())
    decision = json.loads((cpu / "analysis/decision.json").read_text())
    if (
        evidence["status"] != "completed"
        or evidence["errors"]
        or not decision["useful_bounded_fit"]
        or not decision["fixed_case_complete"]
        or decision["selected_case"] != "anchor_preserving"
        or decision["selected_component"] != "joint"
        or decision["report_sha256"] != digest(cpu / "analysis/report.json")
    ):
        raise ValueError("registered V3 candidate did not pass the CPU screen")
    bound.update(cpu_plan["bound_files"])
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError("CPU study source/input hashes changed")
    parent = json.loads((old_gpu / "plan.json").read_text())
    core = dict(
        policy=parent["core"]["policy"],
        capacity_planning=parent["core"]["capacity_planning"],
        effective_method=METHOD,
        construction=construction_policy(),
        original_safety_parameter_source="hash_bound_consumed_gpu_parent",
    )
    capacity = plan_compaction_capacity(**core["capacity_planning"])
    if (
        capacity.prompt_positions != 39
        or core["capacity_planning"]["cap_bytes"] != 28 * 2**20
        or AttentionCompactionPolicy(**core["policy"]) != AttentionCompactionPolicy(True)
    ):
        raise ValueError("unchanged original core and 28 MiB physical capacity required")
    jobs = []
    for original in parent["jobs"]:
        index = original["index"]
        cfg_path = Path(original["config"])
        input_path = Path(original["input"])
        for path in (cfg_path, input_path):
            if digest(path) != parent["bound_files"][str(path)]:
                raise ValueError("consumed controls changed")
            bound[str(path)] = digest(path)
        cfg = load_config(cfg_path)
        payload = json.loads(input_path.read_text())
        if (
            config_sha256(cfg) != original["control_config_sha"]
            or cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "fp16"
            or cfg.execution.attention_implementation != "eager"
            or cfg.cache.physical_budget_mandatory_policy != "first_and_last"
            or set(payload)
            != {"task", "dataset_id", "dataset_revision", "split", "sample_id", "prompt", "media"}
        ):
            raise ValueError("require identical consumed label-redacted LLaVA controls")
        for media in payload["media"]:
            if digest(media["path"]) != media["sha256"]:
                raise ValueError("consumed media changed")
            bound[media["path"]] = media["sha256"]
        folder = old_gpu / f"workers/{index}"
        artifact_map = json.loads((folder / "artifact-hashes.json").read_text())
        path = folder / "constructed.pt"
        if digest(path) != artifact_map[str(path)]:
            raise ValueError("frozen dictionary source changed")
        bound[str(path)] = digest(path)
        rows = torch.load(path, map_location="cpu", weights_only=True)
        if len(rows) != 1024 or [(r["layer"], r["head"]) for r in rows] != [
            (layer, head) for layer in range(32) for head in range(32)
        ]:
            raise ValueError("frozen dictionary must contain all 32x32 routes")
        dictionary = [
            [list(rows[layer * 32 + head]["positions"]) for head in range(32)]
            for layer in range(32)
        ]
        if any(
            len(p) != 39 or p != sorted(set(p)) or p[0] != 0 for layer in dictionary for p in layer
        ):
            raise ValueError("invalid original 39-position dictionary")
        new_input = root / f"input-{index}.json"
        new_dictionary = root / f"dictionary-{index}.json"
        write_once(new_input, payload)
        write_once(new_dictionary, dictionary)
        bound[str(new_input)] = digest(new_input)
        bound[str(new_dictionary)] = digest(new_dictionary)
        jobs.append(
            dict(
                index=index,
                task=original["task"],
                config=str(cfg_path),
                input=str(new_input),
                dictionary=str(new_dictionary),
                control_config_sha=config_sha256(cfg),
            )
        )
    if [job["index"] for job in jobs] != [0, 1]:
        raise ValueError("only two consumed workers permitted")
    result = dict(
        source_sha=sha,
        protocol=protocol,
        core=core,
        capacity_planning=asdict(capacity),
        jobs=jobs,
        bound_files=bound,
        new_identities=0,
        **BLOCKED,
    )
    write_once(root / "plan.json", result)
    return result


def validate(root, plan_sha):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("V3 GPU plan changed")
    plan = json.loads((root / "plan.json").read_text())
    clean_source(plan["source_sha"])
    validate_protocol(plan["protocol"])
    if any(plan[key] is not False for key in BLOCKED):
        raise ValueError("larger campaigns remain blocked")
    for path, expected in plan["bound_files"].items():
        if digest(path) != expected:
            raise ValueError("V3 GPU bound source/input changed")
    return plan


def worker(root, plan_sha, index):
    plan = validate(root, plan_sha)
    return base.execute_worker(
        root,
        plan_sha,
        index,
        plan=plan,
        method=METHOD,
        construction_variant=VARIANT,
        run_prefix="anchor-compaction-hf-v3",
        evaluator_version="anchor-compaction-v3",
    )


def review_coefficients(rows, evidence, core, dictionary):
    """Independently replay first-16 fits; no later-prefix coefficient access."""
    import torch

    if not rows or len(rows) != sum(len(layer) for layer in dictionary):
        raise ValueError("missing V3 heads")
    fixed = [tuple(p) for layer in dictionary for p in layer]
    max_delta, max_value, admitted = 0.0, 0.0, 0
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        for j, row in enumerate(chunk):
            if (
                tuple(row["positions"]) != fixed[start + j]
                or not torch.equal(row["keys"], row["original_keys"])
                or not torch.equal(row["values"][[0, -1]], row["original_values"][[0, -1]])
                or (row["log_bias"][[0, -1]] != 0).any()
            ):
                raise ValueError("V3 changed original dictionary, keys, or mandatory entries")
        a = torch.stack(
            [
                evidence[r["layer"]]["prompt_probabilities"][r["head"]][
                    :FIT_PREFIXES, list(r["positions"])
                ]
                for r in chunk
            ]
        )
        v = torch.stack([r["original_values"] for r in chunk])
        y, h, n = [
            torch.stack([evidence[r["layer"]][name][r["head"], :FIT_PREFIXES] for r in chunk])
            for name in ("target", "generated_mass", "generated_numerator")
        ]
        protected = torch.zeros(v.shape[:2], dtype=torch.bool)
        protected[:, [0, -1]] = True
        fit = fit_component(a, v, y, h, n, protected, "joint", core, VALUE_POLICY)
        actual = torch.stack([r["values"] for r in chunk])
        bias = torch.stack([r["log_bias"] for r in chunk])
        if not torch.equal(actual, fit.values) or not torch.equal(bias, fit.log_bias):
            raise ValueError("stored V3 coefficients do not reproduce the frozen first-16 fit")
        norm = v.double().norm(dim=-1)
        delta = (actual.double() - v.double()).norm(dim=-1)
        if (delta > 0.1 * norm).any() or (actual.double().norm(dim=-1) > 4 * norm).any():
            raise ValueError("stored V3 delta or value norm exceeds unchanged limits")
        if (
            not torch.isfinite(actual).all()
            or not torch.isfinite(bias).all()
            or (bias < 0).any()
            or (bias.double().exp() > 64).any()
        ):
            raise ValueError("invalid stored V3 values or mass multipliers")
        max_delta = max(max_delta, float((delta / norm.clamp_min(1e-12)).max()))
        max_value = max(
            max_value, float((actual.double().norm(dim=-1) / norm.clamp_min(1e-12)).max())
        )
        admitted += int(fit.admitted.sum())
    return dict(
        first_16_fit_bitwise_reproduced=True,
        maximum_delta_norm_ratio=max_delta,
        maximum_value_norm_ratio=max_value,
        construction_accepted_heads=admitted,
    )


def analyze(root, plan_sha):
    import torch

    torch.set_num_threads(1)
    plan = validate(root, plan_sha)
    tasks = []
    errors = []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        try:
            completion = json.loads((folder / "completed.json").read_text())
            if (
                completion["status"] != "completed"
                or digest(completion["manifest"]) != completion["manifest_sha256"]
            ):
                raise ValueError("producer failed or manifest changed")
            manifest = json.loads(Path(completion["manifest"]).read_text())
            if (
                manifest["source"]["git_sha"] != plan["source_sha"]
                or manifest["source"]["git_dirty"]
                or manifest["execution"]["effective_method"] != METHOD
                or manifest["execution"]["plan_sha256"] != plan_sha
                or manifest["execution"]["frozen_parent_control_config_sha"]
                != job["control_config_sha"]
                or manifest["source"]["config_sha"]
                != config_sha256(
                    base.manifest_control_config(
                        load_config(job["config"]), plan["core"], method=METHOD
                    )
                )
                or manifest["artifacts"]["raw_output_sha"] != digest(folder / "diagnostic.json")
                or manifest["artifacts"]["log_sha"] != digest(folder / "artifact-hashes.json")
                or completion["diagnostic_sha256"] != digest(folder / "diagnostic.json")
            ):
                raise ValueError("wrong immutable source/method/config provenance")
            hashes = json.loads((folder / "artifact-hashes.json").read_text())
            for path, expected in hashes.items():
                if digest(path) != expected:
                    raise ValueError("V3 GPU artifact changed")
            raw = json.loads((folder / "diagnostic.json").read_text())
            result = raw["result"]
            if (
                raw["status"] != "completed"
                or result["effective_method"] != METHOD
                or result["construction_variant"] != VARIANT
                or result["coefficient_fit_prefixes"] != 16
                or result["prompt_positions"] != 39
                or result["constructed_heads"] != 1024
                or any(row["cap_bytes"] != 28 * 2**20 for row in result["storage"])
            ):
                raise ValueError("unregistered V3 construction or cache budget")
            rows = torch.load(folder / "constructed.pt", map_location="cpu", weights_only=True)
            evidence = torch.load(folder / "evidence.pt", map_location="cpu", weights_only=True)
            coefficients = review_coefficients(
                rows,
                evidence,
                AttentionCompactionPolicy(**plan["core"]["policy"]),
                json.loads(Path(job["dictionary"]).read_text()),
            )
            if coefficients["construction_accepted_heads"] != result["construction_accepted_heads"]:
                raise ValueError("recorded admission count differs from frozen CPU replay")
            logits = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)
            reviewed = base.review_result(result, logits, plan["protocol"])
            tasks.append(
                dict(worker=job["index"], task=job["task"], coefficients=coefficients, **reviewed)
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            errors.append(
                dict(
                    worker=job["index"],
                    error=f"{type(error).__name__}: {error}",
                    diagnostic_path=str(folder / "diagnostic.json"),
                    phase_events=str(folder / "phase_events.jsonl"),
                )
            )
    passed = len(tasks) == 2 and not errors and all(task["diagnostic_passed"] for task in tasks)
    report = dict(
        status="completed" if len(tasks) == 2 and not errors else "incomplete",
        tasks=tasks,
        errors=errors,
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        effective_method=METHOD,
        diagnostic_passed=passed,
        benchmark_quality_established=False,
        no_new_inference=True,
        action="design_review_only_no_automatic_promotion"
        if passed
        else "negative_diagnostic_no_automatic_successor",
        **BLOCKED,
    )
    write_once(root / "analysis/report.json", report)
    write_once(
        root / "analysis/decision.json",
        dict(
            action=report["action"],
            report_sha256=digest(root / "analysis/report.json"),
            gpu_submission_authorized=False,
            **BLOCKED,
        ),
    )
    print(json.dumps(report, allow_nan=False))


def submit(root):
    plan = prepare(root)
    sha = digest(root / "plan.json")
    validate(root, sha)
    return base.submit_prepared(
        root,
        plan,
        sha,
        worker_script="anchor_compaction_hf_v3.sbatch",
        analysis_script="anchor_compaction_hf_analysis_v3.sbatch",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "submit", "worker", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.root.resolve(), args.plan_sha, args.index)
    elif args.mode == "analyze":
        analyze(args.root.resolve(), args.plan_sha)
    else:
        globals()[args.mode](args.root.resolve())
