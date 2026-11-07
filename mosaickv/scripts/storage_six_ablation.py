#!/usr/bin/env python3
"""Bounded, actual-allocation-matched ablation on two consumed images only."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.storage_six_overlay import COMPONENT_MODES, OverlayPolicy

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3")
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/storage-six-overlay-20261106-v2")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
ARMS = ("full_kv", "parent", *COMPONENT_MODES)


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable component plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for path, value in plan["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError(f"immutable component input changed: {path}")
    return plan


def component_policies(proposal_study=False, allocation_study=False):
    base = OverlayPolicy(record_phase_timings=True)
    if allocation_study:
        improved = replace(
            base,
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
        )
        all_adaptive = replace(
            improved,
            utility_signal="parent_reconstruction_error",
            demand_weighted_coverage=True,
            routing_signal="parent_error_opportunity",
        )
        return {
            "full": asdict(base),
            "individual_delta": asdict(improved),
            "individual_delta_fast_graph": asdict(
                replace(improved, graph_accumulation="segmented_tensor_topk")
            ),
            "error_utility": asdict(
                replace(improved, utility_signal="parent_reconstruction_error")
            ),
            "demand_graph": asdict(replace(improved, demand_weighted_coverage=True)),
            "opportunity_route": asdict(
                replace(improved, routing_signal="parent_error_opportunity")
            ),
            "all_adaptive": asdict(all_adaptive),
            **{
                name: asdict(replace(all_adaptive, component_mode=name))
                for name in (
                    "no_forecast",
                    "no_graph",
                    "no_value",
                    "modular_selection",
                    "exact_members",
                    "no_repair",
                    "forced_repair",
                )
            },
            "parent_reserved": asdict(replace(base, component_mode="parent_reserved")),
        }
    if not proposal_study:
        return {mode: asdict(replace(base, component_mode=mode)) for mode in COMPONENT_MODES}
    return {
        "full": asdict(base),
        "parent_reserved": asdict(replace(base, component_mode="parent_reserved")),
        "individual_forecast": asdict(replace(base, forecast_distribution="individual_queries")),
        "delta_prototype": asdict(replace(base, prototype_representation="parent_residual_delta")),
        "individual_delta": asdict(
            replace(
                base,
                forecast_distribution="individual_queries",
                prototype_representation="parent_residual_delta",
            )
        ),
        "exact_members": asdict(replace(base, component_mode="exact_members")),
        "forced_repair": asdict(replace(base, component_mode="forced_repair")),
    }


def prepare(root, proposal_study=False, allocation_study=False):
    proposal_study = proposal_study or allocation_study
    if root.exists():
        raise FileExistsError("new immutable component output root required")
    source_sha = clean_source(SOURCE)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "e2e1208", source_sha], cwd=SOURCE, check=True
    )
    parent = read(PARENT / "plan.json")
    options = read(PARENT / "options.json")["options"]
    inputs = read(PREVIOUS / "inputs.json")
    if len(inputs) != 2 or any(set(s) != {"sample_id", "prompt", "media"} for s in inputs):
        raise ValueError("exactly the two previously consumed, label-free inputs required")
    paths = [PARENT / n for n in ("plan.json", "options.json", "ledger.json")]
    paths += [PREVIOUS / "inputs.json", Path(parent["profile_path"])]
    paths += [Path(m["path"]) for s in inputs for m in s["media"]]
    root.mkdir(parents=True)
    write_once(root / "inputs.json", inputs)
    if proposal_study:
        import torch

        from mosaickv.evaluation.residual_delta_prototype import study

        torch.set_num_threads(2)
        previous_ablation = Path(
            "/scratch/djy8hg/runs/evicastkv/storage-six-component-ablation-20261106-v1"
        )
        proof = {}
        for worker in (0, 1):
            row_path = next(
                p for p in (previous_ablation / f"workers/{worker}").glob("[0-9][0-9]-full.json")
            )
            row = read(row_path)
            capture = row["evaluation_capture"]
            if sha(Path(capture["path"])) != capture["sha256"]:
                raise ValueError("reviewed consumed precision capture changed")
            result = study(torch.load(capture["path"], weights_only=True, map_location="cpu"))
            variants = result["variants"]
            for boundary in ("first_prefix_half_mean_error", "later_prefix_half_mean_error"):
                if not variants["prototype_only"][boundary] < variants["parent"][boundary]:
                    raise ValueError(
                        "residual prototype has no consistent local improvement; stop GPU follow-up"
                    )
            proof[str(worker)] = result
            paths += [row_path, Path(capture["path"])]
        write_once(
            root / "cpu_precision_feasibility.json",
            dict(
                source_sha=source_sha,
                consumed_only=True,
                results=proof,
                no_benchmark_answers_used=True,
            ),
        )
        paths += [root / "cpu_precision_feasibility.json"]
    with (root / "cpu.txt").open("x") as stream:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_storage_component_gate.py",
                "mosaickv/tests/unit/test_residual_delta_prototype.py",
                "mosaickv/tests/unit/test_storage_error_aware.py",
                "mosaickv/tests/unit/test_opportunity_demand.py",
                "mosaickv/tests/unit/test_bounded_probe_horizon.py",
                "mosaickv/tests/unit/test_streaming_source_constructor.py",
                "mosaickv/tests/unit/test_prototype_rope.py",
                "mosaickv/tests/unit/test_six_mechanism_gate.py",
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json", dict(passed=result.returncode == 0, log_sha=sha(root / "cpu.txt"))
    )
    if result.returncode:
        raise RuntimeError("component CPU gate failed; no GPU submission")
    paths += [root / "inputs.json", root / "cpu.json"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            parent=parent,
            options=options,
            bound_files={str(p): sha(p) for p in paths},
            arms=("full_kv", "parent", *component_policies(proposal_study, allocation_study)),
            component_policies=component_policies(proposal_study, allocation_study),
            component_reference_arm="all_adaptive" if allocation_study else "full",
            overlay_policy=asdict(OverlayPolicy(record_phase_timings=True)),
            output_predictions=128,
            guard_predictions=32,
            guard_kl=0.02,
            guard_logit=1.0,
            classification="same_consumed_two_image_matched_overlay_component_diagnostic",
            labels_accessible_to_planner=False,
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
            matching_rule="actual_initial_final_counter_storage_and_all_auxiliary_tensors",
            reservation_control="allocated_parent_reserved_not_efficient_parent_benchmark",
            replacement_authorized_only_after_useful_component_readout=True,
            registered_interpretation=(
                "No component benefit is inferred from execution alone. Local readout changes do "
                "not establish answer quality. Unmatched final/peak points are disclosed. "
                "No automatic replacement or new-cohort GPU campaign from a two-image result."
            ),
        ),
    )
    import six_mechanism_feasibility as infra

    infra.SOURCE = SOURCE
    infra.SHELL = SOURCE / "mosaickv/slurm/storage_six_ablation.sbatch"
    infra.launch(root, "components")


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_component_gate import logit_diagnostics, storage_match
    from mosaickv.evaluation.storage_six_overlay import integrated_storage
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("identical deterministic CUDA launch policy required")
    torch.set_num_threads(12)
    torch.manual_seed(plan["parent"]["protocol"]["seed"])
    torch.use_deterministic_algorithms(True)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    cfg = RunConfig.from_mapping(plan["parent"]["config"])
    rows, errors, values_by_arm = [], [], {}
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=plan["source_sha"],
                source_dirty=False,
                config_sha=digest(plan),
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                environment=environment(),
                cuda=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                visible_gpu_count=1,
                backend="huggingface",
                attention="eager",
                precision="bf16",
                seed=plan["parent"]["protocol"]["seed"],
                measurement_type="validation_smoke",
                **scheduler_identity({}),
                parent_ledger_sha=sha(PARENT / "ledger.json"),
                reference_captions_available_during_inference=False,
            ),
        )
        inputs = read(root / "inputs.json")[index]
        transport = dict(plan["parent"], options=plan["options"])
        options = campaign.request_options(
            plan["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )
        # Cold-start warmup is identical and not a measured comparison arm.
        warm, _ = campaign.execute(adapter, inputs, transport, "mosaickv_exact", audit=False)
        del warm
        # Reverse component order on worker 1; fixed controls always first.
        policies = plan.get("component_policies", component_policies())
        component_names = tuple(policies)
        components = component_names if index == 0 else tuple(reversed(component_names))
        for arm in ("full_kv", "parent", *components):
            gc.collect()
            row = dict(
                arm=arm,
                sample_id=inputs["sample_id"],
                status="failed",
                manifest_path=str(folder / "manifest.json"),
            )
            try:
                if arm in policies:
                    policy = OverlayPolicy(
                        **dict(
                            policies[arm],
                            layers=tuple(policies[arm]["layers"]),
                        )
                    )

                    def capture(payload, record=row):
                        path = (
                            folder / f"{record['arm']}-layer-{payload['layer']}-evaluation-only.pt"
                        )
                        if path.exists():
                            raise FileExistsError("immutable routed capture already exists")
                        torch.save(payload, path)
                        record["evaluation_capture"] = dict(path=str(path), sha256=sha(path))

                    with integrated_storage(
                        adapter,
                        policy,
                        reader_options=options,
                        diagnostic_capture=capture if arm in ("full", "all_adaptive") else None,
                    ) as proof:
                        values, detail = campaign.execute(
                            adapter, inputs, transport, "mosaickv_exact", audit=False
                        )
                    row["mechanism_evidence"] = proof
                    if detail["prefill_storage"]["original_KV_tensor_objects_alive"] != 0:
                        raise RuntimeError("original dense KV owner survived")
                    if not proof["all_residuals_pinned"] or proof["extra_gpu_bytes"] != 6713:
                        raise RuntimeError("registered real matched auxiliary allocations changed")
                    if arm == "no_graph" and any(
                        e["graph_edges"] for e in proof["layers"].values()
                    ):
                        raise RuntimeError("no-graph ablation stored evidence edges")
                    if arm == "no_repair" and proof["repair_events"]:
                        raise RuntimeError("disabled repair triggered")
                    if arm == "forced_repair" and len(proof["repair_events"]) != 1:
                        raise RuntimeError("actual controlled residual restoration required")
                    if arm == "parent_reserved" and not torch.equal(
                        values.cpu(), values_by_arm["parent"]
                    ):
                        raise RuntimeError("inactive reservation control changed parent logits")
                    detail["effective_method"] = f"evicastkv_storage_six_component_{arm}"
                else:
                    values, detail = campaign.execute(
                        adapter,
                        inputs,
                        transport,
                        "full_kv" if arm == "full_kv" else "mosaickv_exact",
                        audit=False,
                    )
                actual = values.cpu()
                values_by_arm[arm] = actual
                if arm != "full_kv":
                    row["fullkv_readout"] = logit_diagnostics(values_by_arm["full_kv"], actual)
                if arm in policies:
                    row["parent_guard"] = logit_diagnostics(values_by_arm["parent"], actual)
                    row["parent_guard_passed"] = (
                        row["parent_guard"]["maximum_kl"] <= plan["guard_kl"]
                        and row["parent_guard"]["maximum_logit_difference"] <= plan["guard_logit"]
                        and row["parent_guard"]["token_agreement"] == 1.0
                    )
                logits_path = folder / f"{arm}.logits.pt"
                if logits_path.exists():
                    raise FileExistsError("immutable saved logits already exist")
                torch.save(actual, logits_path)
                row.update(
                    status="completed",
                    detail=detail,
                    logits_path=str(logits_path),
                    logits_sha256=sha(logits_path),
                )
                del values
            except Exception:
                row["error"] = traceback.format_exc()
                errors.append(row["error"])
            write_once(folder / f"{len(rows):02}-{arm}.json", row)
            rows.append(row)
            # Preserve every attempted row; continue independent ablations.
        completed = {r["arm"]: r for r in rows if r["status"] == "completed"}
        if all(a in completed for a in component_names):
            matched = storage_match([completed[a] for a in component_names])
            write_once(folder / "storage_matching.json", matched)
            if not torch.equal(values_by_arm["exact_members"], values_by_arm["forced_repair"]):
                errors.append(
                    "preloaded exact-members control differs from forced original restoration"
                )
            if "individual_delta_fast_graph" in values_by_arm and not torch.equal(
                values_by_arm["individual_delta_fast_graph"], values_by_arm["individual_delta"]
            ):
                errors.append("segmented graph execution changed a final model logit bit")
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            row_count=len(rows),
            source_sha=plan["source_sha"],
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    from statistics import mean

    import torch

    from mosaickv.evaluation.coco_all_methods import caption_text
    from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
    from mosaickv.evaluation.metrics.types import MetricSample
    from mosaickv.evaluation.storage_component_gate import precision_evidence, stage_effect

    plan = checked(root, expected)
    arms = plan.get("arms", ARMS)
    component_names = tuple(plan.get("component_policies", component_policies()))
    rows, errors, matches = [], [], {}
    for index in (0, 1):
        folder = root / f"workers/{index}"
        result = read(folder / "result.json") if (folder / "result.json").exists() else {}
        if result.get("status") != "completed":
            errors += result.get("errors", [f"worker {index} incomplete"])
        rows += [read(p) for p in sorted(folder.glob("[0-9][0-9]-*.json"))]
        if (folder / "storage_matching.json").exists():
            matches[str(index)] = read(folder / "storage_matching.json")
    expected_rows = 2 * len(arms)
    if len(rows) != expected_rows:
        errors.append(f"incomplete rows {len(rows)}/{expected_rows}")
    for row in rows:
        if row["status"] == "completed" and sha(Path(row["logits_path"])) != row["logits_sha256"]:
            raise ValueError("saved readout artifact changed")
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(out / "rows.json", rows)
    effects, aligned, precision = {}, {}, {}
    for sample in read(root / "inputs.json"):
        by_arm = {
            r["arm"]: r
            for r in rows
            if r["sample_id"] == sample["sample_id"] and r["status"] == "completed"
        }
        reference_arm = plan.get("component_reference_arm", "full")
        if reference_arm in by_arm:
            capture = by_arm[reference_arm].get("evaluation_capture")
            if capture is not None:
                if sha(Path(capture["path"])) != capture["sha256"]:
                    raise ValueError("immutable routed capture changed")
                precision[sample["sample_id"]] = precision_evidence(
                    torch.load(capture["path"], map_location="cpu", weights_only=True)
                )
            effects[sample["sample_id"]] = {
                arm: stage_effect(by_arm[reference_arm]["fullkv_readout"], row["fullkv_readout"])
                for arm, row in by_arm.items()
                if arm in component_names and arm != reference_arm
            }
            from mosaickv.evaluation.storage_component_gate import logit_diagnostics

            support = min(
                r["fullkv_readout"]["shared_prefix_prediction_count"]
                for a, r in by_arm.items()
                if a in component_names
            )
            reference = torch.load(
                by_arm["full_kv"]["logits_path"], map_location="cpu", weights_only=True
            )
            common = {
                a: logit_diagnostics(
                    reference,
                    torch.load(r["logits_path"], map_location="cpu", weights_only=True),
                    horizon=support,
                )
                for a, r in by_arm.items()
                if a in component_names
            }
            aligned[sample["sample_id"]] = dict(
                common_fullkv_prefix_predictions=support,
                readouts=common,
                effects={a: stage_effect(common[reference_arm], v) for a, v in common.items()},
                evaluation_only_not_proposal_input=True,
            )
    scores = {}
    if not errors:
        refs = {
            s["sample_id"]: tuple(s["references"]) for s in read(PARENT / "ledger.json")["samples"]
        }
        groups = {
            arm: [
                MetricSample(r["sample_id"], caption_text(r["detail"]), refs[r["sample_id"]])
                for r in rows
                if r["arm"] == arm
            ]
            for arm in arms
        }
        scorer = TokenizedCider([s for part in groups.values() for s in part])
        scores = {arm: scorer(part) for arm, part in groups.items()}
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            execution_passed=not errors,
            errors=errors,
            expected_rows=expected_rows,
            actual_rows=len(rows),
            storage_matching=matches,
            component_effects=effects,
            common_prefix_effects=aligned,
            precision_decomposition=precision,
            diagnostic_cider=scores,
            all_parent_stability_guards_passed=all(
                r.get("parent_guard_passed", False) for r in rows if r["arm"] in component_names
            )
            and len(rows) == expected_rows,
            replacement_gpu_authorized=False,
            fresh_evaluation_authorized=False,
            superiority_established=False,
            reason=(
                "Consumed two-image audit; execution/readout shifts cannot establish "
                "useful answer quality or justify source retirement."
            ),
        ),
    )
    lines = [
        "# Matched-storage component diagnostic",
        "",
        f"Completed records: {len(rows)}/{expected_rows}. Errors: {len(errors)}.",
        "",
        "Two consumed images, all 128 predictions. No independent superiority test.",
        "Parent-reserved is a real-allocation inactive control, not an efficient parent benchmark.",
        "Peak is a decode-owned counter. All actual storage boundaries are checked, not assumed.",
        "",
        "| Component arm | CIDEr | Initial MiB | Final MiB | Counter peak MiB | "
        "Mean FullKV shared-prefix KL | Request s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm and r["status"] == "completed"]
        if not group:
            continue
        ds = [r["detail"] for r in group]
        kl = (
            mean(r["fullkv_readout"]["shared_prefix_mean_kl"] for r in group)
            if arm != "full_kv"
            else 0.0
        )

        def boundary(d, field, native):
            return d.get(field, d.get("storage", {}).get(native))

        initial = mean(
            boundary(d, "initial_encoded_bytes", "initial_active_kv_storage_bytes") for d in ds
        )
        final = mean(
            boundary(d, "final_encoded_bytes", "final_active_kv_storage_bytes") for d in ds
        )
        peaks = [d.get("reported_counter_peak_bytes", d.get("decode_owned_peak_bytes")) for d in ds]
        peak = (
            f"{mean(peaks) / 2**20:.3f}" if all(p is not None for p in peaks) else "not_certified"
        )
        seconds = mean(d["end_to_end_seconds"] for d in ds)
        lines.append(
            f"| {arm} | {scores.get(arm, 'not_scored')} | {initial / 2**20:.3f} | "
            f"{final / 2**20:.3f} | {peak} | {kl:.6g} | {seconds:.3f} |"
        )
    lines += [
        "",
        "FullKV is a quality reference only. Timings include diagnostic synchronization "
        "and a single call per arm; no latency-superiority conclusion.",
        "Shared-prefix supports can differ after first divergence; "
        "the decision artifact identifies incomparable effects explicitly.",
        "",
        "## Matching and effects",
        "",
        "```json",
        json.dumps(dict(storage_matching=matches, component_effects=effects), indent=2),
        "```",
        "",
        "## Failures",
        *[f"\n```\n{e}\n```" for e in errors],
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--phase", default="components")
    p.add_argument("--proposal-study", action="store_true")
    p.add_argument("--allocation-study", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.proposal_study, a.allocation_study)
    elif a.action == "worker":
        return worker(a.root, a.plan_sha)
    elif a.action == "analyze":
        analyze(a.root, a.plan_sha)
    else:
        checked(a.root, a.plan_sha)
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(a.root / "components_admission.json"),
                "--state",
                str(a.root / "components_admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
