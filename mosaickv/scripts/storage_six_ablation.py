#!/usr/bin/env python3
"""Bounded, actual-allocation-matched ablation on two consumed images only."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import traceback
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

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


def component_policies(
    proposal_study=False,
    allocation_study=False,
    replacement_study=False,
    transaction_study=False,
    readout_study=False,
    readout_selection_study=False,
    global_draft_study=False,
    horizon_study=False,
    inline_study=False,
    graph_runtime_study=False,
):
    if graph_runtime_study:
        policies = component_policies(inline_study=True)
        return {
            **policies,
            "full_indexed": dict(policies["full"], graph_positional_search="indexed"),
            "inline_indexed": dict(
                policies["full"], graph_positional_search="indexed", inline_repair=True
            ),
        }
    if inline_study:
        policies = component_policies(readout_study=True)
        keep = (
            "full",
            "source_replaced",
            "source_replaced_parent",
            "parent_reserved",
            "exact_members",
            "forced_repair",
        )
        template = policies["source_replaced"]
        return {
            **{k: policies[k] for k in keep},
            "inline_entropy": dict(template, inline_repair=True),
            "inline_forced": dict(template, inline_repair=True, force_repair=True),
            "inline_no_repair": dict(template, inline_repair=True, component_mode="no_repair"),
        }
    if horizon_study:
        policies = component_policies(global_draft_study=True)
        keys = (
            "full",
            "source_replaced",
            "source_replaced_parent",
            "parent_reserved",
            "exact_members",
            "forced_repair",
            "readout_joint",
        )
        return {
            **{key: policies[key] for key in keys},
            **{f"controlled_plan_{i}": policies["readout_joint"] for i in range(10)},
        }
    readout_selection_study = readout_selection_study or global_draft_study
    base = OverlayPolicy(record_phase_timings=True)
    if readout_selection_study:
        seed = replace(
            base,
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
            graph_accumulation="segmented_tensor_topk",
            precision_transaction="prototype",
        )
        replaced = replace(seed, replace_source_codes=True)
        target = "fullkv_draft" if global_draft_study else "conditional_head"
        active = replace(replaced, readout_selection="joint", readout_target=target)
        joint = active
        return {
            "full": asdict(seed),
            "source_replaced": asdict(replaced),
            "source_replaced_parent": asdict(
                replace(replaced, enabled=False, precision_transaction="none")
            ),
            "parent_reserved": asdict(replace(replaced, component_mode="parent_reserved")),
            "exact_members": asdict(replace(replaced, component_mode="exact_members")),
            "forced_repair": asdict(replace(replaced, component_mode="forced_repair")),
            "readout_exact": asdict(replace(active, readout_selection="exact")),
            "readout_repair": asdict(replace(active, readout_selection="repair")),
            "readout_joint": asdict(joint),
            **{
                name: asdict(
                    replace(
                        joint,
                        component_mode=name,
                        precision_transaction="none" if name == "no_forecast" else "prototype",
                        readout_selection="none"
                        if name == "no_forecast"
                        else "exact"
                        if name == "no_repair"
                        else "joint",
                        readout_target="conditional_head" if name == "no_forecast" else target,
                    )
                )
                for name in (
                    "no_forecast",
                    "no_graph",
                    "no_value",
                    "modular_selection",
                    "no_repair",
                )
            },
        }
    if readout_study:
        improved = replace(
            base,
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
            graph_accumulation="segmented_tensor_topk",
            precision_transaction="prototype",
        )
        replaced = replace(improved, replace_source_codes=True)
        return {
            "full": asdict(improved),
            "source_replaced": asdict(replaced),
            "source_replaced_parent": asdict(
                replace(replaced, enabled=False, precision_transaction="none")
            ),
            "parent_reserved": asdict(replace(replaced, component_mode="parent_reserved")),
            "exact_members": asdict(replace(replaced, component_mode="exact_members")),
            "forced_repair": asdict(replace(replaced, component_mode="forced_repair")),
            "readout_control": asdict(replace(replaced, readout_control=True)),
        }
    if transaction_study:
        improved = replace(
            base,
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
            graph_accumulation="segmented_tensor_topk",
        )
        replaced = replace(improved, replace_source_codes=True)
        adaptive = replace(replaced, precision_transaction="joint_guarded")
        return {
            "full": asdict(improved),
            "source_replaced": asdict(replaced),
            "source_replaced_parent": asdict(replace(replaced, enabled=False)),
            "parent_reserved": asdict(replace(replaced, component_mode="parent_reserved")),
            "transaction_exact": asdict(replace(replaced, precision_transaction="exact")),
            "transaction_prototype": asdict(replace(replaced, precision_transaction="prototype")),
            "transaction_joint_guarded": asdict(adaptive),
            **{
                name: asdict(
                    replace(
                        adaptive,
                        component_mode=name,
                        precision_transaction=(
                            "none"
                            if name == "no_forecast"
                            else "prototype"
                            if name == "no_value"
                            else "joint_guarded"
                        ),
                    )
                )
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
        }
    if replacement_study:
        improved = replace(
            base,
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
            graph_accumulation="segmented_tensor_topk",
        )
        replaced = replace(improved, replace_source_codes=True)
        return {
            "full": asdict(improved),
            "parent_reserved": asdict(replace(improved, component_mode="parent_reserved")),
            "exact_members": asdict(replace(improved, component_mode="exact_members")),
            "forced_repair": asdict(replace(improved, component_mode="forced_repair")),
            "source_replaced": asdict(replaced),
            "source_replaced_parent": asdict(replace(replaced, enabled=False)),
            "signed_gain": asdict(replace(replaced, utility_signal="exact_restoration_gain")),
            "actual_attention_risk": asdict(
                replace(replaced, repair_policy="entropy_or_prototype_risk")
            ),
        }
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


def prepare(
    root,
    proposal_study=False,
    allocation_study=False,
    replacement_study=False,
    transaction_study=False,
    readout_study=False,
    readout_selection_study=False,
    global_draft_study=False,
    horizon_study=False,
    inline_study=False,
    graph_runtime_study=False,
    external_graph_parity_root=None,
    after_gpu_job=None,
):
    inline_study = inline_study or graph_runtime_study
    replacement_study = replacement_study or inline_study or graph_runtime_study
    global_draft_study = global_draft_study or horizon_study
    readout_selection_study = readout_selection_study or global_draft_study
    replacement_study = (
        replacement_study or transaction_study or readout_study or readout_selection_study
    )
    proposal_study = proposal_study or allocation_study or replacement_study
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
    cross_source = {}
    if external_graph_parity_root is not None:
        gate = Path(external_graph_parity_root)
        external = read(gate / "plan.json")
        if read(gate / "analysis/decision.json").get("execution_passed") is not True:
            raise ValueError("completed external graph/logit evidence required")
        if digest(external["parent"]["config"]) != digest(parent["config"]) or digest(
            read(gate / "inputs.json")
        ) != digest(inputs):
            raise ValueError("external graph parity uses different model/inputs/controls")
        paths += [gate / "plan.json", gate / "analysis/decision.json", gate / "inputs.json"]
        for i in (0, 1):
            p = next((gate / f"workers/{i}").glob("[0-9][0-9]-full.json"))
            row = read(p)
            path = Path(row["logits_path"])
            if row["status"] != "completed" or sha(path) != row["logits_sha256"]:
                raise ValueError("external graph reference missing or changed")
            cross_source[str(i)] = dict(
                logits_path=str(path),
                graph_coo_sha256=row["mechanism_evidence"]["layers"]["31"]["graph_coo_sha256"],
            )
            paths += [p, path]
    if after_gpu_job is not None and (
        not isinstance(after_gpu_job, str) or not after_gpu_job.isdigit()
    ):
        raise ValueError("one explicit predecessor GPU array/job ID required")
    fixed_plans = {}
    if horizon_study:
        completed = Path("/scratch/djy8hg/runs/evicastkv/offloaded-fullkv-draft-20261107-v1")
        decision = completed / "analysis/decision.json"
        if read(decision).get("execution_passed") is not True:
            raise ValueError("actual FullKV forecast/isolation controls must pass first")
        paths.append(decision)
        for i in (0, 1):
            p = next((completed / f"workers/{i}").glob("[0-9][0-9]-readout_joint.json"))
            row = read(p)
            if not row.get("online_fullkv_draft_bitwise_parity"):
                raise ValueError("ordinary FullKV forecast parity required")
            plans = row["mechanism_evidence"]["conditional_readout_selection"]["plans"]
            if len(plans) != 10:
                raise ValueError("exactly the existing ten fixed plans required")
            fixed_plans[str(i)] = [
                {k: r[k] for k in ("name", "ordinal", "exact_ids", "restore")} for r in plans
            ]
            paths.append(p)
    if readout_selection_study:
        gate = Path("/scratch/djy8hg/runs/evicastkv/exact-last-layer-readout-20261107-v2")
        decision = gate / "analysis/decision.json"
        if read(decision).get("execution_passed") is not True:
            raise ValueError("actual exact-readout engineering gate has not passed")
        paths.append(decision)
        for i in (0, 1):
            row_path = next((gate / f"workers/{i}").glob("[0-9][0-9]-readout_control.json"))
            row = read(row_path)
            if row.get("readout_control_passed") is not True:
                raise ValueError("both actual readout controls required before selection")
            evidence = row["mechanism_evidence"]["exact_last_layer_readout"]
            if not evidence["original_overlay_restored"] or not evidence["model_weights_unchanged"]:
                raise ValueError("readout isolation control failed")
            paths += [row_path, Path(row["readout_capture"]["path"])]
            if sha(paths[-1]) != row["readout_capture"]["sha256"]:
                raise ValueError("immutable readout evidence changed")
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
                "mosaickv/tests/unit/test_source_replacement.py",
                "mosaickv/tests/unit/test_precision_gain.py",
                "mosaickv/tests/unit/test_source_hotpath.py",
                "mosaickv/tests/unit/test_shared_source_map_accounting.py",
                "mosaickv/tests/unit/test_precision_transaction_study.py",
                "mosaickv/tests/unit/test_exact_last_layer_readout.py",
                "mosaickv/tests/unit/test_conditional_readout_selection.py",
                "mosaickv/tests/unit/test_offloaded_fullkv_draft.py",
                "mosaickv/tests/unit/test_readout_horizon_audit.py",
                "mosaickv/tests/unit/test_inline_first_step_repair.py",
                "mosaickv/tests/unit/test_indexed_positional_graph.py",
                "mosaickv/tests/unit/test_graph.py",
                "mosaickv/tests/unit/test_direct_attention_pooling.py",
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
    if transaction_study:
        from mosaickv.evaluation.precision_transaction_study import study as transaction_audit

        proof = {}
        consumed = Path("/scratch/djy8hg/runs/evicastkv/storage-source-replacement-20261106-v6")
        for worker in (0, 1):
            row_path = next((consumed / f"workers/{worker}").glob("[0-9][0-9]-full.json"))
            row = read(row_path)
            capture = row["evaluation_capture"]
            if sha(Path(capture["path"])) != capture["sha256"]:
                raise ValueError("consumed precision capture changed")
            result = transaction_audit(
                torch.load(capture["path"], weights_only=True, map_location="cpu")
            )
            if not (result["accepted_meaningful_change"] and result["final_joint_guard_passed"]):
                raise ValueError("guarded transaction has no useful prospective change; stop GPU")
            proof[str(worker)] = result
            paths += [row_path, Path(capture["path"])]
        write_once(root / "cpu_transaction_feasibility.json", proof)
        paths += [root / "cpu_transaction_feasibility.json"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            parent=parent,
            options=options,
            bound_files={str(p): sha(p) for p in paths},
            arms=(
                "full_kv",
                "parent",
                *component_policies(
                    proposal_study,
                    allocation_study,
                    replacement_study,
                    transaction_study,
                    readout_study,
                    readout_selection_study,
                    global_draft_study,
                    horizon_study,
                    inline_study,
                    graph_runtime_study,
                ),
            ),
            component_policies=component_policies(
                proposal_study,
                allocation_study,
                replacement_study,
                transaction_study,
                readout_study,
                readout_selection_study,
                global_draft_study,
                horizon_study,
                inline_study,
                graph_runtime_study,
            ),
            source_replacement_study=replacement_study,
            exact_last_layer_readout_study=readout_study,
            conditional_readout_selection_study=readout_selection_study,
            offloaded_online_fullkv_draft_study=global_draft_study,
            evaluation_only_fixed_plan_horizon_audit=horizon_study,
            oracle_horizon_audit_not_online_selection=horizon_study,
            fixed_consumed_counterfactual_plans=fixed_plans,
            inline_first_step_repair_study=inline_study,
            indexed_positional_graph_runtime_study=graph_runtime_study,
            external_graph_parity=cross_source,
            gpu_start_after_any_job=after_gpu_job,
            storage_boundary_version="shared_initial_final_peak_v2",
            component_reference_arm=(
                "transaction_joint_guarded"
                if transaction_study
                else "readout_joint"
                if readout_selection_study
                else "all_adaptive"
                if allocation_study
                else "full"
            ),
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
    original_submit = infra.submit

    def ordered_submit(*args, **kwargs):
        if after_gpu_job is not None and kwargs.get("array") is not None:
            kwargs["after"] = after_gpu_job
        return original_submit(*args, **kwargs)

    infra.submit = ordered_submit
    try:
        infra.launch(root, "components")
    finally:
        infra.submit = original_submit


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
        if plan.get("source_replacement_study"):
            from mosaickv.evaluation.source_replacement_controls import gpu_controls

            write_once(folder / "source_operator_controls.json", gpu_controls())
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
                storage_boundary_version=plan.get(
                    "storage_boundary_version", "legacy_initial_private"
                ),
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
                        kind = payload.get("capture_kind")
                        suffix = f"-{kind}" if kind else ""
                        path = (
                            folder
                            / f"{record['arm']}{suffix}-layer-{payload['layer']}-evaluation-only.pt"
                        )
                        if path.exists():
                            raise FileExistsError("immutable routed capture already exists")
                        torch.save(payload, path)
                        record[
                            "fullkv_draft_capture"
                            if kind == "fullkv_draft"
                            else "readout_capture"
                            if kind
                            else "evaluation_capture"
                        ] = dict(path=str(path), sha256=sha(path))

                    with ExitStack() as scope:
                        if plan.get("evaluation_only_fixed_plan_horizon_audit") and arm.startswith(
                            "controlled_plan_"
                        ):
                            from mosaickv.evaluation import (
                                conditional_readout_selection as selection,
                            )
                            from mosaickv.evaluation.readout_horizon_audit import forced_plan

                            ordinal = int(arm.rsplit("_", 1)[1])

                            def force(records, index=ordinal):
                                return forced_plan(records, index)

                            scope.enter_context(patch.object(selection, "choose", force))
                        proof = scope.enter_context(
                            integrated_storage(
                                adapter,
                                policy,
                                reader_options=options,
                                diagnostic_capture=capture
                                if arm in ("full", "all_adaptive", "readout_joint")
                                or policy.readout_control
                                or policy.readout_selection != "none"
                                else None,
                            )
                        )
                        values, detail = campaign.execute(
                            adapter, inputs, transport, "mosaickv_exact", audit=False
                        )
                    row["mechanism_evidence"] = proof
                    if (
                        plan.get("evaluation_only_fixed_plan_horizon_audit")
                        and policy.readout_selection != "none"
                    ):
                        actual_plans = [
                            {
                                k: list(r[k]) if k == "exact_ids" else r[k]
                                for k in ("name", "ordinal", "exact_ids", "restore")
                            }
                            for r in proof["conditional_readout_selection"]["plans"]
                        ]
                        if actual_plans != plan["fixed_consumed_counterfactual_plans"][str(index)]:
                            raise RuntimeError("frozen counterfactual candidate pool changed")
                    if detail["prefill_storage"]["original_KV_tensor_objects_alive"] != 0:
                        raise RuntimeError("original dense KV owner survived")
                    if not proof["all_residuals_pinned"] or proof["extra_gpu_bytes"] != (
                        6713 if policy.enabled else 0
                    ):
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
                if arm in policies and policy.readout_control:
                    row["readout_control_passed"] = proof["exact_last_layer_readout"]["passed"]
                    if not row["readout_control_passed"]:
                        errors.append("actual last-layer counterfactual numerical controls failed")
                if arm in policies and policy.readout_target == "fullkv_draft":
                    forecast = torch.load(
                        row["fullkv_draft_capture"]["path"], weights_only=True, map_location="cpu"
                    )
                    row["online_fullkv_draft_bitwise_parity"] = torch.equal(
                        forecast["logits"], values_by_arm["full_kv"][:2]
                    )
                    if not row["online_fullkv_draft_bitwise_parity"]:
                        errors.append(
                            "online CPU-offloaded FullKV draft differs from ordinary FullKV"
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
            if "readout_control" in values_by_arm and not torch.equal(
                values_by_arm["readout_control"], values_by_arm["source_replaced"]
            ):
                errors.append("readout diagnostic leaked into the final decode state")
            if plan.get("evaluation_only_fixed_plan_horizon_audit") and not torch.equal(
                values_by_arm["controlled_plan_0"], values_by_arm["source_replaced"]
            ):
                errors.append("controlled identity changed a final decode logit bit")
            if plan.get("inline_first_step_repair_study"):
                for arm, reference in (
                    ("inline_entropy", "source_replaced"),
                    ("inline_forced", "forced_repair"),
                ):
                    if not torch.equal(values_by_arm[arm], values_by_arm[reference]):
                        errors.append(f"inline repair differs bitwise from original policy: {arm}")
                    evidence = completed[arm]["mechanism_evidence"]["inline_first_step_repair"]
                    if not evidence["completed"] or evidence["retained_private_tensor_owners"]:
                        errors.append("inline observation did not engage or retire")
            if plan.get("indexed_positional_graph_runtime_study"):
                expected_graph = completed["full"]["mechanism_evidence"]["layers"]["31"][
                    "graph_coo_sha256"
                ]
                for arm in ("full_indexed", "inline_indexed"):
                    if not torch.equal(values_by_arm[arm], values_by_arm["full"]):
                        errors.append(f"indexed graph changed a final logit bit: {arm}")
                    if (
                        completed[arm]["mechanism_evidence"]["layers"]["31"]["graph_coo_sha256"]
                        != expected_graph
                    ):
                        errors.append(f"indexed graph changed an actual edge/weight: {arm}")
            if plan.get("external_graph_parity"):
                external = plan["external_graph_parity"][str(index)]
                if completed["full"]["mechanism_evidence"]["layers"]["31"][
                    "graph_coo_sha256"
                ] != external["graph_coo_sha256"] or not torch.equal(
                    values_by_arm["full"],
                    torch.load(external["logits_path"], weights_only=True, map_location="cpu"),
                ):
                    errors.append("direct attention pooling changed a frozen graph or logit bit")
            if "individual_delta_fast_graph" in values_by_arm and not torch.equal(
                values_by_arm["individual_delta_fast_graph"], values_by_arm["individual_delta"]
            ):
                errors.append("segmented graph execution changed a final model logit bit")
            if plan.get("source_replacement_study"):
                for arm, reference in (
                    ("source_replaced", "full"),
                    ("source_replaced_parent", "parent"),
                ):
                    if not torch.equal(values_by_arm[arm], values_by_arm[reference]):
                        errors.append(f"lossless source replacement changed logits: {arm}")
                for row in rows:
                    policy = policies.get(row["arm"], {})
                    if policy.get("replace_source_codes") and row["status"] == "completed":
                        proof = row["mechanism_evidence"]["source_replacement"]
                        if (
                            proof["replaced_code_owners"] != 32
                            or proof["old_code_tensor_objects_alive"] != 0
                            or proof["saved_gpu_bytes"] <= 0
                        ):
                            errors.append("incomplete source replacement/retirement")
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
    from mosaickv.evaluation.storage_component_gate import (
        precision_evidence,
        stage_effect,
        storage_match,
    )

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
    replacement_pairs = []
    if plan.get("source_replacement_study"):
        for sample in read(root / "inputs.json"):
            group = {
                r["arm"]: r
                for r in rows
                if r["sample_id"] == sample["sample_id"] and r["status"] == "completed"
            }
            for arm, reference in (
                ("source_replaced", "full"),
                ("source_replaced_parent", "parent"),
            ):
                if arm not in group or reference not in group:
                    errors.append(f"missing lossless replacement pair: {sample['sample_id']} {arm}")
                    continue
                r, old = group[arm], group[reference]
                proof = r["mechanism_evidence"]["source_replacement"]
                bitwise = torch.equal(
                    torch.load(r["logits_path"], weights_only=True, map_location="cpu"),
                    torch.load(old["logits_path"], weights_only=True, map_location="cpu"),
                )
                ds, before = r["detail"], old["detail"]
                initial = before["initial_encoded_bytes"] - ds["initial_encoded_bytes"]
                final = before["final_encoded_bytes"] - ds["final_encoded_bytes"]
                passed = (
                    bitwise
                    and initial == final == proof["saved_gpu_bytes"] > 0
                    and proof["old_code_tensor_objects_alive"] == 0
                    and proof["replaced_code_owners"] == 32
                )
                replacement_pairs.append(
                    dict(
                        sample_id=sample["sample_id"],
                        arm=arm,
                        reference=reference,
                        bitwise_128_prediction_logits=bitwise,
                        initial_saved_bytes=initial,
                        final_saved_bytes=final,
                        code_retirement=proof,
                        passed=passed,
                    )
                )
                if not passed:
                    errors.append(
                        f"source replacement/accounting gate failed: {sample['sample_id']} {arm}"
                    )
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
    horizon_proof = {}
    if plan.get("evaluation_only_fixed_plan_horizon_audit") and not errors:
        from mosaickv.evaluation.readout_horizon_audit import audit as horizon_audit

        for sample in read(root / "inputs.json"):
            by_arm = {r["arm"]: r for r in rows if r["sample_id"] == sample["sample_id"]}
            reference = torch.load(
                by_arm["full_kv"]["logits_path"], weights_only=True, map_location="cpu"
            )
            candidates = {}
            for i in range(10):
                record = by_arm[f"controlled_plan_{i}"]
                name = "identity" if i == 0 else f"plan_{i}"
                candidates[name] = torch.load(
                    record["logits_path"], weights_only=True, map_location="cpu"
                )
            proof = horizon_audit(reference, candidates)
            proof["measured_storage_matching"] = storage_match(
                [by_arm[f"controlled_plan_{i}"] for i in range(10)]
            )
            horizon_proof[sample["sample_id"]] = proof
        write_once(out / "evaluation_only_horizon_audit.json", horizon_proof)
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
            source_replacement_pairs=replacement_pairs,
            component_effects=effects,
            common_prefix_effects=aligned,
            precision_decomposition=precision,
            diagnostic_cider=scores,
            evaluation_only_horizon_audit=horizon_proof,
            all_parent_stability_guards_passed=all(
                r.get("parent_guard_passed", False) for r in rows if r["arm"] in component_names
            )
            and len(rows) == expected_rows,
            replacement_gpu_authorized=False,
            fresh_evaluation_authorized=False,
            superiority_established=False,
            reason=(
                "Consumed two-image audit; source replacement requires verified owner retirement, "
                "positive net backing savings and bitwise parent logits. Local readout changes "
                "do not establish benchmark superiority or authorize larger campaigns."
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
        "## Lossless source replacement",
        "",
        "This common codec is not prototype compaction. Mixed-budget rows above are not "
        "byte-matched quality comparisons. Only policy ablations using the same replacement "
        "and auxiliary buffers are storage-matched. Whole-request peak remains uncertified.",
        "```json",
        json.dumps(replacement_pairs, indent=2),
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
    p.add_argument("--replacement-study", action="store_true")
    p.add_argument("--transaction-study", action="store_true")
    p.add_argument("--readout-study", action="store_true")
    p.add_argument("--readout-selection-study", action="store_true")
    p.add_argument("--global-draft-study", action="store_true")
    p.add_argument("--horizon-study", action="store_true")
    p.add_argument("--inline-study", action="store_true")
    p.add_argument("--graph-runtime-study", action="store_true")
    p.add_argument("--graph-parity-root", type=Path)
    p.add_argument("--after-gpu-job")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(
            a.root,
            a.proposal_study,
            a.allocation_study,
            a.replacement_study,
            a.transaction_study,
            a.readout_study,
            a.readout_selection_study,
            a.global_draft_study,
            a.horizon_study,
            a.inline_study,
            a.graph_runtime_study,
            a.graph_parity_root,
            a.after_gpu_job,
        )
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
