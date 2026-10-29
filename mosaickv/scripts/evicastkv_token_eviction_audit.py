#!/usr/bin/env python3
"""Bounded diagnostic: saved greedy outputs + missing per-token teacher NLL."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_layerwise_prior as previous
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.config import RunConfig, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.token_eviction import compare_nll, first_divergence

SOURCE = Path(__file__).resolve().parents[2]
EPSILON = 1e-6
LOSS_SUM_ATOL = 1e-8
ROLES = (
    "full_kv_reference",
    "native_prefixkv_reference",
    "window_topk_control",
    "uniform_0.5_control",
    "uniform_1_control",
)


def bind_row(bound, path):
    row = common.read(path)
    if row["status"] != "completed":
        raise ValueError("completed historical control required")
    for p in (path, Path(row["manifest_path"])):
        bound[str(p)] = sha(p)
    for p, digest in row["traces"].items():
        if sha(Path(p)) != digest:
            raise ValueError("historical trace changed")
        bound[p] = digest
    return row


def config_arm(row):
    manifest = common.read(row["manifest_path"])
    arm = next(a for a in manifest["all_arms"] if a["label"] == row["label"])
    cfg = RunConfig.from_mapping(arm["config"])
    if config_sha256(cfg) != row["config_sha"]:
        raise ValueError("historical configuration cannot be reproduced exactly")
    return dict(arm, config_sha=row["config_sha"])


def greedy_alignment(row, ref, *, generated_capacity):
    a, b = previous.phase_trace(row), previous.phase_trace(ref)
    first = first_divergence(a["generated_token_ids"], b["generated_token_ids"])

    # EOS positions are observed, one-based. A different EOS decision itself
    # can change the scored answer; differences only after both EOS cannot.
    def eos(t):
        ids = set(t["quality_decoding"]["configured_terminal_token_ids"])
        return next(
            (i + 1 for i, token in enumerate(t["generated_token_ids"]) if token in ids), None
        )

    end_a, end_b = eos(a), eos(b)
    first_possible = generated_capacity + 2 if generated_capacity is not None else None
    return dict(
        first_greedy_divergence=first,
        candidate_first_eos=end_a,
        reference_first_eos=end_b,
        divergence_after_both_eos=bool(
            first is not None
            and end_a is not None
            and end_b is not None
            and first > max(end_a, end_b)
        ),
        first_possible_fifo_affected_prediction=first_possible,
        divergence_before_fifo_eviction=(first < first_possible)
        if first is not None and first_possible is not None
        else None,
        fifo_boundary_is_source_derived_until_teacher_capture_verifies_it=True,
    )


def prepare(root, parent, expected):
    source = clean_source(SOURCE)
    if root.exists() or sha(parent / "plan.json") != expected:
        raise ValueError("new immutable root and pinned consumed plan required")
    old = common.read(parent / "plan.json")
    gate = common.read(parent / "analysis/GATE.json")
    if (
        gate["status"] != "complete_consumed_diagnostic"
        or gate["failures"]
        or gate["point_eligible_schedules"]
    ):
        raise ValueError("complete negative layer-wise diagnostic required")
    bound = dict(old["bound_files"])
    for p in (parent / "plan.json", parent / "analysis/GATE.json"):
        bound[str(p)] = sha(p)
    for p, h in bound.items():
        if sha(Path(p)) != h:
            raise ValueError("bound evidence changed")
    jobs, greedy = [], []
    for task in ("llava_description", "mmvet_prefixkv"):
        template = next(j for j in old["jobs"] if j["task"] == task)
        full_path = (
            Path(template["frozen_fullkv_reference"]).parent / "rows/full_kv-native-control.json"
        )
        controls = {
            "full_kv_reference": full_path,
            "native_prefixkv_reference": Path(template["native_control_row"]),
        }
        for role in ROLES[2:]:
            j = next(
                j for j in old["jobs"] if j["task"] == task and j["diagnostic_layer_policy"] == role
            )
            controls[role] = parent / f"workers/{j['index']}/rows/{j['arms'][0]['label']}.json"
        for role, path in controls.items():
            row = bind_row(bound, path)
            arm = config_arm(row)
            cfg = RunConfig.from_mapping(arm["config"])
            if (
                cfg.repair.enabled
                or cfg.prototypes.enabled
                or (cfg.forecasting.enabled and cfg.forecasting.draft_steps)
            ):
                raise ValueError("diagnostic excludes repair/prototypes/draft probes")
            jobs.append(
                dict(
                    template,
                    index=len(jobs),
                    diagnostic_role=role,
                    frozen_measurement_row=str(path),
                    arms=[arm],
                )
            )
        # Reuse every existing schedule's generation; do not rerun the prior sweep.
        refs = {k: common.read(v) for k, v in controls.items() if k in ROLES[:3]}
        for j in (j for j in old["jobs"] if j["task"] == task):
            row_path = parent / f"workers/{j['index']}/rows/{j['arms'][0]['label']}.json"
            row = bind_row(bound, row_path)
            for name, ref in refs.items():
                greedy.append(
                    dict(
                        task=task,
                        sample_id=j["sample_id"],
                        policy=j["diagnostic_layer_policy"],
                        reference=name,
                        **greedy_alignment(row, ref, generated_capacity=256),
                    )
                )
    if len(jobs) != 10:
        raise ValueError("ten teacher-only captures, no additional identities")
    write_once(root / "saved_greedy_alignment.json", dict(records=greedy))
    bound[str(root / "saved_greedy_alignment.json")] = sha(root / "saved_greedy_alignment.json")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=source,
            jobs=jobs,
            bound_files=bound,
            protocol="consumed_per_token_NLL_eviction_alignment_v1",
            source_parent_sha=old["source_sha"],
            saved_generation_reused=True,
            generation_reruns_authorized=False,
            per_token_nll_epsilon_nats=EPSILON,
            repeated_nll_sum_absolute_tolerance=LOSS_SUM_ATOL,
            fixed_window_tokens=16,
            no_algorithm_or_budget_changes=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(root / "plan.json")


def worker(root, expected, index):
    import torch

    import mosaickv.backends.hf_runtime as runtime_module
    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.token_eviction import TeacherTokenCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    if out.exists():
        raise FileExistsError("write-once attempt required")
    out.mkdir(parents=True)
    started = time.monotonic()
    capture = None
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        role=job["diagnostic_role"],
    )
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace required before Python")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        ledger = load_paper_native_ledger(
            job["ledger"],
            expected_sha256=sha(Path(job["ledger"])),
            task=default_task_registry().resolve(job["task"]),
            dataset_revision=cfg.dataset.revision,
        )
        sample = next(s for s in ledger.samples if s.sample_id == job["sample_id"])
        messages = build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                config_sha=config_sha256(cfg),
                config=job["arms"][0]["config"],
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="evaluation_only_token_eviction_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                frozen_measurement_row=job["frozen_measurement_row"],
                frozen_row_sha256=sha(Path(job["frozen_measurement_row"])),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                baseline_commit="597f1ab032704951550f93bcc8a23f1454b80aa4"
                if cfg.method.is_prefixkv_reimplementation
                else plan["source_sha"],
                no_paper_or_confirmation_claim=True,
                timings_diagnostic_only=True,
            ),
        )
        parity = compare_with_generate(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
        write_once(out / "parity.json", asdict(parity))
        if parity.token_agreement != 1 or parity.maximum_logit_difference > 0.001:
            raise ValueError("shared FullKV output parity failed")
        runtime = runtime_module.HuggingFaceMosaicKVModel(
            adapter, cfg, trace_directory=out / "traces"
        )
        targets = runtime._teacher_forced_target_ids(sample.teacher_forcing_target)
        old = common.read(job["frozen_measurement_row"])
        old_trace = previous.phase_trace(old, "teacher_forced")
        if targets != tuple(old_trace["generated_token_ids"]):
            raise ValueError("teacher target tokenization changed")
        request = EvaluationRequest(
            "token-audit-" + str(index),
            sample.sample_id,
            job["task"],
            messages,
            {"_mosaickv_capture_kv_ownership": True},
        )
        with TeacherTokenCapture(adapter, targets, runtime_module) as capture:
            loss = runtime.teacher_forced_nll(request, sample.teacher_forcing_target)
        detail = capture.record()
        traces = list((out / "traces").rglob("*.json"))
        if len(traces) != 1:
            raise ValueError("exactly one teacher-forced replay trace required")
        trace = common.read(traces[0])
        fields = (
            "generated_token_ids",
            "selected_blocks",
            "packed_prefill_active_kv_bytes",
            "cache_lifecycle",
        )
        equivalent = all(trace.get(k) == old_trace.get(k) for k in fields)
        equivalent = equivalent and loss.evaluated_tokens == old["evaluated_tokens"]
        equivalent = (
            equivalent and abs(loss.negative_log_likelihood_sum - old["nll_sum"]) <= LOSS_SUM_ATOL
        )
        equivalent = (
            equivalent
            and abs(detail["sum_nll"] - loss.negative_log_likelihood_sum) <= LOSS_SUM_ATOL
        )
        if not equivalent:
            raise ValueError(
                "diagnostic replay changed frozen NLL, targets, selected positions or cache storage"
            )
        write_once(out / "per_token.json", detail)
        result.update(
            status="completed",
            config_sha=config_sha256(cfg),
            sample_id=sample.sample_id,
            nll_sum=loss.negative_log_likelihood_sum,
            evaluated_tokens=loss.evaluated_tokens,
            exact_control_equivalence=True,
            per_token_sha256=sha(out / "per_token.json"),
            teacher_trace=str(traces[0]),
            teacher_trace_sha256=sha(traces[0]),
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        if capture is not None:
            write_once(
                out / "partial_capture.json",
                dict(
                    status="failed_partial_diagnostic",
                    records=capture.records,
                    native_post_forward_cleanup_events=capture.native_events,
                    complete_target_count=len(capture.target_ids),
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(out / "result.json", result)


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    rows, comparisons, failures, captures = [], [], [], {}
    for job in plan["jobs"]:
        w = root / f"workers/{job['index']}"
        if not (w / "result.json").exists():
            failures.append(dict(worker=job["index"], error="missing worker result"))
            continue
        result = common.read(w / "result.json")
        if result["status"] != "completed":
            failures.extend(result["failures"])
            continue
        if (
            sha(w / "per_token.json") != result["per_token_sha256"]
            or sha(Path(result["teacher_trace"])) != result["teacher_trace_sha256"]
        ):
            raise ValueError("captured artifact changed")
        detail = common.read(w / "per_token.json")
        captures[job["task"], job["diagnostic_role"]] = detail
        rows.extend(
            dict(x, task=job["task"], sample_id=job["sample_id"], role=job["diagnostic_role"])
            for x in detail["records"]
        )
    for (task, role), detail in captures.items():
        for ref in ("full_kv_reference", "native_prefixkv_reference", "window_topk_control"):
            if ref == role or (task, ref) not in captures:
                continue
            compare = compare_nll(detail["records"], captures[task, ref]["records"], EPSILON)
            comparisons.append(dict(task=task, role=role, reference=ref, **compare))
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("immutable derived output required")
    out.mkdir()
    pd.DataFrame(rows).to_parquet(out / "per_token_nll.parquet", index=False)
    pd.DataFrame(
        [
            {k: v for k, v in x.items() if k not in {"per_token_deltas", "fixed_windows"}}
            for x in comparisons
        ]
    ).to_parquet(out / "comparisons.parquet", index=False)
    write_once(out / "comparisons.json", dict(records=comparisons))
    gate = dict(
        status="complete_consumed_diagnostic" if len(captures) == 10 and not failures else "failed",
        expected_captures=10,
        actual_captures=len(captures),
        failures=failures,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        superiority_established=False,
        temporal_alignment_not_causal_proof=True,
    )
    write_once(out / "GATE.json", gate)
    table = pd.read_parquet(out / "comparisons.parquet")
    greedy = common.read(root / "saved_greedy_alignment.json")["records"]
    display = pd.DataFrame(
        [
            {
                k: v
                for k, v in x.items()
                if k
                not in {
                    "sample_id",
                    "fifo_boundary_is_source_derived_until_teacher_capture_verifies_it",
                }
            }
            for x in greedy
        ]
    )
    (out / "report.md").write_text(
        "# Per-token NLL and history-eviction diagnosis\n\n"
        "Same two consumed inputs; no new policy, budget, cohort, or generation sweep. "
        "Saved 1024-token greedy trajectories are reused; post-EOS divergence is distinguished. "
        "Teacher replay must reproduce frozen aggregate NLL, selected positions and storage. "
        "Position 1 uses uncompressed prefill logits. Native PrefixKV cleanup after a forward "
        "cannot affect that forward's logits; FIFO replacement within update can. "
        "Differences are descriptive, not statistical quality/superiority evidence. "
        "Native PrefixKV is a separate operating point, not byte matched. "
        "These capture timings are not comparable to the primary systems measurements.\n\n"
        + table.to_markdown(index=False)
        + "\n\n## Reused greedy trajectories\n\n"
        + display.to_markdown(index=False)
        + "\n\n```json\n"
        + json.dumps(gate, indent=2)
        + "\n```\n"
    )
    if len(captures) == 10 and not failures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, task in zip(axes, ("llava_description", "mmvet_prefixkv"), strict=True):
            for x in comparisons:
                if x["task"] == task and x["reference"] == "full_kv_reference":
                    cumulative, series = 0.0, []
                    for d in x["per_token_deltas"]:
                        cumulative += d
                        series.append(cumulative)
                    ax.plot(range(1, len(series) + 1), series, label=x["role"])
            if len(captures[task, "window_topk_control"]["records"]) >= 258:
                ax.axvline(
                    258, color="black", linestyle="--", label="FIFO first affected prediction"
                )
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set(
                title=task,
                xlabel="Teacher target position (1-based)",
                ylabel="Cumulative NLL delta vs FullKV (nats)",
            )
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "cumulative_nll.pdf")
        plt.close(fig)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={str(p): sha(p) for p in (root / "workers").rglob("*.json")},
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    print(json.dumps(gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--index", type=int)
    a = parser.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root, a.parent, a.plan_sha))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_token_eviction_audit.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
