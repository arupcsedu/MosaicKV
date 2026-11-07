#!/usr/bin/env python3
"""Lossless selected-pack / single-head guard verification against storage-only."""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.source_hybrid_validation import ARMS, BOUNDARIES, COMPARISONS
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-model-20261107-v1")
CPU = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-feasibility-20261107-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(p):
    return json.loads(Path(p).read_text())


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("immutable worktree package boundary required")
    if sha(root / "plan.json") != expected:
        raise ValueError("registered plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if any(sha(Path(p)) != checksum for p, checksum in plan["bound_files"].items()):
        raise ValueError("immutable controls, original logits or source streams changed")
    return plan


def prepare(
    root,
    bitmap_prefix=False,
    reuse_anchor_readout=False,
    shared_suffix=False,
    planner_exact_fast=False,
    cpu_torch_threads=12,
    guard_tail=False,
):
    import all_head_native_source as infra

    source = clean_source(SOURCE)
    if guard_tail and (not shared_suffix or not planner_exact_fast or reuse_anchor_readout):
        raise ValueError("native tail diagnostic requires the exact planner/shared suffix controls")
    parent = read(PARENT / "plan.json")
    if not read(PARENT / "analysis/decision.json")["model_parity_gate_passed"]:
        raise ValueError("pretrained all-head retirement and logit parity must pass first")
    if root.exists():
        raise FileExistsError("new immutable paired-validation output root required")
    paths = [PARENT / "plan.json", PARENT / "analysis/decision.json", PARENT / "inputs.json"]
    paths += [Path(p) for p in parent["bound_files"]]
    if bitmap_prefix:
        native_gate = Path("/scratch/djy8hg/runs/evicastkv/source-bitmap-prefix-20261107-v1")
        if read(native_gate / "analysis/decision.json")["passed"] is not True:
            raise ValueError("all native bitmap-prefix controls must pass before model timing")
        paths += [native_gate / "plan.json", native_gate / "analysis/decision.json"]
    refs, codes = {}, {}
    for i in (0, 1):
        refs[str(i)] = {}
        for arm, kind in (("storage_only", "storage"), ("hybrid_codec", "hybrid")):
            rowpath = PARENT / f"workers/{i}/{arm}.json"
            row = read(rowpath)
            path = Path(row["logits_path"])
            if row["status"] != "completed" or sha(path) != row["logits_sha256"]:
                raise ValueError("complete frozen model references required")
            refs[str(i)][kind] = str(path)
            paths += [rowpath, path]
        path = CPU / f"workers/{i}/source_codes.pt"
        if sha(path) != read(CPU / f"workers/{i}/source_codes.json")["source_codes_sha256"]:
            raise ValueError("bound integer-code fixture changed")
        paths.append(path)
        codes[str(i)] = str(path)
    storage, hybrid = parent["policies"]["storage_only"], parent["policies"]["hybrid_codec"]
    policies = {
        "storage_only": storage,
        "codec_reference": storage,
        "codec_fast": storage,
        "hybrid_reference": hybrid,
        "hybrid_fast": dict(
            hybrid,
            anchor_counterfactual="single_head",
            reuse_anchor_readout=reuse_anchor_readout,
            anchor_suffix="shared_graph" if shared_suffix else "reference",
            graph_spatial_search="zero_overlap" if planner_exact_fast else "reference",
            reuse_precision_targets=planner_exact_fast,
            prototype_anchor_scope="selected" if planner_exact_fast else "legacy",
            anchor_tail="cuda_graph" if guard_tail else "reference",
        ),
    }
    compact = reuse_anchor_readout or shared_suffix or planner_exact_fast
    arms = ("storage_only", "codec_fast", "hybrid_fast") if compact else ARMS
    policies = {arm: policies[arm] for arm in arms}
    for p in policies.values():
        OverlayPolicy(**dict(p, layers=tuple(p["layers"])))
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as log:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_all_head_source_gpu.py",
                "mosaickv/tests/unit/test_all_head_source_codec.py",
                "mosaickv/tests/unit/test_live_anchor_guard.py",
                "mosaickv/tests/unit/test_guard_readout_reuse.py",
                "mosaickv/tests/unit/test_guard_suffix_replay.py",
                "mosaickv/tests/unit/test_guard_tail_graph.py",
                "mosaickv/tests/unit/test_zero_overlap_spatial.py",
                "mosaickv/tests/unit/test_precision_target_reuse.py",
                "mosaickv/tests/unit/test_source_hybrid_validation.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
            ],
            cwd=SOURCE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise ValueError("CPU controls failed; no submission")
    paths.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            parent=parent["parent"],
            options=parent["options"],
            policies=policies,
            inputs=str(PARENT / "inputs.json"),
            references=refs,
            source_codes=codes,
            warmups=6 if compact else 5,
            repetitions=21 if compact else 20,
            arms=list(arms),
            seed=20261107,
            baseline="evicastkv_storage_only",
            optimized_common_control="codec_fast",
            classification="two_consumed_input_paired_engineering_not_independent_quality",
            comparison="same_source_KV_values_lossless_representation_not_byte_matched_selection",
            prototype_hybrid="guarded_precision_overlay_not_original_all_layer_slot_merging",
            bootstrap_replicates=10000,
            multiplicity_comparisons=6 if compact else 10,
            source_codec_optimization="count_both_pack_only_identical_winner",
            hybrid_optimization="one_changed_head_parent_counterfactual",
            fast_decoder="bitmap_prefix" if bitmap_prefix else "reference",
            reuse_anchor_readout=reuse_anchor_readout,
            shared_suffix=shared_suffix,
            planner_exact_fast=planner_exact_fast,
            cpu_torch_threads=cpu_torch_threads,
            guard_tail=guard_tail,
            tail_operator_gate="bitwise_native_then_capture_inclusive_128_call_ratio_below_0.95"
            if guard_tail
            else "not_used",
            profile_and_verify_suffix_before_timed_trials=shared_suffix,
            independent_quality_or_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    infra.SOURCE = SOURCE
    infra.SHELL = SOURCE / "mosaickv/slurm/source_hybrid_latency.sbatch"
    array = infra.submit(root, "worker")
    write_once(
        root / "admission.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    receipt = dict(
        array=array,
        controller=infra.submit(root, "controller"),
        analyzer=infra.submit(root, "analyze", after=array),
        maximum_gpu_workers=2,
        gpu_count_per_worker=1,
        cpus_per_worker=12,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def operator_controls(path):
    """Actual packet parity, in-place replay and single-head attention on CUDA."""
    import torch

    from mosaickv.evaluation import all_head_source_gpu as native
    from mosaickv.evaluation.eager_core_operator import reference_scores
    from mosaickv.evaluation.live_anchor_guard import head_mask, replace_head_output
    from mosaickv.evaluation.lossless_code_delta_gpu import encode as legacy

    previous, records = None, []
    fixture = torch.load(path, weights_only=True, map_location="cpu")
    for layer in range(32):
        current = fixture[layer].cuda()
        old, alternative = legacy(previous, current), native.encode(previous, current)
        selected = alternative is not None and alternative.serialized_bytes < old.serialized_bytes
        reference = alternative if selected else old
        actual, accepted, legacy_bytes = native.encode_best(previous, current)
        if (
            accepted != selected
            or legacy_bytes != old.serialized_bytes
            or type(actual) is not type(reference)
        ):
            raise RuntimeError("selected-pack construction changed format or tie-breaking")
        if actual.serialized_bytes != reference.serialized_bytes or any(
            not torch.equal(a, b)
            for a, b in zip(actual.tensors(), reference.tensors(), strict=True)
        ):
            raise RuntimeError("selected-pack construction changed serialized packet bits")
        if isinstance(actual, native.DeviceSymbols):
            replay = previous.clone() if previous is not None else torch.empty_like(current)
            if not torch.equal(
                native.decode(replay if previous is not None else None, actual, replay), current
            ):
                raise RuntimeError("in-place source replay changed original codes")
        records.append(dict(layer=layer, bitwise_packet=True, selected=selected))
        previous = current
    for n in (603, 604, 730):
        q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for dtype in (None, torch.bfloat16, torch.float32):
            mask = None if dtype is None else torch.zeros(1, 1, 1, n, device="cuda", dtype=dtype)
            original = torch.matmul(reference_scores(q, k, mask, 128**-0.5), v)
            original = original.transpose(1, 2).contiguous().reshape(1, 1, -1)
            pk, pv = k[:, 4:5].clone() + 0.1, v[:, 4:5].clone() - 0.1
            k[:, 4:5], v[:, 4:5] = pk, pv
            expected = torch.matmul(reference_scores(q, k, mask, 128**-0.5), v)
            expected = expected.transpose(1, 2).contiguous().reshape(1, 1, -1)
            probability = reference_scores(q[:, 4:5], pk, head_mask(mask, 4), 128**-0.5)
            if not torch.equal(replace_head_output(original, probability, pv, 4), expected):
                raise RuntimeError("single-head attention counterfactual differs bitwise")
    torch.cuda.synchronize()
    return dict(passed=True, packets=records, cuda_attention_cases=9)


def worker(root, expected):
    import evicastkv_paper_precision as campaign
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.all_head_source_gpu import use as source_codec
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.storage_six_overlay import integrated_storage
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("identical deterministic CUDA launch required")
    torch.set_num_threads(plan["cpu_torch_threads"])
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    errors, count = [], 0
    try:
        cfg = RunConfig.from_mapping(plan["parent"]["config"])
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=plan["source_sha"],
                config_sha=digest(plan),
                source_dirty=False,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                environment=environment(),
                seed=plan["seed"],
                cuda=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                gpu_count=1,
                backend="huggingface",
                attention="eager",
                precision="bf16",
                measurement_type="paired_consumed_input_latency_not_independent_quality",
                cpu_torch_threads=torch.get_num_threads(),
                fixed_clocks_applied=False,
                **scheduler_identity({}),
            ),
        )
        write_once(folder / "operator.json", operator_controls(plan["source_codes"][str(i)]))
        if plan.get("guard_tail"):
            from mosaickv.evaluation.guard_tail_diagnostic import run as tail_diagnostic

            operator = tail_diagnostic()
            write_once(folder / "tail_operator.json", operator)
            if not operator["passed"]:
                raise RuntimeError("native guard-tail operator did not pass; no model benchmark")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        refs = {
            k: torch.load(p, map_location="cpu", weights_only=True)
            for k, p in plan["references"][str(i)].items()
        }
        features = read(plan["inputs"])[i]
        transport = dict(plan["parent"], options=plan["options"])
        options = campaign.request_options(
            plan["options"]["mosaickv_exact"], transport, "mosaickv_exact", False
        )

        def profile_once(label):
            import cProfile
            import pstats

            profile_policy = OverlayPolicy(
                **dict(
                    plan["policies"]["hybrid_fast"],
                    layers=tuple(plan["policies"]["hybrid_fast"]["layers"]),
                    verify_anchor_suffix=True,
                    record_phase_timings=True,
                )
            )
            profiler = cProfile.Profile()
            with (
                source_codec(True, construction="pack_selected", decoder=plan["fast_decoder"]),
                integrated_storage(adapter, profile_policy, reader_options=options) as evidence,
            ):
                values, detail = profiler.runcall(
                    campaign.execute, adapter, features, transport, "mosaickv_exact", audit=False
                )
            if not torch.equal(values.cpu(), refs["hybrid"]):
                raise RuntimeError("profiled suffix model-logit parity failed")
            guard = evidence["live_anchor_guard"]
            if (
                guard["suffix_graph_replays"] != 127
                or guard["early_middle_late_native_suffix_verifications"] != 3
            ):
                raise RuntimeError("actual early/middle/late native suffix controls missing")
            profiler.dump_stats(str(folder / f"{label}.pstats"))
            with (folder / f"{label}.txt").open("x") as stream:
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(100)
            write_once(
                folder / f"{label}.json",
                dict(
                    classification="consumed_input_instrumented_profile_excluded_from_latency_trials",
                    all_128_logits_bitwise=True,
                    detail=detail,
                    mechanism_evidence=evidence,
                    profiled_source_sha=plan["source_sha"],
                    labels_accessed=False,
                ),
            )
            del values

        if plan["profile_and_verify_suffix_before_timed_trials"]:
            profile_once("cold_profile")
        rng = random.Random(plan["seed"] + i)
        with (folder / "raw.jsonl").open("x") as log:
            for phase, n in (("warmup", plan["warmups"]), ("timed", plan["repetitions"])):
                if phase == "timed" and plan["profile_and_verify_suffix_before_timed_trials"]:
                    profile_once("profile")
                arms = tuple(plan["arms"])
                orders = [list(arms[t % len(arms) :] + arms[: t % len(arms)]) for t in range(n)]
                rng.shuffle(orders)
                for trial, order in enumerate(orders):
                    for arm in order:
                        row = dict(
                            sample_id=features["sample_id"],
                            phase=phase,
                            trial=trial,
                            arm=arm,
                            method_order=order,
                            status="failed",
                            manifest_path=str(folder / "manifest.json"),
                        )
                        try:
                            gc.collect()
                            policy = OverlayPolicy(
                                **dict(
                                    plan["policies"][arm],
                                    layers=tuple(plan["policies"][arm]["layers"]),
                                )
                            )
                            torch.cuda.synchronize()
                            begin = perf_counter()
                            with (
                                source_codec(
                                    arm != "storage_only",
                                    construction="pack_selected"
                                    if arm.endswith("fast")
                                    else "both",
                                    decoder=plan["fast_decoder"]
                                    if arm.endswith("fast")
                                    else "reference",
                                ) as codec,
                                integrated_storage(
                                    adapter, policy, reader_options=options
                                ) as evidence,
                            ):
                                values, detail = campaign.execute(
                                    adapter, features, transport, "mosaickv_exact", audit=False
                                )
                            torch.cuda.synchronize()
                            elapsed = perf_counter() - begin
                            wanted = refs["hybrid" if arm.startswith("hybrid") else "storage"]
                            if not torch.equal(values.cpu(), wanted):
                                raise RuntimeError("128-token model-logit bitwise parity failed")
                            proof = codec.report()
                            if (
                                arm == "hybrid_fast"
                                and plan["shared_suffix"]
                                and evidence["live_anchor_guard"]["suffix_graph_replays"] != 127
                            ):
                                raise RuntimeError("shared suffix did not engage on every update")
                            if arm == "hybrid_fast" and plan.get("guard_tail"):
                                tail = evidence["live_anchor_guard"]["tail_graph"]
                                if tail["replays"] != 127 or tail["captures"] != 1:
                                    raise RuntimeError("native guard-tail replay did not engage")
                            if arm == "hybrid_fast" and plan["reuse_anchor_readout"]:
                                reuse = evidence["live_anchor_guard"]["readout_reuse"]
                                if any(
                                    reuse[k] != 127
                                    for k in ("staged", "normalization_reuses", "vocabulary_reuses")
                                ):
                                    raise RuntimeError(
                                        "guarded readout did not reuse every actual step"
                                    )
                            if arm != "storage_only" and (
                                not proof["all_shared_source_consumers_covered"]
                                or proof["compressed_replacements"]
                                != proof["original_owners_retired"]
                            ):
                                raise RuntimeError("complete all-head source retirement failed")
                            row.update(
                                status="completed",
                                detail=detail,
                                mechanism_evidence=evidence,
                                source_codec_evidence=proof,
                                request_wall_seconds=elapsed,
                                all_128_expected_logits_bitwise=True,
                                storage_parent_greedy_token_agreement=float(
                                    (values.cpu().argmax(-1) == refs["storage"].argmax(-1))
                                    .float()
                                    .mean()
                                ),
                            )
                            del values
                        except Exception:
                            row["error"] = traceback.format_exc()
                            errors.append(row["error"])
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                        log.flush()
                        count += 1
                        if errors:
                            raise RuntimeError("fail-fast numerical/retirement gate")
                    print(json.dumps(dict(phase=phase, completed_trial=trial)), flush=True)
        write_once(folder / "gpu_after.json", asdict(capture_gpu_environment(torch)))
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            row_count=count,
            source_sha=plan["source_sha"],
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    from mosaickv.evaluation.source_hybrid_validation import ordered_trials, paired_summary

    plan = checked(root, expected)
    results, errors = {}, {}
    for i in (0, 1):
        folder = root / f"workers/{i}"
        try:
            if read(folder / "result.json")["status"] != "completed":
                raise ValueError(read(folder / "result.json")["errors"])
            rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            timed = ordered_trials(
                rows,
                warmups=plan["warmups"],
                repetitions=plan["repetitions"],
                arms=tuple(plan["arms"]),
            )
            result = dict(comparisons={}, storage={}, baseline="storage_only")
            for old, new in COMPARISONS:
                if old not in timed or new not in timed:
                    continue
                result["comparisons"][f"{new}_versus_{old}"] = paired_summary(
                    [r["request_wall_seconds"] for r in timed[old]],
                    [r["request_wall_seconds"] for r in timed[new]],
                    seed=plan["seed"] + i,
                    replicates=plan["bootstrap_replicates"],
                    comparisons=plan["multiplicity_comparisons"],
                )
            for arm in plan["arms"]:
                result["storage"][arm] = {
                    key: sorted({r["detail"][key] for r in timed[arm]}) for key in BOUNDARIES
                }
                result["storage"][arm]["greedy_agreement"] = min(
                    r["storage_parent_greedy_token_agreement"] for r in timed[arm]
                )
                result["storage"][arm]["mean_seconds"] = sum(
                    r["request_wall_seconds"] for r in timed[arm]
                ) / len(timed[arm])
            results[str(i)] = result
        except Exception:
            errors[str(i)] = traceback.format_exc()
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            complete=not errors and len(results) == 2,
            results=results,
            errors=errors,
            baseline="evicastkv_storage_only",
            common_source_codec_also_applied_to_storage_control=True,
            six_component_quality_contribution_established=False,
            independent_quality_or_paper_superiority_established=False,
            whole_request_KV_peak_certified=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    lines = [
        "# Paired storage/hybrid engineering comparison",
        "",
        "Two consumed inputs; source coding is common to the optimized storage control.",
        "Peak means decode-owned, not whole-request GPU memory.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Decode peak MiB | "
        "Mean request s | Greedy agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in results.items():
        for arm, data in r["storage"].items():
            sizes = [max(data[k]) / 2**20 for k in BOUNDARIES]
            lines.append(
                f"| {i} | {arm} | {sizes[0]:.4f} | {sizes[1]:.4f} | {sizes[2]:.4f} | "
                f"{data['mean_seconds']:.4f} | {data['greedy_agreement']:.4f} |"
            )
    lines += [
        "",
        f"{plan['warmups']} warmups; {plan['repetitions']} balanced paired repetitions; "
        f"adjusted one-sided intervals cover {plan['multiplicity_comparisons']} comparisons. "
        "See decision.json for all failed gates.",
        "",
        str(errors),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(complete=not errors, errors=errors)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--bitmap-prefix", action="store_true")
    p.add_argument("--reuse-anchor-readout", action="store_true")
    p.add_argument("--shared-suffix", action="store_true")
    p.add_argument("--planner-exact-fast", action="store_true")
    p.add_argument("--cpu-torch-threads", type=int, choices=(1, 2, 12), default=12)
    p.add_argument("--guard-tail", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
        return prepare(
            a.root,
            bitmap_prefix=a.bitmap_prefix,
            reuse_anchor_readout=a.reuse_anchor_readout,
            shared_suffix=a.shared_suffix,
            planner_exact_fast=a.planner_exact_fast,
            cpu_torch_threads=a.cpu_torch_threads,
            guard_tail=a.guard_tail,
        )
    if a.action == "worker":
        return worker(a.root, a.plan_sha)
    if a.action == "analyze":
        return analyze(a.root, a.plan_sha)
    checked(a.root, a.plan_sha)
    return subprocess.call(
        [
            PYTHON,
            str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
            "--policy",
            str(a.root / "admission.json"),
            "--state",
            str(a.root / "admission_state.json"),
            "--apply",
            "--watch",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
