#!/usr/bin/env python3
"""Finite two-input phase/precision/capacity screen; no paper promotion."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/runs/evicastkv/funded-delimiter-consumed-20261031-v2")
HISTORY = Path("/scratch/djy8hg/runs/evicastkv/draft-history-consumed-20261031-v2")
METHODS = parent.METHODS
ARMS = ("shadow", "phase33", "phase23_capacity", "phase22_capacity", "post22_control")
RATIOS = (0.5, 0.6, 0.65, 0.7)
VARIANTS = {m + "--" + a for m in METHODS for a in ARMS}


def prepare(root, *, exclude_exact_rows=False):
    if root.exists():
        raise ValueError("new immutable diagnostic root required")
    old = previous.read(OLD / "plan.json")
    bound = dict(old["bound_files"])
    for base in (OLD, HISTORY):
        bound[str(base / "plan.json")] = sha(base / "plan.json")
        for index in range(2):
            folder = base / f"workers/{index}"
            result = previous.read(folder / "result.json")
            if result["status"] != "completed" or result["failures"]:
                raise ValueError("complete consumed producer required")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in result["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer artifact changed")
                bound[str(folder / name)] = digest
    decision = HISTORY / "analysis/decision.json"
    status = previous.read(decision)
    if not status["complete"] or status["model_level_diagnostic_pass"]:
        raise ValueError("preserve completed negative history diagnostic")
    bound[str(decision)] = sha(decision)
    if exclude_exact_rows:
        prior = Path("/scratch/djy8hg/runs/evicastkv/phase-capacity-consumed-20261031-v1")
        status = previous.read(prior / "analysis/decision.json")
        if not status["complete"] or status["failures"] or status["model_level_diagnostic_pass"]:
            raise ValueError("completed negative unconditional phase study required")
        bound[str(prior / "analysis/decision.json")] = sha(prior / "analysis/decision.json")
        for index in range(2):
            folder = prior / f"workers/{index}"
            record = previous.read(folder / "result.json")
            if record["status"] != "completed" or record["failures"]:
                raise ValueError("complete unconditional byte/trajectory controls required")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("unconditional control changed")
                bound[str(folder / name)] = digest
    protocol = (
        SOURCE
        / "mosaickv/docs"
        / (
            "conditional_precision_diagnostic_20261101.md"
            if exclude_exact_rows
            else "phase_capacity_diagnostic_20261031.md"
        )
    )
    bound[str(protocol)] = sha(protocol)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_inverse_phase_capacity_diagnostic",
            exclude_exact_rows_from_body_ranges=exclude_exact_rows,
            variants=sorted(VARIANTS),
            capacity_ratios=RATIOS,
            choose_ratio_by_backing_bytes_only=True,
            original_caps_unchanged=True,
            common_representation_offered_to_all_methods=True,
            greedy_predictions=128,
            history_repair_disabled=True,
            no_untouched_or_paper_claim=True,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
            original_source_root=str(OLD),
        ),
    )
    print(sha(root / "plan.json"))


def kernel_control():
    import torch

    from mosaickv.evaluation.phase_capacity import rotate_cpu
    from mosaickv.evaluation.phase_capacity_kernel import rotate

    generator = torch.Generator().manual_seed(4921)
    ids = torch.arange(0, 66, 2, dtype=torch.int32).repeat(4, 1)
    angles = torch.randn((1, 66, 64), generator=generator).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    reference = torch.randn((1, 4, 48, 128), generator=generator).bfloat16()
    actual = reference.cuda()
    rotate(actual, ids.cuda(), *(t.cuda() for t in phases))
    rotate_cpu(reference, ids, *phases)
    if not torch.equal(reference, actual.cpu()):
        raise RuntimeError("phase kernel differs from BF16 CPU paired-rotation oracle")
    return dict(bitwise_equal=True, strided_prompt_and_untouched_history=True)


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.phase_capacity import PhaseCapture, recode
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture, optimize
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    (out / "raw.jsonl").touch(exist_ok=False)
    name = "setup"
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        write_once(out / "kernel_control.json", kernel_control())
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        if (
            cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("pinned BF16 eager LLaVA required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        ref_folder = Path(plan["parent"]) / f"workers/{index}"
        full = torch.load(ref_folder / "logits.pt", map_location="cpu", weights_only=True)[
            "0-full_kv-ordinary"
        ]
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        if not bool(prepared.model_inputs["attention_mask"].bool().all()):
            raise ValueError("one unpadded consumed input required")
        with (
            PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture,
            TextSourceCapture(adapter, prepared) as text_capture,
            PhaseCapture(adapter) as phase_capture,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = parent.causal.tensor_hash(source)
        layers = adapter._language_layers()
        if len(phase_capture.seen) != len(layers):
            raise RuntimeError("incomplete shared phase capture")
        phases = phase_capture.phases
        manifest = previous.read(ref_folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256(
                json.dumps(dict(base=config_sha256(cfg), plan=expected), sort_keys=True).encode()
            ).hexdigest(),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            baseline_commit=plan["source_sha"],
            evaluator_commit=plan["source_sha"],
            measurement_type="ablation_measured",
            diagnostic_only=True,
            independent_quality_N=0,
            no_shared_byte_or_superiority_claim=True,
            all_five_methods_receive_common_representation=True,
            disabled_components=[
                "future_query_forecast",
                "evidence_graph",
                "value_aware_utility",
                "submodular_selector",
                "prototypes",
                "repair",
            ],
            variant="inverse_phase_precision_funded_capacity",
            actual_timing_protocol=dict(
                warmups=0, repetitions=1, order="fixed", paper_latency_claim=False
            ),
            whole_request_peak_certified=False,
            exclude_exact_rows_from_body_ranges=plan.get(
                "exclude_exact_rows_from_body_ranges", False
            ),
        )
        write_once(out / "manifest.json", manifest)
        raw = tuple(
            ExactFormat(
                "kv",
                tuple(k.shape),
                pair=(
                    PackedBF16(tuple(k.shape), True, (k,)),
                    PackedBF16(tuple(v.shape), True, (v,)),
                ),
            )
            for k, v in source
        )
        forced = tuple(full.argmax(-1)[:33].tolist())

        def fixed_run(formats):
            return storage_runs.run(
                adapter, prefill, formats, tokens=33, forced=forced, sealed=False, audit=False
            )

        ordinary, _ = fixed_run(raw)
        check = comparison(full[:33], ordinary)
        if check["maximum_logit_difference"] > 0.001 or check["token_agreement"] != 1:
            raise RuntimeError("retention-1 original cache replay failed")
        write_once(out / "fullcache_control.json", check)
        constructions = {}
        for method in METHODS:
            name = method + "--construction"
            started = time.monotonic()
            feasible = {}
            ladder = []
            original_cap = None
            for ratio in RATIOS:
                actual_pairs = []

                def observe(layer_index, pair, *, selected=actual_pairs):
                    if layer_index != len(selected):
                        raise RuntimeError("actual baseline pair capture order changed")
                    selected.append(pair)

                packets, proof = bundle(
                    adapter,
                    prepared,
                    prefill,
                    capture,
                    cfg,
                    method,
                    plan["profile_path"],
                    ratio,
                    policies=("head3_boundary8",),
                    allocation_mode="balanced"
                    if method in ("mosaickv_exact", "prompt_attention_topk")
                    else "profile",
                    pair_observer=observe,
                )
                packets = packets["head3_boundary8"]
                positions = proof["source_memberships"]
                if ratio == 0.5:
                    original_cap = backing_bytes(tuple(t for f in packets for t in f.tensors()))
                optimized = tuple(
                    optimize(
                        packet,
                        text_capture.records[i],
                        positions[i],
                        text_capture.positions,
                        layers[i].self_attn,
                    )[0]
                    for i, packet in enumerate(packets)
                )
                if ratio == 0.5:
                    feasible["shadow"] = (optimized, ratio, positions)
                for arm, kb, vb, rotary in (
                    ("phase33", 3, 3, True),
                    ("phase23_capacity", 2, 3, True),
                    ("phase22_capacity", 2, 2, True),
                ):
                    if arm == "phase33" and ratio != 0.5:
                        continue
                    candidate = tuple(
                        packet
                        if i < 2 or i >= len(layers) - 2
                        else recode(
                            packet,
                            actual_pairs[i],
                            positions[i],
                            phases,
                            key_bits=kb,
                            value_bits=vb,
                            pre_rotary=rotary,
                            exclude_exact_rows=plan.get(
                                "exclude_exact_rows_from_body_ranges", False
                            ),
                        )
                        for i, packet in enumerate(optimized)
                    )
                    size = backing_bytes(tuple(t for f in candidate for t in f.tensors()))
                    ladder.append(
                        dict(
                            arm=arm,
                            ratio=ratio,
                            bytes=size,
                            cap_bytes=original_cap,
                            feasible=size <= original_cap,
                        )
                    )
                    if size <= original_cap:
                        feasible[arm] = (candidate, ratio, positions)
                        if arm == "phase22_capacity":
                            post = tuple(
                                packet
                                if i < 2 or i >= len(layers) - 2
                                else recode(
                                    packet,
                                    actual_pairs[i],
                                    positions[i],
                                    phases,
                                    key_bits=2,
                                    value_bits=2,
                                    pre_rotary=False,
                                    exclude_exact_rows=plan.get(
                                        "exclude_exact_rows_from_body_ranges", False
                                    ),
                                )
                                for i, packet in enumerate(optimized)
                            )
                            feasible["post22_control"] = (post, ratio, positions)
                del actual_pairs, optimized, packets, observe
            construction_seconds = time.monotonic() - started
            constructions[method] = dict(
                original_own_cap_bytes=original_cap,
                ladder=ladder,
                construction_seconds=construction_seconds,
            )
            old = torch.load(
                OLD / f"workers/{index}/{method}--source-logits.pt",
                map_location="cpu",
                weights_only=True,
            )
            for arm in ARMS:
                name = method + "--" + arm
                common = dict(
                    variant=name,
                    method=method,
                    task=payload["task"],
                    sample_id=payload["sample_id"],
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    config_sha256=hashlib.sha256(
                        (manifest["config_sha256"] + name).encode()
                    ).hexdigest(),
                    measurement_type="ablation_measured",
                    different_measured_points_not_byte_matched=True,
                    whole_request_peak_certified=False,
                    independent_quality_N=0,
                    exclude_exact_rows_from_body_ranges=arm != "shadow"
                    and plan.get("exclude_exact_rows_from_body_ranges", False),
                )
                if arm not in feasible:
                    row = dict(
                        common,
                        status="unsupported",
                        reason="no_registered_ratio_fits_unchanged_original_cap",
                        attempted_ladder=ladder,
                    )
                else:
                    formats, ratio, memberships = feasible[arm]
                    fixed, _ = fixed_run(formats)
                    greedy, details = storage_runs.run(
                        adapter, prefill, formats, tokens=128, sealed=False, audit=False
                    )
                    torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
                    if arm == "shadow" and (
                        not torch.equal(fixed, old["fixed"])
                        or not torch.equal(greedy, old["greedy"])
                    ):
                        raise RuntimeError(
                            "shadow changed historical selected values or token trajectory"
                        )
                    if details["storage"]["initial_active_kv_storage_bytes"] > original_cap:
                        raise RuntimeError("measured initial cache exceeds unchanged own cap")
                    eos = adapter.model.generation_config.eos_token_id
                    eos_ids = {eos} if type(eos) is int else set(eos or ())
                    scored = parent.scored_text(
                        adapter.processor.tokenizer, greedy.argmax(-1).tolist(), eos_ids
                    )
                    reference = parent.scored_text(
                        adapter.processor.tokenizer, full.argmax(-1).tolist(), eos_ids
                    )
                    row = dict(
                        common,
                        status="completed",
                        actual_source_selection_ratio=ratio,
                        source_membership_sha256=hashlib.sha256(
                            json.dumps(memberships).encode()
                        ).hexdigest(),
                        selected_source_tokens=sum(len(h) for layer in memberships for h in layer),
                        original_own_cap_bytes=original_cap,
                        fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                        fullkv_reference_rouge_l=rouge_l_f1(
                            scored["answer"], (reference["answer"],)
                        ),
                        scored_answer=scored,
                        storage=details["storage"],
                        construction_seconds=construction_seconds,
                        decode_wall_seconds=details["decode_wall_seconds"],
                        decode_cuda_ms=details["decode_cuda_ms"],
                        original_source_unchanged=before == parent.causal.tensor_hash(source),
                        cpu_residual_bytes=0,
                        no_second_prefill_or_draft_token_emission=True,
                        original_logical_positions_and_raw_history_preserved=True,
                    )
                    if not row["original_source_unchanged"]:
                        raise RuntimeError("construction/decode mutated source cache")
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            del feasible
        write_once(out / "constructions.json", constructions)
        result["status"] = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["failures"].append(error)
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        variant=name,
                        error=error,
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                    )
                )
                + "\n"
            )
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for index in range(2):
        folder = root / f"workers/{index}"
        try:
            record = previous.read(folder / "result.json")
            if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != expected:
                raise ValueError("producer provenance changed")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer artifact changed")
            failures.extend(record["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    if plan.get("exclude_exact_rows_from_body_ranges", False):
        prior = Path("/scratch/djy8hg/runs/evicastkv/phase-capacity-consumed-20261031-v1")
        previous_rows = {}
        for index in range(2):
            for line in (prior / f"workers/{index}/raw.jsonl").read_text().splitlines():
                old = json.loads(line)
                previous_rows[(old.get("sample_id"), old.get("variant"))] = old
        for row in rows:
            old = previous_rows.get((row.get("sample_id"), row.get("variant")))
            matched = old is not None and row["status"] == old["status"]
            if matched and row["status"] == "completed":
                matched = row["source_membership_sha256"] == old[
                    "source_membership_sha256"
                ] and all(
                    row["storage"][field] == old["storage"][field]
                    for field in (
                        "initial_active_kv_storage_bytes",
                        "final_active_kv_storage_bytes",
                        "peak_active_kv_storage_bytes",
                    )
                )
            row["same_bytes_and_memberships_as_unconditional_control"] = matched
            if not matched:
                failures.append(
                    "conditional arm differs in membership/storage/status: "
                    + str(row.get("variant"))
                )
    opportunities = []
    for arm in ARMS[1:]:
        decisions = []
        for job in plan["jobs"]:
            sample = previous.read(job["inputs"][0])["sample_id"]
            group = {
                r["variant"]: r
                for r in rows
                if r.get("sample_id") == sample and r["status"] == "completed"
            }
            anchor, candidate = (
                group.get("mosaickv_exact--shadow"),
                group.get("mosaickv_exact--" + arm),
            )
            decisions.append(
                bool(
                    anchor
                    and candidate
                    and candidate["fullkv_fixed_prefix"]["registered_guard_pass"]
                    and candidate["fullkv_reference_rouge_l"] >= anchor["fullkv_reference_rouge_l"]
                )
            )
        opportunities.append(
            dict(arm=arm, passes_both_consumed_inputs=len(decisions) == 2 and all(decisions))
        )
    complete = (
        len(rows) == 50
        and not failures
        and all(r["status"] in ("completed", "unsupported") for r in rows)
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=complete,
            failures=failures,
            rows=rows,
            opportunities=opportunities,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            model_level_diagnostic_pass=complete
            and any(o["passes_both_consumed_inputs"] for o in opportunities),
            independent_quality_N=0,
            superiority_established=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def submit(root, expected):
    original = parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_phase_capacity_diagnostic.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(parent.subprocess, "check_output", scoped):
        parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "prepare_exclusion", "cpu", "submit", "worker", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "prepare_exclusion":
        prepare(args.root, exclude_exact_rows=True)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
