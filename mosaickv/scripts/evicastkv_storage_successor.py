#!/usr/bin/env python3
"""Finite consumed-input compact replay -> timing -> source-sharing quality diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_early_prefix as messages_api
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.config import RunConfig
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/exact-storage-stages-20261030-v1-retry1/stage-4")
COMPACT_PARENT = Path("/scratch/djy8hg/runs/evicastkv/quality-latency-consumed-20261030-v1/compact")
BASE = Path("/scratch/djy8hg/runs/evicastkv/quality-latency-compact-direct-20261030-v2")
PHASES = ("compact", "timing", "layer_addition")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
    "mosaickv_exact_source_shared_plus2",
    "prompt_attention_topk_source_shared_plus2",
)


def prepare(root, phase):
    source = clean_source(SOURCE)
    if root.exists() or phase not in PHASES:
        raise ValueError("new immutable root and a registered phase required")
    previous = common.read(PARENT / "plan.json")
    gate = common.read(PARENT / "analysis/GATE.json")
    if gate["status"] != "complete_exact_storage_diagnostic" or gate["failures"]:
        raise ValueError("completed exact-storage controls required")
    bound = dict(previous["bound_files"])
    compact_parent = COMPACT_PARENT
    prior_gate = common.read(compact_parent / "analysis/GATE.json")
    if prior_gate["status"] != "complete_consumed_engineering_diagnostic" or prior_gate["failures"]:
        raise ValueError("original compact replay must pass before fused replay")
    for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
        bound[str(compact_parent / name)] = sha(compact_parent / name)
    for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
        bound[str(PARENT / name)] = sha(PARENT / name)
    offset = PHASES.index(phase)
    if offset:
        prior = root.parent / PHASES[offset - 1]
        gate = common.read(prior / "analysis/GATE.json")
        if gate["status"] != "complete_consumed_engineering_diagnostic" or gate["failures"]:
            raise ValueError("previous completeness/numerical controls failed")
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(prior / name)] = sha(prior / name)
    jobs = previous["jobs"]
    if phase == "baselines":
        jobs = [
            dict(job, index=len(METHODS) * offset + i, sample_index=offset, method=method)
            for offset, job in enumerate(jobs)
            for i, method in enumerate(METHODS)
        ]
    plan = dict(
        source_sha=source,
        bound_files=bound,
        jobs=jobs,
        phase=phase,
        protocol="compact_direct_shared_bit_transport_v2",
        same_memberships=phase != "layer_addition",
        selector_changes_authorized=phase == "layer_addition",
        bounded_selection_scope="one_layer_restore_only" if phase == "layer_addition" else None,
        fused_compact_direct=True,
        layer_addition_draft_steps=4,
        layer_addition_maximum_plans=8,
        layer_addition_no_donor_removal=True,
        requested_workers=len(jobs),
        maximum_a100s=6,
        cpus_per_worker=12,
        wall_minutes=30,
        output_tokens=32,
        original_KV_bits_required=True,
        unchanged_32_token_guard=True,
        patch_repetitions=3,
        projection_reduction_policies=[True, False],
        no_benchmark_label_access=True,
        same_two_consumed_identities=True,
        topup_slots=[2, 8, 32],
        topup_initial_cap_bytes=62 * 2**20,
        topup_peak_cap_bytes=96 * 2**20,
        topup_no_donor_eviction=True,
        warmups=5,
        timed_repetitions=20,
        timing_seed=20261030,
        same_storage_choices_offered_to_both_selectors=True,
        timings_exclude_prefill_and_original_selection=True,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        whole_request_peak_certified=False,
        finite_next_phase_authorized=offset < len(PHASES) - 1,
        maximum_phases=len(PHASES),
        baseline_methods=list(METHODS),
        baseline_selection_budget="same_frozen_source_slots_before_lossless_storage",
        exact_encoded_byte_matching_claim=False,
        absolute_kl_limit=0.02,
        absolute_logit_limit=1.0,
        guard=dict(
            horizon=32, checkpoints=[1, 4, 16, 32], absolute_tolerance=1e-7, relative_tolerance=1e-6
        ),
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def choices(adapter, prefill, capture, positions, *, compact):
    import torch

    from mosaickv.evaluation.compact_storage import build_compact
    from mosaickv.evaluation.exact_storage import formats_for_layer

    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    variants = {"kv": [], "token": [], "compact": []}
    proofs = []
    for layer, ((key, value), heads) in enumerate(zip(pairs, positions, strict=True)):
        attention = adapter._language_layers()[layer].self_attn
        candidates, proof = formats_for_layer(capture.records[layer], attention, key, value, heads)
        kv = next(f for f in candidates if f.kind == "kv")
        token = min(candidates, key=lambda f: (f.component_bytes, f.kind))
        variants["kv"].append(kv)
        variants["token"].append(token)
        compact_choices = [kv]
        records = []
        if compact:
            index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
                1, key.shape[1], len(heads[0]), key.shape[-1]
            )
            references = tuple(t.gather(2, index) for t in (key, value))
            for candidate in candidates:
                if candidate.archive is None:
                    continue
                for policy in (True, False):
                    replay, record = build_compact(
                        candidate, attention, references, reduced_precision=policy
                    )
                    compact_choices.append(replay)
                    records.append(record)
        chosen = min(compact_choices, key=lambda f: (f.component_bytes, f.kind))
        variants["compact"].append(chosen)
        proofs.append(
            dict(
                layer=layer,
                original_format=token.kind,
                chosen_compact=chosen.kind,
                **proof,
                compact_candidates=records,
            )
        )
    return {name: tuple(values) for name, values in variants.items()}, proofs


def prepare_lookm_retry(root):
    parent = Path("/scratch/djy8hg/runs/evicastkv/quality-latency-consumed-20261030-v1/baselines")
    source = clean_source(SOURCE)
    if root.exists():
        raise ValueError("LOOK-M retry root must be new")
    original = common.read(parent / "plan.json")
    gate = common.read(parent / "analysis/GATE.json")
    if gate["status"] != "failed" or {f["index"] for f in gate["failures"]} != {4, 12}:
        raise ValueError("retry only the two explicitly failed LOOK-M workers")
    jobs = [job for job in original["jobs"] if job["method"] == "lookm_reimpl"]
    if [j["index"] for j in jobs] != [4, 12]:
        raise ValueError("unexpected LOOK-M identities")
    bound = dict(original["bound_files"])
    for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
        bound[str(parent / name)] = sha(parent / name)
    original.update(
        source_sha=source,
        bound_files=bound,
        requested_workers=2,
        protocol="lookm_balanced_cap_planning_repair_retry_only_v1",
        jobs=[dict(job, index=i, parent_index=job["index"]) for i, job in enumerate(jobs)],
        finite_next_phase_authorized=False,
        retry_scope="only_failed_LOOKM_indices_4_12_original_inputs_and_settings",
    )
    write_once(root / "plan.json", original)
    return sha(root / "plan.json")


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.evaluation.compact_direct_kernel import gpu_controls as compact_direct_controls
    from mosaickv.evaluation.delimiter_control import source_sha
    from mosaickv.evaluation.exact_storage import gpu_controls
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.projection_archive import ProjectionCapture
    from mosaickv.evaluation.projection_delta import gpu_controls as delta_controls
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        status="failed",
        failures=[],
        task=job["task"],
        sample_id=job["sample_id"],
        phase=plan["phase"],
    )
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace missing")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        write_once(
            out / "synthetic_cuda_controls.json",
            dict(strided=gpu_controls(), delta=delta_controls()),
        )
        if plan["phase"] != "baselines":
            write_once(out / "compact_direct_cuda_controls.json", compact_direct_controls())
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        messages = messages_api.redacted_messages(common.read(job["input"]))
        parity = compare_with_generate(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
        if parity.token_agreement != 1 or parity.maximum_logit_difference > 0.001:
            raise RuntimeError("ordinary FullKV parity failed")
        write_once(out / "ordinary_parity.json", asdict(parity))
        with ProjectionCapture(adapter) as capture:
            prefill = adapter.prefill(
                adapter.prepare_inputs(messages),
                capture_queries=False,
                capture_attentions=plan["phase"] in ("evidence", "baselines"),
                attention_query_window=None if plan["phase"] == "baselines" else 16,
            )
        capture.validate()
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = source_sha(source)
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=expected,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                reference_artifact_sha256=sha(Path(job["delimiter_worker"]) / "captures.pt"),
                measurement_type="consumed_storage_quality_latency_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                annotation_targets_used=False,
                whole_request_peak_certified=False,
            ),
        )
        frozen = torch.load(
            Path(job["delimiter_worker"]) / "captures.pt", weights_only=True, map_location="cpu"
        )
        full_positions = tuple(
            tuple(tuple(range(k.shape[-2])) for _ in range(k.shape[1])) for k, _ in source
        )
        full_reference = standard_greedy_reference(adapter, prefill, full_positions)
        torch.save(full_reference, out / "fullkv-reference.pt")
        if plan["phase"] == "baselines":
            rows = baseline_study(
                adapter,
                prefill,
                capture,
                frozen,
                job,
                cfg,
                full_reference,
                adapter.prepare_inputs(messages),
                out,
                plan,
            )
            if source_sha(source) != before:
                raise RuntimeError("baseline diagnostic mutated original prefill")
            write_once(out / "controls.json", rows)
            result["status"] = "completed"
            return
        if plan["phase"] == "compact":
            identity_formats, identity_proof = choices(
                adapter, prefill, capture, full_positions, compact=True
            )
            actual, identity_record = run(adapter, prefill, identity_formats["compact"], audit=True)
            if not torch.equal(actual, full_reference):
                raise RuntimeError("retention-1 compact replay changed ordinary FullKV logits")
            write_once(
                out / "retention_one.json",
                dict(bitwise_logits=True, proof=identity_proof, **identity_record),
            )
            del identity_formats, identity_proof, actual
        rows = []
        for arm in job["arms"]:
            role = arm["role"]
            positions = frozen[role]["anchor"]["positions"]
            if plan["phase"] == "layer_addition":
                rows.append(
                    layer_addition(
                        adapter, prefill, capture, positions, full_reference, role, out, plan
                    )
                )
                continue
            if plan["phase"] == "evidence":
                rows.extend(
                    evidence(
                        adapter,
                        prefill,
                        capture,
                        positions,
                        full_reference,
                        role,
                        out,
                        plan,
                        adapter.prepare_inputs(messages).model_inputs["input_ids"],
                    )
                )
                continue
            representations, proof = choices(adapter, prefill, capture, positions, compact=True)
            write_once(out / f"{role}-format-proof.json", proof)
            reference = standard_greedy_reference(adapter, prefill, positions)
            torch.save(reference, out / f"{role}-ordinary-reference.pt")
            if plan["phase"] == "compact":
                for name, formats in representations.items():
                    actual, row = run(adapter, prefill, formats, audit=True)
                    torch.save(actual, out / f"{role}-{name}-logits.pt")
                    if not torch.equal(actual, reference):
                        raise RuntimeError(f"{role}/{name} changed original greedy logits")
                    rows.append(dict(role=role, format=name, bitwise_logits=True, **row))
            else:
                trials = timing(adapter, prefill, representations, reference, role, plan)
                write_once(out / f"{role}-trials.json", trials)
                rows.extend(trials)
        if source_sha(source) != before:
            raise RuntimeError("original prefill mutated")
        write_once(out / "controls.json", rows)
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def timing(adapter, prefill, representations, reference, role, plan):
    import gc

    import torch

    from mosaickv.evaluation.storage_runs import run

    generator = random.Random(plan["timing_seed"])
    rows = []
    for trial in range(-plan["warmups"], plan["timed_repetitions"]):
        order = list(representations)
        generator.shuffle(order)
        for order_index, name in enumerate(order):
            gc.collect()
            actual, row = run(adapter, prefill, representations[name], audit=False)
            if not torch.equal(actual, reference):
                raise RuntimeError("timed compact transport changed original logits")
            row.pop("storage")  # Instrumented phase owns memory evidence, not this timing.
            rows.append(
                dict(
                    trial=trial,
                    warmup=trial < 0,
                    order_index=order_index,
                    role=role,
                    format=name,
                    bitwise_logits=True,
                    **row,
                )
            )
    return rows


def evidence(adapter, prefill, capture, positions, full_reference, role, out, plan, input_ids):
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability
    from mosaickv.config import DraftAllocationGuardConfig
    from mosaickv.evaluation.delimiter_control import compare_prefix_logits, delimiter_positions
    from mosaickv.evaluation.storage_runs import run, source_topup

    forced = tuple(full_reference.argmax(-1).tolist())
    base, _ = choices(adapter, prefill, capture, positions, compact=False)
    anchor, anchor_row = run(adapter, prefill, base["token"], forced=forced, audit=True)
    write_once(out / f"{role}-anchor-control.json", anchor_row)
    torch.save(anchor, out / f"{role}-anchor-forced-logits.pt")
    delimiter = delimiter_positions(input_ids.flatten().tolist(), adapter.processor.tokenizer)
    rows = []
    for slots in plan["topup_slots"]:
        candidate, transaction = source_topup(
            positions, prefill.attention_weights, slots=slots, delimiter=delimiter
        )
        formats, proof = choices(adapter, prefill, capture, candidate, compact=False)
        actual, row = run(adapter, prefill, formats["token"], forced=forced, audit=True)
        torch.save(actual, out / f"{role}-plus-{slots}-forced-logits.pt")
        per_prefix = compare_prefix_logits(full_reference, anchor, actual)
        verdict = decide_draft_stability(
            per_prefix, DraftAllocationGuardConfig.from_mapping(plan["guard"])
        )
        absolute = all(
            r["candidate"]["logit_kl"] <= 0.02 and r["candidate"]["maximum_logit_difference"] <= 1
            for r in per_prefix
        )
        storage = row["storage"]
        fits = (
            storage["initial_active_kv_storage_bytes"] <= plan["topup_initial_cap_bytes"]
            and storage["peak_active_kv_storage_bytes"] <= plan["topup_peak_cap_bytes"]
        )
        ordinary = standard_reference(adapter, prefill, candidate)
        own, own_row = run(adapter, prefill, formats["token"], audit=True)
        if not torch.equal(own, ordinary):
            raise RuntimeError("topup transport changed its original selected-cache logits")
        changed = sum(e["added_count"] for e in transaction["events"])
        meaningful = changed > 0 and (
            any(r["candidate"]["logit_kl"] < r["anchor"]["logit_kl"] - 1e-7 for r in per_prefix)
            or any(
                not r["anchor"]["greedy_token_agreement"]
                and r["candidate"]["greedy_token_agreement"]
                for r in per_prefix
            )
        )
        record = dict(
            role=role,
            extra_slots=slots,
            **row,
            transaction=transaction,
            format_proof=proof,
            per_prefix=per_prefix,
            guard=verdict,
            absolute_limits_passed=absolute,
            memory_cap_passed=fits,
            meaningful=meaningful,
            diagnostic_accepted=bool(verdict["accepted"] and absolute and fits and meaningful),
            own_greedy_token_ids=own.argmax(-1).tolist(),
            fullkv_token_ids=forced,
            own_greedy_storage=own_row["storage"],
            bitwise_transport=True,
            annotation_targets_used=False,
            budget_comparison="fixed_hard_cap_not_equal_achieved_bytes",
        )
        write_once(out / f"{role}-plus-{slots}.json", record)
        rows.append(record)
    return rows


def standard_reference(adapter, prefill, positions):
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference

    return standard_greedy_reference(adapter, prefill, positions)


def layer_addition(adapter, prefill, capture, positions, full_reference, role, out, plan):
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability
    from mosaickv.config import DraftAllocationGuardConfig
    from mosaickv.evaluation.delimiter_control import compare_prefix_logits
    from mosaickv.evaluation.draft_addition import collect_draft_gains, layer_addition_plans
    from mosaickv.evaluation.storage_runs import run

    gains, draft = collect_draft_gains(adapter, prefill, positions, steps=4)
    proposals = layer_addition_plans(positions, gains, maximum=8)
    torch.save(tuple(g.cpu() for g in gains), out / f"{role}-prospective-draft-gains.pt")
    layer_scores = [
        dict(layer=i, head_maxima=g.max(-1).values.cpu().tolist()) for i, g in enumerate(gains)
    ]
    write_once(out / f"{role}-prospective-layer-scores.json", layer_scores)
    forced = tuple(full_reference.argmax(-1).tolist())
    formats, _ = choices(adapter, prefill, capture, positions, compact=True)
    anchor, anchor_record = run(adapter, prefill, formats["compact"], forced=forced, audit=True)
    torch.save(anchor, out / f"{role}-layer-addition-anchor.pt")
    records = []
    for i, proposal in enumerate(proposals):
        formats, format_proof = choices(
            adapter, prefill, capture, proposal["positions"], compact=True
        )
        actual, record = run(adapter, prefill, formats["compact"], forced=forced, audit=True)
        ordinary = standard_reference(adapter, prefill, proposal["positions"])
        own, own_record = run(adapter, prefill, formats["compact"], audit=True)
        if not torch.equal(own, ordinary):
            raise RuntimeError("single-layer addition changed ordinary selected-cache logits")
        torch.save(
            dict(forced_logits=actual, own_logits=own), out / f"{role}-layer-addition-{i}.pt"
        )
        per_prefix = compare_prefix_logits(full_reference, anchor, actual)
        verdict = decide_draft_stability(
            per_prefix, DraftAllocationGuardConfig.from_mapping(plan["guard"])
        )
        safe = all(
            r["candidate"]["logit_kl"] <= 0.02 and r["candidate"]["maximum_logit_difference"] <= 1.0
            for r in per_prefix
        )
        storage = record["storage"]
        fits = (
            storage["initial_active_kv_storage_bytes"] <= 62 * 2**20
            and storage["peak_active_kv_storage_bytes"] <= 96 * 2**20
        )
        improved = any(
            r["candidate"]["logit_kl"] < r["anchor"]["logit_kl"] - 1e-7 for r in per_prefix
        )
        row = dict(
            proposal_index=i,
            **{k: v for k, v in proposal.items() if k != "positions"},
            per_prefix=per_prefix,
            guard=verdict,
            absolute_limits_passed=safe,
            memory_cap_passed=fits,
            bitwise_transport=True,
            **record,
            own_greedy_storage=own_record["storage"],
            own_token_ids=own.argmax(-1).tolist(),
            diagnostic_accepted=bool(verdict["accepted"] and safe and fits and improved),
            format_proof=format_proof,
        )
        write_once(out / f"{role}-layer-addition-{i}.json", row)
        records.append(row)
    return dict(
        role=role,
        draft_provenance=draft,
        anchor=anchor_record,
        proposals=records,
        complete_plans=len(records),
        benchmark_labels_used=False,
        diagnostic_accepted=bool(records and records[0]["diagnostic_accepted"]),
        prospective_policy="top1_signed_draft_gain_no_32_prefix_selection",
        oracle_any_diagnostic_pass=any(r["diagnostic_accepted"] for r in records),
    )


def baseline_study(
    adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
):
    import torch

    from mosaickv.evaluation.delimiter_control import compare_prefix_logits, delimiter_positions
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.storage_baselines import baseline_payload, ordinary_pairs_reference
    from mosaickv.evaluation.storage_runs import run, source_topup

    method = job["method"]
    base = method.removesuffix("_source_shared_plus2")
    profile = cfg.bounded_decode.incremental_history.layer_profile_path
    positions, pairs, proof = baseline_payload(
        adapter, prepared, prefill, cfg, base, frozen, profile
    )
    if method != base:
        delimiter = delimiter_positions(
            prepared.model_inputs["input_ids"].flatten().tolist(), adapter.processor.tokenizer
        )
        positions, transaction = source_topup(
            positions,
            tuple(p[..., -16:, :] for p in prefill.attention_weights),
            slots=2,
            delimiter=delimiter,
        )
        proof.update(
            selection_variant=method,
            source_sharing_transaction=transaction,
            same_source_capacity_after_topup=False,
        )
    started = time.monotonic()
    if pairs is None:
        representations, format_proof = choices(adapter, prefill, capture, positions, compact=True)
        original = standard_reference(adapter, prefill, positions)
    else:
        # LOOK-M's actual merged tensors, not the original projected source.
        representations = {
            "kv": tuple(
                ExactFormat("kv", tuple(key.shape), pair=(encode(key), encode(value)))
                for key, value in pairs
            )
        }
        format_proof = [
            dict(layer=i, source_eligible=False, original_transformation_preserved=True)
            for i in range(len(pairs))
        ]
        original = ordinary_pairs_reference(adapter, prefill, pairs)
    torch.cuda.synchronize()
    constructor = time.monotonic() - started
    write_once(out / "baseline_selection.json", proof)
    write_once(out / "baseline_formats.json", format_proof)
    torch.save(original, out / "baseline-ordinary-logits.pt")
    forced = tuple(full_reference.argmax(-1).tolist())
    controls = []
    from mosaickv.generation import decode_quality_output, terminal_token_ids

    terminals = terminal_token_ids(adapter.model, adapter.processor)
    reference_text = decode_quality_output(adapter.processor, forced, terminals).text
    texts = []
    for name, formats in representations.items():
        own, record = run(adapter, prefill, formats, audit=True)
        if not torch.equal(own, original):
            raise RuntimeError(f"{method}/{name} altered its actual baseline K/V output")
        shadow, forced_record = run(adapter, prefill, formats, forced=forced, audit=True)
        per_prefix = compare_prefix_logits(full_reference, shadow, shadow)
        token_ids = own.argmax(-1).tolist()
        quality = decode_quality_output(adapter.processor, token_ids, terminals)
        text = quality.text
        log_probs = shadow.log_softmax(-1)
        nll = float(-log_probs[torch.arange(32), torch.tensor(forced)].sum())
        torch.save(
            dict(own_greedy_logits=own, fullkv_prefix_logits=shadow), out / f"{name}-logits.pt"
        )
        controls.append(
            dict(
                format=name,
                bitwise_baseline_transport=True,
                **record,
                fixed_prefix_storage=forced_record["storage"],
                fullkv_prefix_mean_kl=sum(x["candidate"]["logit_kl"] for x in per_prefix) / 32,
                fullkv_prefix_max_logit_difference=max(
                    x["candidate"]["maximum_logit_difference"] for x in per_prefix
                ),
                fullkv_prefix_token_agreement=sum(
                    x["candidate"]["greedy_token_agreement"] for x in per_prefix
                )
                / 32,
                own_greedy_token_agreement=float(
                    (own.argmax(-1) == full_reference.argmax(-1)).float().mean()
                ),
                fullkv_reference_rouge_l_f1=rouge_l_f1(text, [reference_text]),
                fullkv_generated_prefix_nll_sum=nll,
                fullkv_generated_prefix_token_count=32,
                not_annotation_ppl_or_official_benchmark_score=True,
            )
        )
        texts.append(
            dict(
                format=name,
                answer=text,
                reference=reference_text,
                token_ids=token_ids,
                quality_decode=quality.to_record(),
            )
        )
    write_once(out / "predictions.json", texts)
    trials = (
        timing(adapter, prefill, representations, original, method, plan)
        if method != "full_kv"
        else []
    )
    write_once(out / "trials.json", trials)
    return [
        dict(
            method=method,
            selection=proof,
            controls=controls,
            construction_and_reference_seconds=constructor,
            trials=trials,
            fullkv_excluded_from_latency=method == "full_kv",
            no_superiority_or_statistical_quality_claim=True,
        )
    ]


def analyze(root, expected):
    plan = common.inputs(root, expected)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    rows, failures, inputs = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            failures.append(dict(index=job["index"], error="missing worker"))
            continue
        result = common.read(path)
        inputs[str(path)] = sha(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("worker provenance changed")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker evidence changed")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(index=job["index"], errors=result["failures"]))
            continue
        rows.extend(
            dict(task=job["task"], sample_id=job["sample_id"], **r)
            for r in common.read(folder / "controls.json")
        )
    expected_count = (
        4
        if plan["phase"] == "layer_addition"
        else len(plan["jobs"])
        if plan["phase"] == "baselines"
        else 300
        if plan["phase"] == "timing"
        else 12
    )
    complete = not failures and len(rows) == expected_count
    summaries = []
    if plan["phase"] == "timing" and complete:
        import numpy as np

        for task in sorted({r["task"] for r in rows}):
            for role in sorted({r["role"] for r in rows}):
                table = [
                    r for r in rows if r["task"] == task and r["role"] == role and not r["warmup"]
                ]
                for name in ("kv", "token", "compact"):
                    values = [r["decode_wall_seconds"] for r in table if r["format"] == name]
                    base = {
                        r["trial"]: r["decode_wall_seconds"] for r in table if r["format"] == "kv"
                    }
                    ratios = [
                        r["decode_wall_seconds"] / base[r["trial"]]
                        for r in table
                        if r["format"] == name
                    ]
                    summaries.append(
                        dict(
                            task=task,
                            role=role,
                            format=name,
                            timed_repetitions=len(values),
                            median_seconds=float(np.median(values)),
                            p95_seconds=float(np.percentile(values, 95)),
                            median_paired_ratio_vs_encoded_KV=float(np.median(ratios)),
                        )
                    )
    accepted = [
        dict(task=r["task"], role=r["role"], extra_slots=r.get("extra_slots", "single_layer_slot"))
        for r in rows
        if r.get("diagnostic_accepted")
    ]
    gate = dict(
        status="complete_consumed_engineering_diagnostic" if complete else "failed",
        phase=plan["phase"],
        failures=failures,
        completed_rows=len(rows),
        expected_rows=expected_count,
        controls=rows,
        timing_summaries=summaries,
        accepted_consumed_diagnostics=accepted,
        superiority_established=False,
        independent_quality_established=False,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        whole_request_peak_certified=False,
    )
    write_once(out / "GATE.json", gate)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs=inputs,
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    print(json.dumps(dict(status=gate["status"], failures=failures, accepted=accepted)))
    if complete and plan["finite_next_phase_authorized"]:
        offset = PHASES.index(plan["phase"])
        child = root.parent / PHASES[offset + 1]
        digest = prepare(child, PHASES[offset + 1])
        launch.submit(
            child,
            digest,
            script_name="evicastkv_storage_successor.sbatch",
            worker_minutes=30,
            memory_gb=24,
        )
        write_once(
            out / "advancement.json",
            dict(root=str(child), plan_sha256=digest, no_fresh_campaign=True),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root, args.phase))
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_storage_successor.sbatch",
            worker_minutes=30,
            memory_gb=24,
        )
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
