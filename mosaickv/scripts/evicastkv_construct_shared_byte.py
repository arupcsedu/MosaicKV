#!/usr/bin/env python3
"""Construct V10 conditional source witnesses; initial-byte controls, not claims."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import evicastkv_review_source_controls as parent
import evicastkv_review_source_prefix as fixed

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V10 = Path("/scratch/djy8hg/workdir/fox-review-v10/run-20261110-3")
HEADERS = ("cos", "sin", "union", "slots")
CHOICES = ("optional-0.00", "optional-0.03", "optional-0.05")
TARGETS = (47710208, 51380224)
POLICY = dict(
    id="constructed_mixed_source_protection_initial_byte_control_v1",
    targets_initial_bytes=TARGETS,
    frozen_protection_choices=CHOICES,
    witness_source="V10 actual saved per-layer backing costs, no labels or logits",
    topk="existing uniform BF16 dense all-head/all-layer Prompt Top-K",
    history="existing direct lossless source history on both arms",
    shared_headers="bitwise verified common cos/sin/union/slots owners, charged once",
    output_predictions=128,
    continuation_updates=127,
    seed=13,
    original_guard=fixed.POLICY["guard"],
    constructed_initial_equality_required=True,
    final_and_peak_equality_claim=False,
    paper_quality_claim=False,
    fresh_confirmation_authorized=False,
    no_fallback=True,
    worker_soft_seconds=1500,
    worker_hard_seconds=1800,
)


def read(path):
    return json.loads(Path(path).read_text())


def coalesce_headers(families):
    """Deduplicate ONLY proven-identical headers; leave all K/V/code values alone."""
    import torch

    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if set(families) != set(CHOICES):
        raise ValueError("exactly the three registered protection choices required")
    base = families[CHOICES[0]]
    if not base or any(len(body) != len(base) for body in families.values()):
        raise ValueError("complete equal-length layer families required")
    common = {name: getattr(base[0], name) for name in HEADERS}
    result = {}
    for name in CHOICES:
        parts = []
        for packet in families[name]:
            if type(packet) is not SharedSourceFormat:
                raise ValueError("only audited shared-source formats can mix")
            for key, value in common.items():
                current = getattr(packet, key)
                if (
                    not isinstance(current, torch.Tensor)
                    or not isinstance(value, torch.Tensor)
                    or current.shape != value.shape
                    or current.dtype != value.dtype
                    or current.device != value.device
                    or not torch.equal(
                        current.contiguous().view(torch.uint8),
                        value.contiguous().view(torch.uint8),
                    )
                ):
                    raise ValueError(f"cross-family header is not bitwise identical: {key}")
            parts.append(replace(packet, **common))
        result[name] = tuple(parts)
    return result


def choose_constructed(families, witness, target):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    if (
        type(target) is not int
        or target <= 0
        or not witness
        or any(key not in CHOICES for key in witness)
        or any(len(body) != len(witness) for body in families.values())
    ):
        raise ValueError("invalid frozen mixed-layer witness")
    families = coalesce_headers(families)
    selected = tuple(families[key][layer] for layer, key in enumerate(witness))
    actual = backing_bytes(tuple(t for part in selected for t in part.tensors()))
    if actual != target:
        raise ValueError(f"constructed cache costs {actual}, not frozen target {target}")
    return selected


def verify_header_rebinding(adapter, families, witness, selected):
    """Changing shared owner identity must not change any reconstructed K/V."""
    import torch

    blocks = adapter._language_layers()
    if len(blocks) != len(selected):
        raise ValueError("complete model layer geometry required")
    for layer, (choice, packet) in enumerate(zip(witness, selected, strict=True)):
        before = families[choice][layer].materialize(blocks[layer].self_attn)
        after = packet.materialize(blocks[layer].self_attn)
        if any(
            not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
            or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            for a, b in zip(before, after, strict=True)
        ):
            raise RuntimeError("shared-header rebinding changed reconstructed original positions")
        del before, after


def mixed_formats(args, witness, target, proof):
    """Build every registered packet, then select without seeing evaluation tokens."""
    families = {}
    ordinary = parent.protection_masks
    for choice, fraction in zip(CHOICES, (0.0, 0.03, 0.05), strict=True):

        def masks(scores, text_positions, *, seed=13, fraction=fraction):
            return ordinary(scores, text_positions, fraction=fraction, seed=seed)

        with patch.object(parent, "protection_masks", masks):
            packets, _ = parent._formats(*args)
        families[choice] = packets
    selected = choose_constructed(families, witness, target)
    verify_header_rebinding(args[0], families, witness, selected)
    proof.update(
        witness=witness,
        initial_backing_bytes=target,
        shared_headers_verified_bitwise=True,
        shared_header_count=len(HEADERS),
        every_layer_reconstructed_KV_bitwise_unchanged_after_rebinding=True,
        original_source_values_and_logical_positions_unchanged=True,
        mandatory_policy_unchanged=True,
        labels_read=False,
        candidate_family_construction_is_charged_to_request=True,
    )
    del families
    gc.collect()
    return selected, []


def prepare(root):
    revision = clean_source(SOURCE)
    old, jobs, bindings = fixed._bind_parent(fixed.DEFAULT_PARENT)
    feasibility = V10 / "byte-feasibility.json"
    record = read(feasibility)
    if record["common_conditional_initial_targets_bytes"][fixed.ARMS[5]] != list(TARGETS):
        raise ValueError("V10 common conditional targets changed")
    bindings[str(feasibility)] = sha(feasibility)
    jobs_out = []
    for index, job in enumerate(jobs):
        choices = record["inputs"][index]["arms"][fixed.ARMS[5]]
        witnesses = choices["conditional_layer_mix_witnesses"]
        jobs_out.append(dict(job, witnesses={str(t): witnesses[str(t)] for t in TARGETS}))
    write_once(
        Path(root) / "plan.json",
        dict(
            source_sha=revision,
            policy_sha256=digest(POLICY),
            policy=POLICY,
            config=old["config"],
            jobs=jobs_out,
            bindings=bindings,
            expected_rows=8,
        ),
    )
    return dict(workers=1, expected_rows=8)


def load(root):
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy_sha256"] != digest(POLICY) or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("initial-byte protocol changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"bound parent artifact changed: {name}")
    return plan


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = load(root)
    if index != 0 or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("one registered Slurm worker required")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    errors, rows, saved_logits = [], [], {}
    started = time.monotonic()
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one deterministic A100 required")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(12)
        torch.manual_seed(POLICY["seed"])
        cfg = RunConfig.from_mapping(plan["config"])
        if (cfg.model.id, cfg.model.revision) != fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 model required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=config_sha256(cfg),
                plan_sha256=sha(root / "plan.json"),
                model=plan["config"]["model"],
                datasets={
                    str(i): job.get("config", plan["config"])["dataset"]
                    for i, job in enumerate(plan["jobs"])
                },
                processor_revision=fixed.MODEL[1],
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                environment=environment(),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                backend="huggingface",
                attention="eager",
                vllm="not_used",
                sglang="not_used",
                seed=POLICY["seed"],
                measurement_type="constructed_initial_byte_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                slurm_job_id=os.environ["SLURM_JOB_ID"],
            ),
        )
        with (out / "raw.jsonl").open("x") as stream, torch.inference_mode():
            for case, job in enumerate(plan["jobs"]):
                payload = fixed.label_blind_payload(job["inputs"][0])
                mapping = json.loads(json.dumps(job.get("config", plan["config"])))
                mapping["execution"]["seed"] = POLICY["seed"]
                case_config = RunConfig.from_mapping(mapping)
                if (
                    case_config.model != cfg.model
                    or case_config.generation.max_new_tokens != 128
                    or case_config.execution.attention_implementation != "eager"
                ):
                    raise ValueError("per-case pinned model/generation controls differ")
                reference, _ = fixed.native_reference(adapter, payload)
                saved_logits[f"{case}-reference"] = reference
                prefix = fixed.forced_prefix(reference)
                for target in TARGETS:
                    for arm in ("prompt_attention_topk", "mixed_source_protection_v1"):
                        row = dict(
                            case=case,
                            target_initial_bytes=target,
                            arm=arm,
                            status="failed",
                            manifest_path=str(out / "manifest.json"),
                            sample_id=payload["sample_id"],
                            task=payload["task"],
                            dataset=asdict(case_config.dataset),
                            config_sha256=config_sha256(case_config),
                        )
                        try:
                            if time.monotonic() - started > POLICY["worker_soft_seconds"]:
                                raise TimeoutError("bounded construction diagnostic deadline")
                            evidence = {}
                            if arm == "prompt_attention_topk":
                                candidate, report = fixed.request(
                                    adapter,
                                    payload,
                                    case_config,
                                    fixed.ARMS[2],
                                    prefix=prefix,
                                    cap=target,
                                )
                            else:
                                witness = job["witnesses"][str(target)]

                                def builder(
                                    *args,
                                    _target=target,
                                    _witness=witness,
                                    _evidence=evidence,
                                    **kwargs,
                                ):
                                    if kwargs or len(args) != 9 or args[-2] != _target:
                                        raise ValueError("unexpected fixed-prefix factory boundary")
                                    return mixed_formats(args[:-2], _witness, _target, _evidence)

                                with patch.object(fixed, "_formats_with_ceiling", builder):
                                    candidate, report = fixed.request(
                                        adapter,
                                        payload,
                                        case_config,
                                        fixed.ARMS[5],
                                        prefix=prefix,
                                        cap=target,
                                    )
                            actual = report["realized_initial_bytes"]
                            if actual != target or report["used_fullkv_fallback"]:
                                raise ValueError("constructed initial target or no-fallback failed")
                            saved_logits[f"{case}-{target}-{arm}"] = candidate
                            row.update(
                                status="completed",
                                actual_initial_bytes=actual,
                                actual_final_bytes=report["realized_final_bytes"],
                                decode_owned_peak_bytes=report["realized_decode_owned_peak_bytes"],
                                original_logical_length=report["original_logical_length"],
                                final_next_position=report["final_next_position"],
                                construction=evidence,
                                diagnostics=fixed.comparisons(reference, candidate),
                                storage=report["storage"],
                                phase_wall_seconds=report["phase_wall_seconds"],
                                whole_request_allocator_peak_bytes=report[
                                    "whole_request_allocator_peak_bytes"
                                ],
                                whole_request_allocator_reserved_peak_bytes=report[
                                    "whole_request_allocator_reserved_peak_bytes"
                                ],
                                exact_whole_request_KV_peak_certified=False,
                                used_fullkv_fallback=False,
                                same_initial_bytes_only=True,
                            )
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            errors.append(
                                dict(case=case, arm=arm, target=target, error=row["error"])
                            )
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    torch.save(saved_logits, out / "logits.pt")
    write_once(
        out / "result.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            status="completed" if not errors and len(rows) == 8 else "failed",
            errors=errors,
            rows=len(rows),
            bindings={str(p): sha(p) for p in out.iterdir() if p.is_file()},
        ),
    )


def analyze(root):
    import torch

    fixed.analysis_cpu_controls()
    root = Path(root)
    plan, result = load(root), read(root / "worker/result.json")
    if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
        root / "plan.json"
    ):
        raise ValueError("construction provenance changed")
    for name, expected in result["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("construction result changed")
    rows = [json.loads(line) for line in (root / "worker/raw.jsonl").read_text().splitlines()]
    logits = torch.load(root / "worker/logits.pt", map_location="cpu", weights_only=True)
    for row in rows:
        if row["status"] == "completed":
            job = plan["jobs"][row["case"]]
            payload = fixed.label_blind_payload(job["inputs"][0])
            if (
                row["sample_id"] != payload["sample_id"]
                or row["task"] != payload["task"]
                or row["dataset"] != job.get("config", plan["config"])["dataset"]
                or row["final_next_position"] != row["original_logical_length"] + 127
                or row["used_fullkv_fallback"] is not False
            ):
                raise ValueError("sample identity, logical positions or fallback record changed")
            reference = logits[f"{row['case']}-reference"]
            candidate = logits[f"{row['case']}-{row['target_initial_bytes']}-{row['arm']}"]
            fixed.verify_recomputed_diagnostics(
                fixed.comparisons(reference, candidate), row["diagnostics"]
            )
    expected = {
        (i, t, a)
        for i in range(2)
        for t in TARGETS
        for a in ("prompt_attention_topk", "mixed_source_protection_v1")
    }
    complete = (
        len(rows) == len(expected)
        and {(r["case"], r["target_initial_bytes"], r["arm"]) for r in rows} == expected
        and not result["errors"]
        and all(
            r["status"] == "completed" and r["actual_initial_bytes"] == r["target_initial_bytes"]
            for r in rows
        )
    )
    gate = dict(
        status="passed_constructed_initial_bytes" if complete else "blocked",
        complete=complete,
        rows=rows,
        failures=result["errors"],
        original_guard_thresholds=POLICY["original_guard"],
        guard_results=[
            dict(
                case=row["case"],
                target_initial_bytes=row["target_initial_bytes"],
                arm=row["arm"],
                passed=row.get("diagnostics", {})
                .get("registered_guard_32", {})
                .get("registered_guard_pass", False),
            )
            for row in rows
        ],
        final_and_peak_byte_match_claim=False,
        independent_quality_N=0,
        future_three_to_five_target_campaign_complete=False,
        superiority_established=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_constructed_shared_bytes.json", gate)
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(globals()[a.mode](a.root), default=str))


if __name__ == "__main__":
    main()
