#!/usr/bin/env python3
"""Readout-aware CPU audit of frozen consumed-input V8.1 captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml
from bounded_history_pilot import SOURCE, clean_source, digest, write_once

from mosaickv.evaluation.readout_audit import (
    PROSPECTIVE_SCORES,
    SCORE_PROVENANCE,
    FrozenReadout,
    ReadoutAuditPolicy,
    audit_ranking,
    kl_divergence,
    projected_residual_gain,
    rank_correlation,
    squared_error_gain,
)

PROTOCOL = SOURCE / "mosaickv/configs/experiments/bounded_readout_v8_2.yaml"
DOWNSTREAM = ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")


def validate_protocol(protocol):
    if (
        any(protocol[k] for k in DOWNSTREAM)
        or ReadoutAuditPolicy(**protocol["screen"]) != ReadoutAuditPolicy()
    ):
        raise ValueError("readout audit cannot change scope or descriptive screening policy")
    if protocol["registered_guard"] != dict(
        horizon=32, absolute_tolerance=1e-7, relative_tolerance=1e-6
    ):
        raise ValueError("original registered stability thresholds must remain unchanged")


def prepare(root, protocol_path=PROTOCOL):
    source_sha = clean_source()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.resolve().parents:
        raise ValueError("readout audit needs a new immutable scratch root")
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    parent = Path(protocol["parent_root"]).resolve()
    report_path = parent / "derived-analysis-v1/report.json"
    if digest(report_path) != protocol["parent_report_sha256"]:
        raise ValueError("V8.1 parent audit report changed")
    report = json.loads(report_path.read_text())
    if report["source_sha"] != protocol["parent_source_sha"]:
        raise ValueError("wrong V8.1 producer source")
    bound = {str(report_path): digest(report_path), str(protocol_path): digest(protocol_path)}
    for path, sha in report["artifacts"].items():
        if digest(path) != sha:
            raise ValueError("V8.1 capture/provenance artifact changed")
        bound[path] = sha
    plan = dict(
        source_sha=source_sha,
        parent_root=str(parent),
        protocol=protocol,
        bound_files=bound,
        new_identities=0,
        **dict.fromkeys(DOWNSTREAM, False),
    )
    write_once(root / "plan.json", plan)
    return plan


def load_plan(root, expected_sha):
    if digest(root / "plan.json") != expected_sha:
        raise ValueError("readout plan changed")
    plan = json.loads((root / "plan.json").read_text())
    clean_source(plan["source_sha"])
    validate_protocol(plan["protocol"])
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError("readout input/provenance changed")
    return plan


def tensor_sha(tensor):
    return hashlib.sha256(memoryview(tensor.contiguous().numpy()).cast("B")).hexdigest()


def load_readout(model):
    """Read ONLY cached final norm and lm_head; no model loader or download."""
    from safetensors import safe_open

    if (
        model["id"] != "llava-hf/llava-1.5-7b-hf"
        or model["revision"] != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    ):
        raise ValueError("readout audit supports only the original pinned LLaVA checkpoint")
    snapshot = (
        Path(os.environ["HF_HUB_CACHE"])
        / "models--llava-hf--llava-1.5-7b-hf/snapshots"
        / model["revision"]
    )
    config = json.loads((snapshot / "config.json").read_text())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    names = ("language_model.model.norm.weight", "language_model.lm_head.weight")
    tensors, metadata = [], {}
    for name in names:
        shard = snapshot / index["weight_map"][name]
        with safe_open(shard, framework="pt", device="cpu") as file:
            value = file.get_tensor(name).clone()
        tensors.append(value)
        metadata[name] = dict(
            path=str(shard.resolve()),
            shape=list(value.shape),
            dtype=str(value.dtype),
            tensor_sha256=tensor_sha(value),
        )
    readout = FrozenReadout(*tensors, config["text_config"]["rms_norm_eps"])
    return readout, dict(
        model=model,
        tensors=metadata,
        epsilon=readout.epsilon,
        config_sha256=digest(snapshot / "config.json"),
        index_sha256=digest(snapshot / "model.safetensors.index.json"),
        arithmetic="CPU_FP64_sideband_not_registered_FP16_GPU_inference",
        model_weights_modified=False,
    )


def task_audit(parent, index, readout, destination):
    import torch

    worker = parent / f"workers/{index}"
    record = json.loads((worker / "diagnostic.json").read_text())["result"]
    manifest = json.loads((worker / "manifest.json").read_text())
    outputs = torch.load(worker / "prefix-outputs.pt", map_location="cpu", weights_only=True)
    predictions = torch.load(
        worker / "frozen-query-predictions.pt", map_location="cpu", weights_only=True
    )
    logits = torch.load(worker / "logits.pt", map_location="cpu", weights_only=True)
    full_h = torch.stack([outputs["fullkv"][s][-1]["layer_output"] for s in range(1, 33)])
    anchor_h = torch.stack([outputs["anchor"][s][-1]["layer_output"] for s in range(1, 33)])
    full_z, anchor_z = readout.logits(full_h), readout.logits(anchor_h)
    grad = readout.kl_gradient(full_z, anchor_h)
    sideband = dict(
        full_logits=full_z, anchor_logits=anchor_z, anchor_hidden_kl_gradient=grad, candidates={}
    )
    probe_rows, prospective_rows, replay_errors = [], [], []
    candidate_h, candidate_keys = [], []
    for proposal in record["replay"]:
        key = proposal["proposal_id"]
        for observed in proposal["steps"]:
            step = observed["step"]
            candidate_h.append(outputs[key][step][-1]["layer_output"])
            candidate_keys.append((key, step))
    candidate_h = torch.stack(candidate_h)
    candidate_z = readout.logits(candidate_h)
    candidate_lookup = {key: i for i, key in enumerate(candidate_keys)}
    for role, z in [("fullkv", full_z), ("anchor", anchor_z)]:
        for i in range(32):
            old = logits[role][i + 1].double().flatten()
            replay_errors.append(
                dict(
                    role=role,
                    prefix=i + 1,
                    maximum_logit_difference=float((z[i] - old).abs().max()),
                    greedy_token_agreement=bool(z[i].argmax() == old.argmax()),
                )
            )
    for proposal in record["replay"]:
        key, route = proposal["proposal_id"], proposal["proposal"]["layer"]
        prediction = predictions[key]
        delta = prediction["predicted_projected_change_fp64"]
        frozen_head_gain = squared_error_gain(
            prediction["fullkv_query_target"],
            prediction["predicted_anchor"],
            prediction["predicted_candidate"],
        ).mean(-1)
        full_residual = torch.stack(
            [outputs["fullkv"][s][route]["post_attention_residual"] for s in range(1, 33)]
        )
        anchor_residual = torch.stack(
            [outputs["anchor"][s][route]["post_attention_residual"] for s in range(1, 33)]
        )
        residual_gain = projected_residual_gain(full_residual, anchor_residual, delta)
        identity_linear, identity_quadratic = readout.directional_gains(full_z, anchor_h, delta)
        scores = dict(
            head_reconstruction=frozen_head_gain,
            projected_residual=residual_gain,
            identity_tail_readout_linear=identity_linear,
            identity_tail_readout_quadratic=identity_quadratic,
        )
        averages = {name: float(t.mean()) for name, t in scores.items()}
        for step in range(1, 33):
            prospective_rows.append(
                dict(
                    proposal_id=key,
                    prefix=step,
                    layer=route,
                    **{name: float(t[step - 1]) for name, t in scores.items()},
                )
            )
        for observed in proposal["steps"]:
            step = observed["step"]
            i = step - 1
            ci = candidate_lookup[key, step]
            ch, cz = candidate_h[ci], candidate_z[ci]
            actual_delta = ch.double() - anchor_h[i].double()
            linear, quadratic = readout.directional_gains(full_z[i], anchor_h[i], actual_delta)
            kl_a, kl_c = (float(kl_divergence(full_z[i], z)) for z in (anchor_z[i], cz))
            row = dict(
                proposal_id=key,
                prefix=step,
                layer=route,
                evidence_horizon=32,
                registered_kl_before=observed["anchor"]["logit_kl"],
                registered_kl_gain=observed["anchor"]["logit_kl"]
                - observed["candidate"]["logit_kl"],
                registered_maximum_logit_gain=observed["anchor"]["maximum_logit_difference"]
                - observed["candidate"]["maximum_logit_difference"],
                registered_token_agreement=observed["candidate"]["greedy_token_agreement"],
                **averages,
                actual_final_hidden=float(squared_error_gain(full_h[i], anchor_h[i], ch)),
                actual_normalized_hidden=float(
                    squared_error_gain(
                        readout.normalize(full_h[i]),
                        readout.normalize(anchor_h[i]),
                        readout.normalize(ch),
                    )
                ),
                actual_readout_linear=float(linear),
                actual_readout_quadratic=float(quadratic),
                actual_cpu_readout_kl=kl_a - kl_c,
                cpu_readout_kl_before=kl_a,
                cpu_readout_kl_after=kl_c,
            )
            probe_rows.append(row)
            old = logits[key][step].double().flatten()
            replay_errors.append(
                dict(
                    role=key,
                    prefix=step,
                    maximum_logit_difference=float((cz - old).abs().max()),
                    greedy_token_agreement=bool(cz.argmax() == old.argmax()),
                )
            )
        sideband["candidates"][key] = dict(
            prefixes=[r["step"] for r in proposal["steps"]],
            normalized_states=readout.normalize(
                torch.stack(
                    [outputs[key][r["step"]][-1]["layer_output"] for r in proposal["steps"]]
                )
            ),
            logits=candidate_z[[candidate_lookup[key, r["step"]] for r in proposal["steps"]]],
        )
    first = [r for r in probe_rows if r["prefix"] == 1]
    screening = audit_ranking(first, ReadoutAuditPolicy())
    explanatory = {
        name: rank_correlation([r[name] for r in first], [r["registered_kl_gain"] for r in first])
        for name, role in SCORE_PROVENANCE.items()
        if role.startswith("post_probe")
    }
    torch.save(sideband, destination / f"readout-sideband-{index}.pt")
    return (
        dict(
            task=manifest["dataset"]["id"],
            worker=index,
            proposals=len(record["replay"]),
            observed_proposal_prefixes=len(probe_rows),
            screening=screening,
            post_probe_explanatory_rank_correlations=explanatory,
            registered_accepted_proposals=record["accepted_proposals"],
            sideband_vs_GPU_logit_maximum_difference=max(
                r["maximum_logit_difference"] for r in replay_errors
            ),
            sideband_vs_GPU_greedy_matches=sum(r["greedy_token_agreement"] for r in replay_errors),
            sideband_vs_GPU_logit_observations=len(replay_errors),
            no_new_forward_pass=True,
            benchmark_labels_accessed=False,
        ),
        probe_rows,
        prospective_rows,
        replay_errors,
    )


def worker(root, plan_sha):
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU audit requires CUDA_VISIBLE_DEVICES empty before Python starts")
    torch.set_num_threads(4)
    plan = load_plan(root, plan_sha)
    parent = Path(plan["parent_root"])
    destination = root / "audit"
    if destination.exists():
        raise FileExistsError("readout audit outputs are write-once")
    destination.mkdir()
    tasks, all_rows, prospective, replay_checks = [], [], [], []
    try:
        models = [
            json.loads((parent / f"workers/{i}/manifest.json").read_text())["model"]
            for i in range(2)
        ]
        if models[0] != models[1]:
            raise ValueError("both consumed inputs must use identical pinned model/precision")
        readout, provenance = load_readout(models[0])
        for i in range(2):
            result, rows, predictions, checks = task_audit(parent, i, readout, destination)
            tasks.append(result)
            for name, target in (
                (rows, all_rows),
                (predictions, prospective),
                (checks, replay_checks),
            ):
                target.extend(dict(worker=i, **r) for r in name)
        for name, rows in [
            ("observed-prefixes.csv", all_rows),
            ("pre-probe-scores.csv", prospective),
            ("readout-reconstruction.csv", replay_checks),
        ]:
            with (destination / name).open("x", newline="") as stream:
                out = csv.DictWriter(stream, fieldnames=list(rows[0]))
                out.writeheader()
                out.writerows(rows)
        useful = [
            name
            for name in PROSPECTIVE_SCORES
            if name != "head_reconstruction"
            and all(t["screening"][name]["descriptive_signal"] for t in tasks)
        ]
        report = dict(
            status="completed",
            classification="consumed_input_CPU_readout_audit_not_quality_evidence",
            source_sha=plan["source_sha"],
            plan_sha256=plan_sha,
            tasks=tasks,
            readout_provenance=provenance,
            score_provenance=SCORE_PROVENANCE,
            useful_prospective_signals=useful,
            descriptive_screen_only=True,
            original_guard_unchanged=True,
            new_GPU_work=False,
            new_identities=0,
            model_weights_modified=False,
            **dict.fromkeys(DOWNSTREAM, False),
            limitations=[
                "Two consumed inputs; ranking correlations are not statistical validation.",
                "Identity-tail scores omit the actual downstream Jacobian.",
                "CPU FP64 reconstruction is sideband, not a replacement "
                "for the FP16 GPU forward/FP32 guard.",
                "All frozen V8 proposals were rejected; "
                "no successful 32-token substitution is inferred.",
            ],
        )
        write_once(destination / "report.json", report)
        write_once(
            destination / "artifact-hashes.json",
            {str(p): digest(p) for p in destination.iterdir() if p.is_file()},
        )
        write_once(
            root / "worker-completed.json",
            dict(
                status="completed",
                report_sha256=digest(destination / "report.json"),
                artifacts_sha256=digest(destination / "artifact-hashes.json"),
                source_sha=plan["source_sha"],
            ),
        )
        print(json.dumps(dict(status="completed", useful_prospective_signals=useful, tasks=tasks)))
    except Exception as error:
        write_once(
            root / "worker-failed.json",
            dict(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                **dict.fromkeys(DOWNSTREAM, False),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.protocol.resolve())
        print(digest(args.root / "plan.json"))
    else:
        worker(args.root.resolve(), args.plan_sha)
