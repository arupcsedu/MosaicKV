#!/usr/bin/env python3
"""Two consumed-input diagnostics only; no fresh cohort or quality campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml
from bounded_history_pilot import SOURCE, clean_source, digest, inputs, write_once

from mosaickv.config import DraftAllocationGuardConfig, canonical_config, config_sha256, load_config

PROTOCOL = SOURCE / "mosaickv/configs/experiments/bounded_head_substitution_v7.yaml"


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def redact(sample):
    """Only prompt/media reach the worker; annotation fields cannot be forwarded."""
    return dict(
        sample_id=sample["sample_id"],
        prompt=sample["prompt"],
        media=[{key: item[key] for key in ("kind", "path", "sha256")} for item in sample["media"]],
    )


def prepare(root, protocol_path=PROTOCOL):
    sha = clean_source()
    root = Path(root).resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("require a new immutable scratch diagnostic directory")
    protocol_path = Path(protocol_path).resolve()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("draft_conditioned_from_v7_1") is not None:
        from prepare_draft_head import prepare_draft

        return prepare_draft(root, sha, protocol_path, protocol)
    if protocol.get("followup_from_v7") is not None:
        from prepare_head_followup import prepare_followup

        return prepare_followup(root, sha, protocol_path, protocol)
    from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy

    raw = protocol["diagnostic"].copy()
    raw["guard"] = DraftAllocationGuardConfig.from_mapping(raw["guard"])
    if HeadDiagnosticPolicy(**raw) != HeadDiagnosticPolicy() or any(
        protocol[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    ):
        raise ValueError("V7 diagnostic scope/thresholds changed")
    parent = inputs(
        Path(protocol["parent_root"]), protocol["parent_plan_sha256"], check_clean=False
    )
    if parent["source_sha"] != protocol["parent_source_sha"]:
        raise ValueError("V6 producer revision changed")
    seal_path = SOURCE / protocol["negative_seal"]
    seal = json.loads(seal_path.read_text())
    if not seal["all_16_identities_consumed"]:
        raise ValueError("diagnostic requires the sealed consumed cohort")
    bound = {str(PROTOCOL): digest(PROTOCOL), str(seal_path): digest(seal_path)}
    jobs = []
    for task in protocol["tasks"]:
        originals = [
            j
            for j in parent["jobs"]
            if j["task"] == task and j["method"] == "mosaickv_exact" and j["cap_mib"] == 40
        ]
        if len(originals) != 1:
            raise ValueError("missing unique unchanged anchor")
        original = originals[0]
        ledger_path = Path(original["ledger"])
        if digest(ledger_path) != original["ledger_sha"]:
            raise ValueError("consumed parent ledger changed")
        ledger = json.loads(ledger_path.read_text())
        if not any(
            Path(item["path"]).resolve() == ledger_path.resolve()
            and item["sha256"] == original["ledger_sha"]
            for item in seal["consumed_identity_ledgers"]
        ):
            raise ValueError("diagnostic identity ledger is not sealed as consumed")
        sample = ledger["samples"][0]  # Fixed order; never a failure-identity choice.
        cfg = load_config(original["config"])
        if cfg.bounded_decode.layer_allocation != "uniform" or cfg.cache.budget_value != 40 * 2**20:
            raise ValueError("diagnostic cannot change anchor/cache controls")
        index = len(jobs)
        payload = dict(
            task=task,
            dataset_id=ledger["dataset_id"],
            dataset_revision=ledger["dataset_revision"],
            split=ledger["split"],
            **redact(sample),
        )
        input_path, config_path = root / f"input-{index}.json", root / f"config-{index}.json"
        write_once(input_path, payload)
        write_once(config_path, canonical_config(cfg))
        bound.update({str(input_path): digest(input_path), str(config_path): digest(config_path)})
        jobs.append(
            dict(
                index=index,
                task=task,
                input=str(input_path),
                config=str(config_path),
                config_sha=config_sha256(cfg),
            )
        )
    if len(jobs) != 2:
        raise ValueError("only two consumed diagnostic identities are permitted")
    plan = dict(
        source_sha=sha,
        source_root=str(SOURCE),
        protocol=protocol,
        bound_files=bound,
        jobs=jobs,
        fresh_cohort_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return plan


def validate(root, plan_sha, index):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("diagnostic plan hash changed")
    plan = json.loads((root / "plan.json").read_text())
    clean_source(plan["source_sha"])
    if type(index) is not int or not 0 <= index < 2 or len(plan["jobs"]) != 2:
        raise ValueError("invalid diagnostic task index")
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError("diagnostic input/configuration changed")
    job = plan["jobs"][index]
    cfg = load_config(job["config"])
    payload = json.loads(Path(job["input"]).read_text())
    expected_keys = {
        "task",
        "dataset_id",
        "dataset_revision",
        "split",
        "sample_id",
        "prompt",
        "media",
    }
    if plan["protocol"].get("followup_from_v7") is not None:
        expected_keys.add("followup")
    if plan["protocol"].get("draft_conditioned_from_v7_1") is not None:
        expected_keys.add("draft_conditioned")
    if set(payload) != expected_keys or config_sha256(cfg) != job["config_sha"]:
        raise ValueError("diagnostic input is not label-redacted or config changed")
    if any(
        plan[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    ):
        raise ValueError("downstream experiments are forbidden")
    return plan, job, cfg, payload


def worker(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.evaluation.head_substitution_diagnostic import HeadSubstitutionDiagnostic
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import MaterializedMedia, _media_payload
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.manifest import (
        ArtifactProvenance,
        EvaluationProvenance,
        InputProvenance,
        RunManifestWriter,
    )
    from mosaickv.types import MeasurementType

    plan, job, cfg, payload = validate(root, plan_sha, index)
    destination = root / "workers" / str(index)
    if destination.exists():
        raise FileExistsError("diagnostic worker output is write-once")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("one A100 and deterministic CUBLAS required")
    torch.set_num_threads(12)
    torch.manual_seed(cfg.execution.seed)
    torch.use_deterministic_algorithms(True)
    media = [
        MaterializedMedia(MediaKind(m["kind"]), Path(m["path"]), m["sha256"])
        for m in payload["media"]
    ]
    messages = build_multimodal_messages(
        payload["prompt"], tuple(MediaItem(m.kind, _media_payload(m)) for m in media)
    )
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    parity = {}
    prepared = adapter.prepare_inputs(messages)
    for name, check in (
        ("generate", compare_with_generate),
        ("reinjection", compare_cache_reinjection),
    ):
        result = check(adapter, prepared, max_new_tokens=16)
        parity[name] = asdict(result)
        if result.token_agreement != 1.0 or result.maximum_logit_difference > 0.001:
            raise RuntimeError("diagnostic FullKV parity failed")
    write_once(destination / "parity.json", parity)
    followup = plan["protocol"].get("followup_from_v7") is not None
    draft = plan["protocol"].get("draft_conditioned_from_v7_1") is not None
    if draft:
        from mosaickv.evaluation.draft_head_diagnostic import DraftConditionedHeadDiagnostic
        from mosaickv.selection.draft_head_substitution import DraftHeadPoolPolicy

        reference = torch.load(
            payload["draft_conditioned"]["reference_logits"], map_location="cpu", weights_only=True
        )
        runtime = DraftConditionedHeadDiagnostic(
            adapter,
            cfg,
            trace_directory=destination / "traces",
            anchor_positions_sha256=payload["draft_conditioned"]["anchor_positions_sha256"],
            reference_logits={role: reference[role] for role in ("fullkv", "anchor")},
            pool_policy=DraftHeadPoolPolicy(**plan["protocol"]["proposal_policy"]),
        )
    elif followup:
        from mosaickv.evaluation.head_substitution_followup import HeadSubstitutionFollowup

        runtime = HeadSubstitutionFollowup(
            adapter, cfg, trace_directory=destination / "traces", followup=payload["followup"]
        )
    else:
        runtime = HeadSubstitutionDiagnostic(adapter, cfg, trace_directory=destination / "traces")
    version = "8" if draft else "7.1" if followup else "7"
    run_id = f"head-v{version}-{index}"
    try:
        result = runtime.diagnose(
            EvaluationRequest(run_id, payload["sample_id"], job["task"], messages, {})
        )
        record = dict(status="completed", result=result)
    except Exception as error:
        record = dict(status="failed", error=f"{type(error).__name__}: {error}")
    write_once(destination / "diagnostic.json", record)
    if followup or draft:
        torch.save(runtime.saved_logits, destination / "logits.pt")
    if draft:
        torch.save(runtime.saved_evidence, destination / "draft-evidence.pt")
    artifacts = {str(p): digest(p) for p in destination.rglob("*") if p.is_file()}
    write_once(destination / "artifact-hashes.json", artifacts)
    preprocessing = dict(
        model=cfg.model.id,
        revision=cfg.model.revision,
        processor=type(adapter.processor).__name__,
        config=getattr(adapter.processor, "to_dict", lambda: {})(),
    )
    manifest = RunManifestWriter(SOURCE).write(
        destination / "manifest.json",
        cfg,
        MeasurementType.VALIDATION_SMOKE,
        InputProvenance(
            object_sha(payload["prompt"]),
            object_sha(payload["media"]),
            object_sha(preprocessing),
            object_sha(prepared.model_inputs["input_ids"].cpu().tolist()),
        ),
        ArtifactProvenance(
            digest(destination / "diagnostic.json"),
            "not_applicable",
            digest(destination / "artifact-hashes.json"),
        ),
        run_id=run_id,
        execution_metadata=dict(
            phase="consumed_identity_draft_conditioned_head_diagnostic"
            if draft
            else "consumed_identity_numerical_controls_and_survivor_replay"
            if followup
            else "consumed_identity_diagnostic",
            plan_sha256=plan_sha,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            diagnostic_policy=plan["protocol"]["diagnostic"],
            **(dict(proposal_policy=plan["protocol"]["proposal_policy"]) if draft else {}),
            benchmark_labels_accessed=False,
            fresh_cohort_authorized=False,
        ),
        evaluation_provenance=EvaluationProvenance(
            evaluator_name="one_head_actual_draft_logit_diagnostic",
            evaluator_version=version,
            evaluator_commit=plan["source_sha"],
            reference_output_sha256="not_used",
            baseline_name="mosaickv_exact_anchor",
            baseline_commit=plan["source_sha"],
        ),
    )
    write_once(
        destination / "completed.json",
        dict(
            status=record["status"],
            manifest=str(manifest),
            manifest_sha256=digest(manifest),
            accepted=any(e["accepted"] for e in record.get("result", {}).get("extensions", []))
            if followup
            else record.get("result", {}).get("selection", {}).get("accepted", False),
            meaningful=any(e["meaningful"] for e in record.get("result", {}).get("extensions", []))
            if followup
            else record.get("result", {}).get("selection", {}).get("meaningful", False),
            **(
                dict(controls_passed=record.get("result", {}).get("controls_passed", False))
                if followup or draft
                else {}
            ),
            **(
                dict(diagnostic_passed=record.get("result", {}).get("diagnostic_passed", False))
                if draft
                else {}
            ),
            fresh_cohort_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    if record["status"] != "completed":
        raise RuntimeError("diagnostic failed; structured error and manifest preserved")


def submit(root, protocol_path=PROTOCOL):
    root = root.resolve()
    prepare(root, protocol_path)
    plan_sha = digest(root / "plan.json")
    for index in range(2):
        validate(root, plan_sha, index)
    logs = root / "logs"
    logs.mkdir()

    def sbatch(args):
        return (
            subprocess.check_output(
                ["sbatch", "--parsable", "--account=bii_dsc_community", *args], text=True
            )
            .strip()
            .split(";")[0]
        )

    array = sbatch(
        [
            "--hold",
            "--array=0-1%2",
            f"--output={logs}/gpu-%A_%a.out",
            f"--error={logs}/gpu-%A_%a.err",
            str(SOURCE / "mosaickv/slurm/head_substitution_v7.sbatch"),
            str(SOURCE),
            str(root),
            plan_sha,
        ]
    )
    snapshot = root / "scheduler"
    snapshot.mkdir()
    for name in ("control_a100_schedule.py", "cache_env.sh"):
        shutil.copy2(SOURCE / "mosaickv/scripts" / name, snapshot / name)
    write_once(
        snapshot / "a100_schedule.json",
        dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            window="all_times",
            arrays=[array],
            adopt_existing_holds=[array],
            source_sha=clean_source(),
            authorization="two_consumed_diagnostics_only_no_fresh_cohort",
        ),
    )
    write_once(
        snapshot / "snapshot.json",
        dict(
            source_sha=clean_source(),
            files={p.name: digest(p) for p in snapshot.iterdir() if p.is_file()},
        ),
    )
    controller = sbatch(
        [
            "--time=03:00:00",
            "--export=ALL,MOSAICKV_POLICY_ROOT=" + str(snapshot),
            f"--output={logs}/controller-%j.out",
            f"--error={logs}/controller-%j.err",
            str(SOURCE / "mosaickv/slurm/a100_schedule_controller.sbatch"),
        ]
    )
    write_once(
        root / "submission.json",
        dict(
            array=array,
            controller=controller,
            plan_sha256=plan_sha,
            maximum_a100s=6,
            new_identities=0,
            fresh_cohort_authorized=False,
        ),
    )
    print(json.dumps(dict(array=array, controller=controller, root=str(root))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "submit", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        globals()[args.mode](args.root, args.protocol)
