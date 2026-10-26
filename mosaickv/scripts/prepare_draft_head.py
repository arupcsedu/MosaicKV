"""Prepare V8 using only the two already consumed, label-redacted V7 inputs."""

from __future__ import annotations

import json
from pathlib import Path

from bounded_history_pilot import SOURCE, digest, write_once

from mosaickv.config import DraftAllocationGuardConfig, config_sha256, load_config
from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy
from mosaickv.selection.draft_head_substitution import DraftHeadPoolPolicy


def prepare_draft(root, source_sha, protocol_path, protocol):
    raw = protocol["diagnostic"].copy()
    raw["guard"] = DraftAllocationGuardConfig.from_mapping(raw["guard"])
    if (
        HeadDiagnosticPolicy(**raw) != HeadDiagnosticPolicy()
        or DraftHeadPoolPolicy(**protocol["proposal_policy"]) != DraftHeadPoolPolicy()
    ):
        raise ValueError("V8 diagnostic must preserve the frozen bounded policy/guard")
    if any(
        protocol[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    ):
        raise ValueError("V8 permits no fresh or larger cohort")
    seal_path = SOURCE / protocol["negative_seal"]
    seal = json.loads(seal_path.read_text())
    sealed = {Path(path).resolve(): sha for path, sha in seal["artifacts"].items()}
    bound = {str(protocol_path): digest(protocol_path), str(seal_path): digest(seal_path)}
    for path, sha in seal["artifacts"].items():
        if digest(path) != sha:
            raise ValueError("sealed negative V7/V7.1 evidence changed")
        bound[path] = sha
    parent = Path(protocol["draft_conditioned_from_v7_1"]).resolve()
    if digest(parent / "plan.json") != protocol["parent_plan_sha256"]:
        raise ValueError("V7.1 parent plan changed")
    plan = json.loads((parent / "plan.json").read_text())
    if plan["source_sha"] != protocol["parent_source_sha"] or len(plan["jobs"]) != 2:
        raise ValueError("V8 requires exactly the unchanged two consumed inputs")
    jobs = []
    for original in plan["jobs"]:
        i = original["index"]
        old = parent / f"workers/{i}"
        record = json.loads((old / "diagnostic.json").read_text())
        if record["status"] != "completed" or not record["result"]["controls_passed"]:
            raise ValueError("V7.1 controls must pass before the separately versioned design")
        for path in (old / "diagnostic.json", old / "logits.pt"):
            if digest(path) != sealed[path.resolve()]:
                raise ValueError("V8 parent evidence is not sealed")
            bound[str(path)] = digest(path)
        for path in (original["input"], original["config"]):
            if digest(path) != plan["bound_files"][path]:
                raise ValueError("V8 parent input/config changed")
            bound[path] = digest(path)
        cfg = load_config(original["config"])
        if config_sha256(cfg) != original["config_sha"] or cfg.cache.budget_value != 40 * 2**20:
            raise ValueError("V8 cannot change model or cache/generation configuration")
        payload = json.loads(Path(original["input"]).read_text())
        if set(payload) != {
            "task",
            "dataset_id",
            "dataset_revision",
            "split",
            "sample_id",
            "prompt",
            "media",
            "followup",
        }:
            raise ValueError("V8 requires label-redacted consumed parent inputs")
        old_followup = payload.pop("followup")
        payload["draft_conditioned"] = dict(
            anchor_positions_sha256=old_followup["anchor_positions_sha256"],
            reference_logits=str(old / "logits.pt"),
        )
        destination = root / f"input-{i}.json"
        write_once(destination, payload)
        bound[str(destination)] = digest(destination)
        jobs.append(dict(original, input=str(destination)))
    result = dict(
        source_sha=source_sha,
        source_root=str(SOURCE),
        protocol=protocol,
        bound_files=bound,
        jobs=jobs,
        fresh_cohort_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "plan.json", result)
    return result
