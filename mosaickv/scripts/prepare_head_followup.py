"""Hash-bound V7.1 preparation; copies only consumed, label-redacted inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from bounded_history_pilot import SOURCE, digest, write_once

from mosaickv.backends.hf_allocation_guard import draft_row_causes
from mosaickv.config import DraftAllocationGuardConfig, config_sha256, load_config
from mosaickv.selection.draft_shortlist import validate_shortlist_trace


def extract_survivors(result: dict[str, Any], expected_count: int) -> dict[str, Any]:
    selection = result["selection"]
    validate_shortlist_trace(selection)
    if selection["horizon"] != 32 or selection["short_horizon"] != 4:
        raise ValueError("follow-up must preserve the registered horizon")
    survivors = []
    for event in selection["events"]:
        if event["status"] != "not_in_actual_logit_shortlist":
            continue
        if len(event["steps"]) != 4 or any(
            draft_row_causes(row, DraftAllocationGuardConfig()) for row in event["steps"]
        ):
            raise ValueError("only unextended four-prefix survivors may be replayed")
        identity = event["proposal_id"]
        survivors.append(
            dict(
                proposal_id=identity,
                proposal=result["proposals"][identity],
                positions_sha256=result["candidate_positions_sha256"][identity],
                short_steps=event["steps"],
            )
        )
    if len(survivors) != expected_count:
        raise ValueError("frozen unextended survivor count changed")
    return dict(
        anchor_positions_sha256=result["anchor_positions_sha256"],
        survivors=copy.deepcopy(survivors),
    )


def prepare_followup(
    root: Path, source_sha: str, protocol_path: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy

    policy = protocol["diagnostic"].copy()
    policy["guard"] = DraftAllocationGuardConfig.from_mapping(policy["guard"])
    if HeadDiagnosticPolicy(**policy) != HeadDiagnosticPolicy() or any(
        protocol[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    ):
        raise ValueError("follow-up may not change the guard or authorize another cohort")
    parent = Path(protocol["followup_from_v7"]).resolve()
    if digest(parent / "plan.json") != protocol["parent_plan_sha256"]:
        raise ValueError("V7 parent plan changed")
    plan = json.loads((parent / "plan.json").read_text())
    if plan["source_sha"] != protocol["parent_source_sha"] or len(plan["jobs"]) != 2:
        raise ValueError("wrong V7 producer/scope")
    bound = {
        str(protocol_path): digest(protocol_path),
        str(parent / "plan.json"): digest(parent / "plan.json"),
    }
    jobs = []
    for original in plan["jobs"]:
        index = original["index"]
        path = parent / f"workers/{index}/diagnostic.json"
        if digest(path) != protocol["parent_diagnostic_sha256"][original["task"]]:
            raise ValueError("reviewed V7 diagnostic changed")
        bound[str(path)] = digest(path)
        payload = json.loads(Path(original["input"]).read_text())
        if set(payload) != {
            "task",
            "dataset_id",
            "dataset_revision",
            "split",
            "sample_id",
            "prompt",
            "media",
        }:
            raise ValueError("follow-up parent input is not label-redacted")
        for p in (original["input"], original["config"]):
            if digest(p) != plan["bound_files"][p]:
                raise ValueError("original V7 input/configuration changed")
            bound[p] = digest(p)
        cfg = load_config(original["config"])
        if config_sha256(cfg) != original["config_sha"] or cfg.cache.budget_value != 40 * 2**20:
            raise ValueError("follow-up cannot change actual cache budget/configuration")
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            raise ValueError("cannot replay an incomplete parent diagnostic")
        payload["followup"] = extract_survivors(
            record["result"], protocol["expected_survivors"][original["task"]]
        )
        destination = root / f"input-{index}.json"
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
