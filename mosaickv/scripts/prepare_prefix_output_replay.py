"""Hash-bound V8.1 output replay; no pool construction or new identities."""

from __future__ import annotations

import json
from pathlib import Path

from bounded_history_pilot import SOURCE, digest, write_once

from mosaickv.config import DraftAllocationGuardConfig, config_sha256, load_config
from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy
from mosaickv.evaluation.prefix_output_replay import frozen_trajectories


def prepare_output_replay(root, source_sha, protocol_path, protocol):
    raw = protocol["diagnostic"].copy()
    raw["guard"] = DraftAllocationGuardConfig.from_mapping(raw["guard"])
    if (
        protocol["active_cap_bytes"] != 40 * 2**20
        or HeadDiagnosticPolicy(**raw) != HeadDiagnosticPolicy()
        or any(
            protocol[k]
            for k in (
                "fresh_cohort_authorized",
                "calibration_authorized",
                "paper_campaign_authorized",
            )
        )
    ):
        raise ValueError("output replay must preserve frozen policy and consumed-input scope")
    parent = Path(protocol["output_replay_from_v8"]).resolve()
    if digest(parent / "plan.json") != protocol["parent_plan_sha256"]:
        raise ValueError("V8 parent plan changed")
    plan = json.loads((parent / "plan.json").read_text())
    if plan["source_sha"] != protocol["parent_source_sha"] or len(plan["jobs"]) != 2:
        raise ValueError("output replay requires exactly the unchanged two consumed V8 inputs")
    bound = {
        str(protocol_path): digest(protocol_path),
        str(parent / "plan.json"): digest(parent / "plan.json"),
    }
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError("V8 input/configuration provenance changed")
        bound[path] = sha
    jobs = []
    for original in plan["jobs"]:
        i = original["index"]
        old = parent / f"workers/{i}"
        for name in ("diagnostic.json", "logits.pt", "draft-evidence.pt"):
            path = old / name
            sha = protocol["parent_artifacts"][str(i)][name]
            if digest(path) != sha:
                raise ValueError("frozen V8 diagnostic/evidence/logits changed")
            bound[str(path)] = sha
        record = json.loads((old / "diagnostic.json").read_text())
        if record["status"] != "completed" or record["result"]["diagnostic_passed"]:
            raise ValueError("output replay requires the completed negative V8 result")
        frozen_trajectories(record["result"])
        cfg = load_config(original["config"])
        if config_sha256(cfg) != original["config_sha"] or cfg.cache.budget_value != 40 * 2**20:
            raise ValueError("output replay cannot change V8 cache/model configuration")
        payload = json.loads(Path(original["input"]).read_text())
        expected = {
            "task",
            "dataset_id",
            "dataset_revision",
            "split",
            "sample_id",
            "prompt",
            "media",
            "draft_conditioned",
        }
        if set(payload) != expected:
            raise ValueError("output replay input is not label-redacted")
        payload.pop("draft_conditioned")
        payload["output_replay"] = {
            name: str(old / name) for name in ("diagnostic.json", "logits.pt", "draft-evidence.pt")
        }
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
