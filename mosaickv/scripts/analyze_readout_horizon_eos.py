#!/usr/bin/env python3
"""Retrospective, write-once first-EOS analysis of fixed consumed-input plans."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from mosaickv.adapters.huggingface.registry import _resolve_model_source
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.readout_horizon_audit import audit, eos_boundary
from mosaickv.evaluation.storage_component_gate import storage_match


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    current = clean_source(source)
    plan = read(args.root / "plan.json")
    if not plan.get("evaluation_only_fixed_plan_horizon_audit"):
        raise ValueError("registered consumed fixed-plan experiment required")
    if args.output.exists():
        raise FileExistsError("new immutable derived-analysis directory required")
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError("producer input/provenance changed")
    model = plan["parent"]["config"]["model"]
    snapshot = _resolve_model_source(model["id"], model["revision"], local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    eos = getattr(config, "eos_token_id", None)
    eos = eos if eos is not None else config.text_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results, artifacts = {}, {}
    for index in (0, 1):
        folder = args.root / f"workers/{index}"
        result = read(folder / "result.json")
        if (
            result["status"] != "completed"
            or result["errors"]
            or result["source_sha"] != plan["source_sha"]
        ):
            raise ValueError("complete successful immutable producers required")
        rows = {row["arm"]: row for p in folder.glob("[0-9][0-9]-*.json") if (row := read(p))}
        if len(rows) != len(plan["arms"]) or any(r["status"] != "completed" for r in rows.values()):
            raise ValueError("missing or failed fixed candidate must not disappear")
        values = {}
        for name, row in rows.items():
            path = Path(row["logits_path"])
            if sha(path) != row["logits_sha256"]:
                raise ValueError("saved prediction artifact changed")
            values[name] = torch.load(path, weights_only=True, map_location="cpu")
            artifacts[str(path)] = sha(path)
        reference = values["full_kv"]
        boundary = eos_boundary(reference, rows["full_kv"]["detail"]["answer"], eos)
        candidates = {
            "identity" if i == 0 else f"plan_{i}": values[f"controlled_plan_{i}"] for i in range(10)
        }
        if (plan["guard_predictions"], plan["guard_kl"], plan["guard_logit"]) != (32, 0.02, 1.0):
            raise ValueError("unchanged registered 32-token parent guard required")
        proof = audit(
            reference, candidates, semantic_predictions=boundary, guard_reference=values["parent"]
        )
        proof["guard_reference"] = "original_storage_parent_as_recorded_by_gpu_workers"
        for i in range(10):
            key = "identity" if i == 0 else f"plan_{i}"
            if (
                proof["candidate_diagnostics"][key]["registered_32_token_guard_passed"]
                != rows[f"controlled_plan_{i}"]["parent_guard_passed"]
            ):
                raise ValueError("recomputed parent guard disagrees with immutable GPU record")
        proof["full_128_step_audit_with_registered_parent_guard"] = audit(
            reference, candidates, guard_reference=values["parent"]
        )["horizons"]
        proof["storage_matching"] = storage_match([rows[f"controlled_plan_{i}"] for i in range(10)])
        if not proof["storage_matching"]["all_boundaries_matched"]:
            raise ValueError("counterfactual storage boundaries differ")
        results[str(index)] = proof
    consistent = []
    for horizon in ("1", "2", "3", "7"):
        entries = [v["horizons"][horizon] for v in results.values()]
        if all(
            e.get("meaningful")
            and e.get("all_128_tokens_equal")
            and e.get("later_kl_difference") is not None
            and e["later_kl_difference"] < 0
            for e in entries
        ):
            consistent.append(int(horizon))
    args.output.mkdir(parents=True)
    write_once(
        args.output / "decision.json",
        dict(
            producer_source_sha=plan["source_sha"],
            analysis_source_sha=current,
            plan_sha256=sha(args.root / "plan.json"),
            source_artifacts=artifacts,
            model=model,
            pinned_model_config_path=str(Path(snapshot) / "config.json"),
            pinned_model_config_sha256=sha(Path(snapshot) / "config.json"),
            pinned_eos_ids=sorted(eos),
            results=results,
            retrospectively_corrected_semantic_boundary=True,
            useful_future_draft_horizons_on_both_consumed_inputs=consistent,
            independent_quality_established=False,
            superiority_established=False,
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
        ),
    )
    lines = [
        "# Retrospective first-EOS fixed-plan audit",
        "",
        "Two consumed inputs; no independent superiority claim.",
        "",
        "| Input | EOS-inclusive predictions | Draft updates | Selected plan | "
        "Later-prefix KL difference vs identity | All 128 tokens equal |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for index, proof in results.items():
        for horizon, entry in proof["horizons"].items():
            lines.append(
                f"| {index} | {proof['semantic_prediction_limit']} | {horizon} | "
                f"{entry.get('selected', 'none')} | {entry.get('later_kl_difference')} | "
                f"{entry.get('all_128_tokens_equal')} |"
            )
    lines += [
        "",
        f"Consistently useful horizons on these consumed inputs only: {consistent}.",
        "",
        "Full 128-step numerical/storage checks remain separate. Post-EOS KL is not "
        "caption-quality evidence. No benchmark labels enter this audit.",
        "The unchanged registered guard is recomputed against the original storage parent "
        "and checked against every saved GPU flag. Additional FullKV absolute deviations "
        "remain in the JSON; they are not silently discarded or relaxed.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
