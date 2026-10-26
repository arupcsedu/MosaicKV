#!/usr/bin/env python3
"""CPU-only, write-once V8.1 mechanistic analysis; not benchmark evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bounded_history_pilot import digest, write_once


def comparison(reference, anchor, candidate):
    """Observed vector errors; no fitted weights, thresholds, or promotion."""
    import torch

    if reference.shape != anchor.shape or reference.shape != candidate.shape:
        raise ValueError("mismatched measured vector geometry")
    if not all(bool(torch.isfinite(t).all()) for t in (reference, anchor, candidate)):
        raise ValueError("nonfinite measured vector")
    before, after = anchor.double() - reference.double(), candidate.double() - reference.double()
    b, a = float(before.square().mean()), float(after.square().mean())
    return dict(
        error_before=b,
        error_after=a,
        improvement=b - a,
        observed_change_rms=float((candidate.double() - anchor.double()).square().mean().sqrt()),
    )


def cosine(left, right):
    left, right = left.double().flatten(), right.double().flatten()
    norm = left.norm() * right.norm()
    return None if float(norm) == 0 else float((left @ right) / norm)


def analyze(root: Path, destination: Path):
    import torch

    torch.set_num_threads(1)
    if destination.exists():
        raise FileExistsError("derived analysis output is write-once")
    plan = json.loads((root / "plan.json").read_text())
    if not plan["protocol"].get("output_replay_from_v8") or any(
        plan[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    ):
        raise ValueError("not a scope-bound V8.1 diagnostic")
    bound = {
        str(root / "plan.json"): digest(root / "plan.json"),
        str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
    }
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError("frozen replay provenance changed")
        bound[path] = sha
    prefixes, layers, tasks = [], [], []
    for job in plan["jobs"]:
        worker = root / f"workers/{job['index']}"
        hashes = json.loads((worker / "artifact-hashes.json").read_text())
        for path, sha in hashes.items():
            if digest(path) != sha:
                raise ValueError("output replay artifact changed")
            bound[path] = sha
        completion = json.loads((worker / "completed.json").read_text())
        manifest = json.loads(Path(completion["manifest"]).read_text())
        if digest(completion["manifest"]) != completion["manifest_sha256"]:
            raise ValueError("output replay manifest changed")
        if (
            manifest["artifacts"]["raw_output_sha"] != digest(worker / "diagnostic.json")
            or manifest["artifacts"]["log_sha"] != digest(worker / "artifact-hashes.json")
            or manifest["execution"]["plan_sha256"] != digest(root / "plan.json")
        ):
            raise ValueError("manifest does not bind the replay outputs and plan")
        if (
            manifest["source"]["git_sha"] != plan["source_sha"]
            or manifest["source"]["git_dirty"]
            or manifest["source"]["config_sha"] != job["config_sha"]
        ):
            raise ValueError("output replay source revision changed")
        bound[completion["manifest"]] = completion["manifest_sha256"]
        bound[str(worker / "artifact-hashes.json")] = digest(worker / "artifact-hashes.json")
        bound[str(worker / "completed.json")] = digest(worker / "completed.json")
        saved = json.loads((worker / "diagnostic.json").read_text())
        if saved["status"] != "completed":
            raise ValueError("output replay worker failed")
        record = saved["result"]
        if not all(
            record[k]
            for k in (
                "instrumentation_passed",
                "instrumented_logits_bitwise_replayed",
                "intermediate_null_controls_bitwise_equal",
                "measured_source_cache_unchanged",
                "logical_positions_unchanged",
            )
        ):
            raise ValueError("instrumentation/isolation gate failed")
        if (
            record["padding_bytes"]
            or record["live_private_states_after_diagnostic"]
            or record["active_cap_bytes"] != plan["protocol"]["active_cap_bytes"]
        ):
            raise ValueError("storage/isolation boundary changed")
        outputs = torch.load(worker / "prefix-outputs.pt", map_location="cpu", weights_only=True)
        predictions = torch.load(
            worker / "frozen-query-predictions.pt", map_location="cpu", weights_only=True
        )
        last_rows = []
        for proposal in record["replay"]:
            key, descriptor = proposal["proposal_id"], proposal["proposal"]
            prediction = predictions[key]
            index, head = descriptor["layer"], descriptor["head"]
            groups = prediction["query_heads_per_kv_head"]
            chosen = slice(head * groups, (head + 1) * groups)
            for observed in proposal["steps"]:
                step = observed["step"]
                full, anchor, candidate = (
                    outputs[role][step] for role in ("fullkv", "anchor", key)
                )
                row = dict(
                    task=job["task"],
                    proposal_id=key,
                    prefix=step,
                    layer=index,
                    kv_head=head,
                    registered_kl_before=observed["anchor"]["logit_kl"],
                    registered_kl_after=observed["candidate"]["logit_kl"],
                    shadow_kl_before=observed["anchor"]["logit_kl_fp64"],
                    shadow_kl_after=observed["candidate"]["logit_kl_fp64"],
                )
                route = [
                    state[index]["attention_heads"][chosen] for state in (full, anchor, candidate)
                ]
                pairs = {
                    "predicted_head": [
                        prediction[k][step - 1]
                        for k in ("fullkv_query_target", "predicted_anchor", "predicted_candidate")
                    ],
                    "observed_head": route,
                    "observed_projected_head": [
                        x.double().flatten() @ prediction["changed_head_projection_columns_fp64"].T
                        for x in route
                    ],
                    "observed_final_layer_output": [
                        state[-1]["layer_output"] for state in (full, anchor, candidate)
                    ],
                }
                pairs.update(
                    {
                        f"observed_{field}": [
                            state[index][field] for state in (full, anchor, candidate)
                        ]
                        for field in record["captured_fields"]
                        if field != "attention_heads"
                    }
                )
                for name, tensors in pairs.items():
                    row.update({f"{name}_{k}": v for k, v in comparison(*tensors).items()})
                row["frozen_target_vs_observed_fullkv_mse"] = float(
                    (prediction["fullkv_query_target"][step - 1].double() - route[0].double())
                    .square()
                    .mean()
                )
                row["predicted_vs_observed_head_change_cosine"] = cosine(
                    prediction["predicted_candidate"][step - 1]
                    - prediction["predicted_anchor"][step - 1],
                    route[2].double() - route[1].double(),
                )
                row["predicted_vs_observed_projected_change_cosine"] = cosine(
                    prediction["predicted_projected_change_fp64"][step - 1],
                    candidate[index]["projected_attention"].double()
                    - anchor[index]["projected_attention"].double(),
                )
                row["route_input_identical_to_anchor"] = bool(
                    torch.equal(candidate[index]["residual_input"], anchor[index]["residual_input"])
                )
                prefixes.append(row)
                for layer in range(len(full)):
                    for field in record["captured_fields"]:
                        layers.append(
                            dict(
                                task=job["task"],
                                proposal_id=key,
                                prefix=step,
                                layer=layer,
                                field=field,
                                **comparison(
                                    full[layer][field],
                                    anchor[layer][field],
                                    candidate[layer][field],
                                ),
                            )
                        )
            last_rows.append(row)
        summary = dict(
            task=job["task"],
            proposals=len(last_rows),
            observed_prefixes=sum(len(r["steps"]) for r in record["replay"]),
            instrumentation_passed=True,
            active_bytes=record["active_cap_bytes"],
            padding_bytes=0,
            diagnostic_wall_seconds=record["diagnostic_wall_seconds"],
            output_capture_storage_bytes=record["output_capture_storage_bytes"],
        )
        for name in (
            "predicted_head",
            "observed_head",
            "observed_projected_head",
            "observed_projected_attention",
            "observed_post_attention_residual",
            "observed_layer_output",
            "observed_final_layer_output",
        ):
            summary[f"{name}_improved_at_rejection"] = sum(
                r[f"{name}_improvement"] > 0 for r in last_rows
            )
        summary["registered_kl_worsened_at_rejection"] = sum(
            r["registered_kl_after"] > r["registered_kl_before"] for r in last_rows
        )
        tasks.append(summary)
    destination.mkdir(parents=True)
    for name, rows in (("prefixes.csv", prefixes), ("layers.csv", layers)):
        with (destination / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(
        classification="retrospective_consumed_input_mechanistic_replay_not_quality_evidence",
        source_sha=plan["source_sha"],
        tasks=tasks,
        artifacts=bound,
        original_guard_unchanged=True,
        proposals_revised=False,
        new_identities=0,
        fresh_cohort_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        limitations=[
            "Two consumed inputs, not independent benchmark-quality evidence.",
            "Only original observed proposal prefixes were replayed; "
            "no 32-prefix acceptance is inferred.",
            "Vector differences describe associations, not causal interventions.",
            "FP64 sideband projections and KL do not replace the registered FP32 gate.",
            "Diagnostic hook/copy timing is not a production latency benchmark.",
        ],
    )
    write_once(destination / "report.json", report)
    lines = [
        "# V8.1 frozen-proposal attention/output diagnostic",
        "",
        "No proposals, thresholds, inputs, or downstream permissions changed.",
        "",
        "Counts below concern the last originally observed prefix of each rejected proposal.",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                f"## {task['task']}",
                "",
                f"{task['proposals']} frozen proposals; "
                f"{task['observed_prefixes']} observed prefixes.",
                "",
            ]
        )
        lines.extend(f"- {key}: {value}" for key, value in task.items() if "at_rejection" in key)
        lines.append("")
    lines.extend(
        [
            "These observations cannot authorize new cohorts or a paper claim.",
            "",
            *report["limitations"],
        ]
    )
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    write_once(
        destination / "artifact-hashes.json",
        {str(p): digest(p) for p in destination.iterdir() if p.is_file()},
    )
    print(json.dumps(dict(tasks=tasks, output=str(destination))))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve())
