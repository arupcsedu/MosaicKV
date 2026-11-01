#!/usr/bin/env python3
"""Write-once CPU screen of consumed head/token intervention interactions."""

import argparse
import json
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.logit_composition_audit import screen

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/scratch/djy8hg/runs/evicastkv")
HEADS = ROOT / "causal-head-routing-consumed-20261031-v1"
TOKENS = ROOT / "coherent-token-consumed-20261031-v1"


def status_header(path):
    """Read only sorted decision metadata; stop before any scored sample rows."""
    prefix = []
    with path.open() as stream:
        for line in stream:
            if line.startswith(('  "rows":', '  "paired_diagnostic_gates":')):
                break
            prefix.append(line)
        else:
            raise ValueError("expected sorted decision header before scored rows")
    return json.loads("".join(prefix).rstrip().rstrip(",") + "\n}")


def main(output):
    if output.exists():
        raise ValueError("new write-once analysis root required")
    source = clean_source(SOURCE)
    bindings = {}
    for root in (HEADS, TOKENS):
        decision = root / "analysis/decision.json"
        status = status_header(decision)
        if not status["complete"] or status["failures"] or status["model_level_diagnostic_pass"]:
            raise ValueError("complete negative consumed diagnostic required")
        bindings[str(decision)] = sha(decision)
        for i in range(2):
            folder = root / f"workers/{i}"
            record = json.loads((folder / "result.json").read_text())
            if record["status"] != "completed" or record["failures"]:
                raise ValueError("producer incomplete")
            bindings[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer artifact changed")
                bindings[str(folder / name)] = digest
    results = []
    for i in range(2):
        a = torch.load(
            HEADS / f"workers/{i}/causal-probe-logits.pt", map_location="cpu", weights_only=True
        )
        b = torch.load(
            TOKENS / f"workers/{i}/coherent-probe-logits.pt", map_location="cpu", weights_only=True
        )
        if not torch.equal(a["source"], b["source"]) or not torch.equal(
            a["reference"], b["reference"]
        ):
            raise ValueError("different anchor/reference/conditioning")
        rows = screen(a["reference"], a["source"][1:], a["proposal_outputs"], b["proposal_outputs"])
        passing = [r for r in rows if r["statistics"]["guard_pass"]]
        results.append(
            dict(
                index=i,
                pairs=len(rows),
                predicted_passing_pairs=len(passing),
                rows=rows,
                bounded_shortlist=passing[:4],
            )
        )
    write_once(
        output / "decision.json",
        dict(
            source_sha=source,
            bound_artifacts=bindings,
            results=results,
            measurement_type="derived_numerical_diagnostic_not_task_quality",
            approximation="additive_actual_full_model_individual_logit_changes",
            nonlinear_joint_interactions_not_measured=True,
            independent_quality_N=0,
            superiority_established=False,
            GPU_execution_authorized=False,
            benchmark_labels_answers_and_task_scores_not_loaded=True,
            must_not_promote_without_actual_joint_model_verification=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
