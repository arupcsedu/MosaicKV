"""CPU-only audit of the unchanged pinned-compatible MileBench answer parser."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evicastkv_transport_scores import gated_rows, validate_rows

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.milebench import (
    MileBenchScoringRoute,
    milebench_multichoice_accuracy,
    route_milebench_dataset,
    score_milebench_sample,
)

SOURCE = Path(__file__).resolve().parents[2]


def choice_audit(prediction, reference, choices):
    """Observe the actual scorer's return locals, without duplicating its rules.

    The released parser's empty-answer fallback to C is retained and labeled,
    not silently replaced with a generic parser or excluded from denominators.
    This hook is CPU diagnostic-only; it is never installed during generation.
    """
    if sys.getprofile() is not None:
        raise RuntimeError("parser diagnostic must not replace another profiler")
    captured = {}

    def returned(frame, event, value):
        if event == "return" and frame.f_code is milebench_multichoice_accuracy.__code__:
            captured.update(frame.f_locals)

    try:
        sys.setprofile(returned)
        score = milebench_multichoice_accuracy(prediction, reference, choices)
    finally:
        sys.setprofile(None)
    labels = captured["labels"]
    empty = captured["candidate"] == ""
    selected = "C" if empty else captured["selected"]
    return dict(
        metric_value=score,
        option_count=len(choices),
        parsed_choice=selected,
        valid_single_choice=selected in labels,
        parse_status="upstream_empty_answer_default_C"
        if empty
        else "valid_choice"
        if selected in labels
        else "unparseable_choice",
        empty_answer=not prediction.strip(),
    )


def audit(scoring, output):
    import pandas as pd

    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("parser-audit output must be new")
    reg_path, gate_path = scoring / "registration.json", scoring / "GATE_scoring.json"
    reg, gate = (json.loads(p.read_text()) for p in (reg_path, gate_path))
    if (
        gate["registration_sha256"] != sha(reg_path)
        or gate["completed"] != 128
        or gate["failed"]
        or not gate["official_scorer_fixtures_passed"]
        or gate["timing_inference_allowed"]
    ):
        raise ValueError("complete pinned-scorer-validated source evidence required")
    for path, expected in reg["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("registered scoring evidence changed")
    rows, bindings = gated_rows([Path(p) for p in reg["prediction_roots"]])
    samples = json.loads(Path(reg["input_ledger"]).read_text())["samples"]
    validate_rows(rows, samples)
    observations, grouped = [], defaultdict(list)
    for row in rows:
        route = route_milebench_dataset(row["subtask"], row["question_type"])
        score = score_milebench_sample(
            route, row["answer"], row["references"][0], choices=tuple(row["choices"])
        )
        if route.route is MileBenchScoringRoute.MULTI_CHOICE:
            parsing = choice_audit(row["answer"], row["references"][0], row["choices"])
            if parsing["metric_value"] != score:
                raise ValueError("parser observation changed the actual metric")
        else:
            parsing = dict(
                metric_value=score,
                option_count=len(row["choices"]),
                parsed_choice=None,
                valid_single_choice=None,
                parse_status="official_open_ended_normalization",
                empty_answer=not row["answer"].strip(),
            )
        item = dict(
            sample_id=row["sample_id"],
            task=row["subtask"],
            method=row["method"],
            budget_bytes=row["budget_bytes"],
            metric_name=route.metric.value,
            source_run_id=row["run_id"],
            raw_answer=row["answer"],
            **parsing,
            scored_token_count=row["quality_decoding"]["scored_token_count"],
            timed_token_count=row["quality_decoding"]["timed_token_count"],
            manifest_path=row["manifest_path"],
            source_sha=row["source_sha"],
        )
        observations.append(item)
        grouped[(row["subtask"], row["method"], row["budget_bytes"], route.metric.value)].append(
            item
        )
    summaries = []
    for key, values in sorted(grouped.items()):
        summaries.append(
            dict(
                task=key[0],
                method=key[1],
                budget_bytes=key[2],
                metric_name=key[3],
                attempted=len(values),
                scored=len(values),
                failed=0,
                option_count_histogram=dict(
                    sorted(Counter(v["option_count"] for v in values).items())
                ),
                parse_status_counts=dict(Counter(v["parse_status"] for v in values)),
                empty_answers=sum(v["empty_answer"] for v in values),
                metric_value=sum(v["metric_value"] for v in values) / len(values),
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(observations).to_parquet(output / "parsing_audit.parquet", index=False)
    pd.DataFrame(observations).to_csv(output / "parsing_audit.csv", index=False)
    write_once(output / "subtask_parser_diagnostics.json", summaries)
    bindings.update({str(p): sha(p) for p in (reg_path, gate_path)})
    write_once(
        output / "GATE_parser_audit.json",
        dict(
            status="passed_descriptive_consumed_parser_audit",
            source_sha=revision,
            audited=128,
            attempted=128,
            excluded_for_parsing=0,
            parser_unchanged=True,
            no_four_choice_assumption=True,
            metric_families_kept_separate=True,
            chance_performance_not_inferred=True,
            new_GPU_inference=0,
            fresh_confirmation=False,
            superiority_established=False,
            audit_script_sha256=sha(Path(__file__)),
            bindings=bindings,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.scoring, args.output)
