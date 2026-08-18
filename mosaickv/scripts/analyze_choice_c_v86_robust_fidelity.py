#!/usr/bin/env python3
"""Apply the preregistered v86 fold-robust, label-blind freeze gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from analyze_choice_c_v74_label_blind_calibration import analyze as analyze_base
from choice_c_v74_protocol import token_agreement


def _normalize(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _token_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    if hasattr(value, "tolist"):
        converted = cast("Any", value).tolist()
        if isinstance(converted, list):
            return [int(item) for item in converted]
    raise TypeError(f"generated token IDs must be a sequence, got {type(value).__name__}")


def _fold_assignments(sample_ids: Sequence[str], *, fold_count: int, salt: str) -> dict[str, int]:
    """Assign label-blind, exactly balanced folds using a salted digest order."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if len(sample_ids) < fold_count or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique and cover every fold")
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest(),
            sample_id,
        ),
    )
    return {sample_id: rank % fold_count for rank, sample_id in enumerate(ordered)}


def _write_immutable_yaml(path: Path, payload: dict[str, Any]) -> None:
    encoded = yaml.safe_dump(payload, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite a different frozen candidate: {path}")
        return
    path.write_text(encoded, encoding="utf-8")
    os.chmod(path, 0o444)


def analyze(
    spec_path: Path,
    output_root: Path,
    frozen_output: Path,
    *,
    run_prefix: str = "v86",
) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    analysis_label = str(spec.get("analysis_label", "V86"))
    prior_boundary = str(spec.get("prior_result_exclusion_boundary", "v85"))
    if not analysis_label or not prior_boundary:
        raise ValueError("robust analysis label and prior-result boundary must be non-empty")
    protocol = cast("dict[str, Any]", spec.get("robust_fidelity_analysis", {}))
    if not protocol:
        raise ValueError(f"{analysis_label} requires robust_fidelity_analysis")
    if (
        protocol.get("labels_accessed") is not False
        or protocol.get("official_task_scores_accessed") is not False
        or protocol.get("fold_assignment") != "sha256_sorted_round_robin"
        or protocol.get("require_every_task_and_fold") is not True
        or protocol.get("require_base_memory_and_fidelity_gates") is not True
    ):
        raise ValueError(
            f"{analysis_label} robust fidelity protocol is not fail-closed and label-blind"
        )

    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{run_prefix}-base-freeze-",
        dir=analysis_root,
    ) as temporary:
        provisional_path = Path(temporary) / "base_gate_candidate.yaml"
        base = analyze_base(
            spec_path,
            output_root,
            provisional_path,
            run_prefix=run_prefix,
        )
        provisional = (
            cast(
                "dict[str, Any]",
                yaml.safe_load(provisional_path.read_text(encoding="utf-8")),
            )
            if provisional_path.exists()
            else None
        )

    trials_path = analysis_root / "label_blind_calibration_trials.parquet"
    trials = pd.read_parquet(trials_path)
    fold_count = int(protocol["fold_count"])
    expected_fold_size = int(protocol["expected_samples_per_fold"])
    salt = str(protocol["fold_assignment_salt"])
    answer_margin = float(protocol["fullkv_answer_noninferiority_margin_vs_prompt_topk"])
    token_margin = float(protocol["fullkv_token_noninferiority_margin_vs_prompt_topk"])
    if not 0 <= answer_margin <= 1 or not 0 <= token_margin <= 1:
        raise ValueError("robust fidelity margins must be in [0, 1]")

    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    candidate_ids = tuple(str(candidate["id"]) for candidate in candidates)
    base_ranking = {
        str(record["candidate_id"]): cast("dict[str, Any]", record)
        for record in cast("list[dict[str, Any]]", base["candidate_ranking"])
    }
    fold_rows: list[dict[str, Any]] = []
    for task_entry in cast("list[dict[str, Any]]", spec["tasks"]):
        task = str(task_entry["task"])
        task_frame = trials[trials["task"] == task]
        by_arm = {
            arm: task_frame[task_frame["arm"] == arm].set_index("sample_id").sort_index()
            for arm in ("full_kv", "prompt_attention_topk", *candidate_ids)
        }
        sample_ids = tuple(str(value) for value in by_arm["full_kv"].index)
        if any(
            tuple(str(value) for value in table.index) != sample_ids for table in by_arm.values()
        ):
            raise RuntimeError(f"{task} robust-fidelity arms are not sample-paired")
        assignments = _fold_assignments(
            sample_ids,
            fold_count=fold_count,
            salt=f"{salt}:{task}",
        )
        fold_sizes = {
            fold: sum(assignments[sample_id] == fold for sample_id in sample_ids)
            for fold in range(fold_count)
        }
        if set(fold_sizes.values()) != {expected_fold_size}:
            raise RuntimeError(
                f"{task} fold sizes {fold_sizes} differ from preregistered size "
                f"{expected_fold_size}"
            )
        full = by_arm["full_kv"]
        prompt = by_arm["prompt_attention_topk"]
        for candidate_id in candidate_ids:
            candidate = by_arm[candidate_id]
            for fold in range(fold_count):
                ids = tuple(sample_id for sample_id in sample_ids if assignments[sample_id] == fold)
                prompt_answer = statistics.fmean(
                    _normalize(prompt.at[sample_id, "answer"])
                    == _normalize(full.at[sample_id, "answer"])
                    for sample_id in ids
                )
                candidate_answer = statistics.fmean(
                    _normalize(candidate.at[sample_id, "answer"])
                    == _normalize(full.at[sample_id, "answer"])
                    for sample_id in ids
                )
                prompt_token = statistics.fmean(
                    token_agreement(
                        _token_ids(prompt.at[sample_id, "generated_token_ids"]),
                        _token_ids(full.at[sample_id, "generated_token_ids"]),
                    )
                    for sample_id in ids
                )
                candidate_token = statistics.fmean(
                    token_agreement(
                        _token_ids(candidate.at[sample_id, "generated_token_ids"]),
                        _token_ids(full.at[sample_id, "generated_token_ids"]),
                    )
                    for sample_id in ids
                )
                answer_delta = candidate_answer - prompt_answer
                token_delta = candidate_token - prompt_token
                fold_rows.append(
                    {
                        "task": task,
                        "candidate_id": candidate_id,
                        "fold": fold,
                        "sample_count": len(ids),
                        "candidate_fullkv_exact_answer_agreement": candidate_answer,
                        "prompt_topk_fullkv_exact_answer_agreement": prompt_answer,
                        "candidate_answer_delta_vs_prompt_topk": answer_delta,
                        "answer_gate_pass": answer_delta >= -answer_margin,
                        "candidate_fullkv_token_agreement": candidate_token,
                        "prompt_topk_fullkv_token_agreement": prompt_token,
                        "candidate_token_delta_vs_prompt_topk": token_delta,
                        "token_gate_pass": token_delta >= -token_margin,
                    }
                )

    fold_frame = pd.DataFrame(fold_rows).sort_values(["candidate_id", "task", "fold"])
    fold_frame.to_parquet(analysis_root / "robust_fidelity_folds.parquet", index=False)
    robust_ranking: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_folds = fold_frame[fold_frame["candidate_id"] == candidate_id]
        base_gate = bool(base_ranking[candidate_id]["gate_eligible"])
        answer_gate = bool(candidate_folds["answer_gate_pass"].all())
        token_gate = bool(candidate_folds["token_gate_pass"].all())
        robust_ranking.append(
            {
                "candidate_id": candidate_id,
                "base_gate_eligible": base_gate,
                "every_fold_answer_gate_eligible": answer_gate,
                "every_fold_token_gate_eligible": token_gate,
                "all_configured_gates_eligible": base_gate and answer_gate and token_gate,
                "worst_fold_fullkv_exact_answer_agreement": float(
                    candidate_folds["candidate_fullkv_exact_answer_agreement"].min()
                ),
                "worst_fold_answer_delta_vs_prompt_topk": float(
                    candidate_folds["candidate_answer_delta_vs_prompt_topk"].min()
                ),
                "worst_fold_fullkv_token_agreement": float(
                    candidate_folds["candidate_fullkv_token_agreement"].min()
                ),
                "worst_fold_token_delta_vs_prompt_topk": float(
                    candidate_folds["candidate_token_delta_vs_prompt_topk"].min()
                ),
                "macro_fullkv_token_agreement": float(
                    base_ranking[candidate_id]["macro_fullkv_token_agreement"]
                ),
                "worst_task_active_plus_residual_reduction": float(
                    base_ranking[candidate_id]["worst_task_active_plus_residual_reduction"]
                ),
            }
        )
    eligible = [row for row in robust_ranking if row["all_configured_gates_eligible"]]
    selected = (
        None
        if not eligible
        else sorted(
            eligible,
            key=lambda row: (
                -float(row["worst_fold_fullkv_exact_answer_agreement"]),
                -float(row["worst_fold_fullkv_token_agreement"]),
                -float(row["macro_fullkv_token_agreement"]),
                str(row["candidate_id"]),
            ),
        )[0]
    )
    selected_id = None if selected is None else str(selected["candidate_id"])
    result = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "canonical_eligible": True,
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "prior_result_exclusion_boundary": prior_boundary,
        "prior_results_used_only_as_exclusion_boundary": True,
        "robust_fidelity_protocol": protocol,
        "base_analysis_selected_candidate": base["selected_candidate"],
        "candidate_ranking": robust_ranking,
        "selected_candidate": selected_id,
        "all_configured_gates_eligible_candidate_found": selected_id is not None,
    }
    if prior_boundary == "v85":
        # Preserve the v86 artifact schema while newer protocols use the
        # generic boundary fields above.
        result["v85_used_only_as_exclusion_boundary"] = True
    result_path = analysis_root / "robust_fidelity_calibration.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(robust_ranking).to_parquet(
        analysis_root / "robust_fidelity_candidate_ranking.parquet", index=False
    )

    report = [
        f"# {analysis_label} label-blind robust-fidelity calibration",
        "",
        f"`{prior_boundary}` was used only as a sample-exclusion boundary. No prior-run "
        "answer, reference, score, generated token, telemetry value, or failure identity "
        "entered candidate selection or this analysis.",
        "",
        f"- Selected candidate: **{selected_id or 'NONE'}**",
        f"- Fold assignment: `{protocol['fold_assignment']}` with {fold_count} folds",
        f"- Per-fold answer noninferiority margin versus Prompt Top-K: {answer_margin:.2%}",
        f"- Per-fold token noninferiority margin versus Prompt Top-K: {token_margin:.2%}",
        "- All pre-existing task-level memory, fidelity, tier, residual, forecast, and "
        "risk gates remained mandatory.",
        "",
        "| Candidate | Base gates | Every-fold answer | Every-fold token | "
        "Worst answer agreement | Worst answer delta | Worst token agreement | "
        "Worst token delta | Worst total reduction |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in robust_ranking:
        report.append(
            f"| {row['candidate_id']} | {'yes' if row['base_gate_eligible'] else 'no'} | "
            f"{'yes' if row['every_fold_answer_gate_eligible'] else 'no'} | "
            f"{'yes' if row['every_fold_token_gate_eligible'] else 'no'} | "
            f"{float(row['worst_fold_fullkv_exact_answer_agreement']):.2%} | "
            f"{float(row['worst_fold_answer_delta_vs_prompt_topk']):+.2%} | "
            f"{float(row['worst_fold_fullkv_token_agreement']):.2%} | "
            f"{float(row['worst_fold_token_delta_vs_prompt_topk']):+.2%} | "
            f"{float(row['worst_task_active_plus_residual_reduction']):.2%} |"
        )
    (analysis_root / "robust_fidelity_calibration.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    if selected_id is not None:
        if provisional is None:
            raise RuntimeError("base gates selected no candidate but robust gates selected one")
        selected_spec = next(
            candidate for candidate in candidates if str(candidate["id"]) == selected_id
        )
        frozen = dict(provisional)
        frozen.update(
            {
                "classification": spec["frozen_candidate_classification"],
                "selected_candidate": selected_id,
                "candidate": selected_spec,
                "effective_method": "mosaickv_exact",
                "supports_prototype_or_repair_claim": False,
                "robust_fidelity_protocol": protocol,
                "robust_fidelity_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            }
        )
        _write_immutable_yaml(frozen_output, frozen)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frozen-output", required=True, type=Path)
    parser.add_argument("--run-prefix", default="v86")
    args = parser.parse_args()
    result = analyze(
        args.spec.resolve(),
        args.output_root.resolve(),
        args.frozen_output.resolve(),
        run_prefix=str(args.run_prefix),
    )
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "labels_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result["selected_candidate"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
