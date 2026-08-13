#!/usr/bin/env python3
"""Run the frozen two-stage Choice C calibration and held-out comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import (
    CacheConfig,
    FidelityDiagnosticsConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.experiment_matrix import _base_config
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation
from mosaickv.repair import RepairCalibrationObservation, calibrate_decision_repair
from mosaickv.selection import (
    CalibrationStabilityError,
    FidelityObservation,
    HeadFidelityObservation,
    derive_adaptive_head_routing,
    select_fidelity_profile,
)
from mosaickv.types import (
    BudgetUnit,
    RepairCandidateRanking,
    RepairPolicy,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_retention_overrides(
    spec_path: Path,
) -> tuple[dict[str, float], str | None, str | None]:
    """Load an optional, separately frozen label-blind byte-budget profile."""

    raw_path = os.environ.get("MOSAICKV_BASELINE_BUDGET_PROFILE")
    if raw_path is None:
        return {}, None, None
    path = Path(raw_path).resolve()
    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise ValueError("baseline budget profile must contain a mapping")
    if record.get("design_spec_sha256") != _sha(spec_path):
        raise RuntimeError("baseline budget profile does not match the frozen design spec")
    if record.get("labels_or_task_scores_used") is not False:
        raise RuntimeError("baseline budget calibration must be label-blind")
    values = record.get("retention_ratio_by_method")
    if not isinstance(values, dict):
        raise ValueError("baseline budget profile has no method ratio mapping")
    result = {str(key): float(value) for key, value in values.items()}
    if any(not 0 < value <= 1 for value in result.values()):
        raise ValueError("baseline budget retention ratios must lie in (0, 1]")
    return result, str(path), _sha(path)


def _git_state(repository: Path) -> dict[str, object]:
    sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_sha": sha, "git_dirty": bool(status.strip())}


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("sample_ids"), list):
        raise ValueError(f"invalid sample ledger: {path}")
    ids = cast("list[str]", value["sample_ids"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"sample ledger IDs must be non-empty and unique: {path}")
    return cast("dict[str, Any]", value), _sha(path)


def _diagnostics(
    ledger: dict[str, Any], ledger_sha: str, *, mosaic: bool
) -> FidelityDiagnosticsConfig:
    return FidelityDiagnosticsConfig(
        enabled=True,
        sample_ids=tuple(cast("list[str]", ledger["sample_ids"])),
        sample_ledger_sha=ledger_sha,
        forecast_oracle_steps=4 if mosaic else 0,
    )


def _mosaic_config(
    parameters: dict[str, Any],
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    retention_ratio: float,
    output_length: int,
    seed: int,
    method_name: str = "mosaickv_full",
    repair_policy: RepairPolicy | None = None,
    repair_evaluation_only: bool = False,
) -> Any:
    config = _base_config(
        model_key="llava_1_5_7b",
        backend_key="hf_eager",
        method_name=method_name,
        task_key="docvqa",
        retention_ratio=retention_ratio,
        output_length=output_length,
        seed=seed,
    )
    utility = cast("dict[str, Any]", parameters["utility"])
    selection = cast("dict[str, Any]", parameters["selection"])
    routing = dict(cast("dict[str, Any]", parameters["routing"]))
    draft_steps = int(routing.pop("draft_steps", 2))
    prompt_window = int(routing.pop("prompt_window", config.forecasting.prompt_window))
    graph = cast("dict[str, Any]", parameters.get("graph", {}))
    prototypes = cast("dict[str, Any]", parameters["prototypes"])
    repair = cast("dict[str, Any]", parameters["repair"])
    repair_values = dict(repair)
    configured_repair_policy = RepairPolicy(
        str(repair_values.pop("policy", RepairPolicy.DECISION_AWARE.value))
    )
    configured_evaluation_only = bool(repair_values.pop("evaluation_only", False))
    return replace(
        config,
        diagnostics=_diagnostics(ledger, ledger_sha, mosaic=True),
        forecasting=replace(
            config.forecasting,
            draft_steps=draft_steps,
            prompt_window=prompt_window,
            adaptive_head_routing=True,
            **routing,
        ),
        graph=replace(config.graph, **graph),
        utility=replace(config.utility, **utility),
        selection=replace(
            config.selection,
            stop_on_nonpositive_gain=False,
            **selection,
        ),
        prototypes=replace(config.prototypes, **prototypes),
        repair=replace(
            config.repair,
            policy=(configured_repair_policy if repair_policy is None else repair_policy),
            evaluation_only=(repair_evaluation_only or configured_evaluation_only),
            **repair_values,
        ),
    )


def _baseline_config(
    method: str,
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    retention_ratio: float,
    output_length: int,
    seed: int,
) -> Any:
    ratio = 1.0 if method == "full_kv" else retention_ratio
    config = _base_config(
        model_key="llava_1_5_7b",
        backend_key="hf_eager",
        method_name=method,
        task_key="docvqa",
        retention_ratio=ratio,
        output_length=output_length,
        seed=seed,
    )
    # Prompt Top-K shares MosaicKV's fixed blocks. LOOK-M and VL-Cache retain
    # native token granularity, but all methods quantize their materialized
    # dense cache against MosaicKV's common block-16 physical-byte lattice.
    block_size = 16 if method in {"full_kv", "prompt_attention_topk"} else 1
    return replace(
        config,
        diagnostics=_diagnostics(ledger, ledger_sha, mosaic=False),
        cache=CacheConfig(
            budget_value=config.cache.budget_value,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=ratio,
            block_size=block_size,
            accounting_spec_sha=config.cache.accounting_spec_sha,
            physical_budget_block_size=16,
            physical_budget_mandatory_policy="last_token",
        ),
    )


def _write_config(config: Any, path: Path) -> None:
    payload = yaml.safe_dump(json.loads(canonical_config_json(config)), sort_keys=True)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f"refusing to change an existing run config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _run(
    config: Any,
    adapter: Any,
    *,
    run_id: str,
    output_root: Path,
    ledger_path: Path,
    ledger_sha: str,
) -> pd.DataFrame:
    print(
        f"choice-c start run_id={run_id} method={config.method.value} "
        f"samples={len(config.diagnostics.sample_ids)}",
        flush=True,
    )
    config_path = output_root / "configs" / f"{run_id}-{config_sha256(config)}.yaml"
    _write_config(config, config_path)
    args = argparse.Namespace(
        task="docvqa",
        run_id=run_id,
        output_dir=str(output_root),
        raw_output=None,
        parquet_output=None,
        manifest=None,
        trace_directory=str(output_root / "traces"),
        subset_size=None,
        full_dataset=False,
        sample_offset=0,
        sample_id_file=str(ledger_path),
        sample_id_file_sha=ledger_sha,
        local_files_only=True,
    )
    status, summary = run_hf_evaluation(args, config, adapter=adapter)
    failed_samples = summary.get("failed_samples", 0)
    if not isinstance(failed_samples, int):
        raise RuntimeError(f"run returned invalid failure count: {run_id}")
    if status or failed_samples:
        raise RuntimeError(f"run failed or contains failed samples: {run_id}: {summary}")
    frame = pd.read_parquet(cast("str", summary["parquet_output"]))
    if len(frame) != len(config.diagnostics.sample_ids) or set(frame["status"]) != {"completed"}:
        raise RuntimeError(f"run is incomplete: {run_id}")
    print(
        f"choice-c complete run_id={run_id} rows={len(frame)}",
        flush=True,
    )
    return frame


def _forecast_regret_by_sample(output_root: Path, run_id: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        record = json.loads(path.read_text())
        diagnostic = record.get("forecast_quality_diagnostic", {})
        value = diagnostic.get("mean_forecast_regret")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values[str(record["sample_id"])] = float(value)
    return values


def _effective_method_by_sample(output_root: Path, run_id: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        record = json.loads(path.read_text())
        sample_id = record.get("sample_id")
        effective_method = record.get("effective_method")
        if isinstance(sample_id, str) and isinstance(effective_method, str):
            values[sample_id] = effective_method
    return values


def _head_errors_by_sample(
    output_root: Path, run_id: str
) -> dict[str, dict[tuple[int, int], float]]:
    values: dict[str, dict[tuple[int, int], float]] = {}
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        record = json.loads(path.read_text())
        sample_id = record.get("sample_id")
        rows = record.get("fidelity_diagnostic", {}).get("attention_output_error_by_kv_head")
        if not isinstance(sample_id, str) or not isinstance(rows, list):
            continue
        parsed: dict[tuple[int, int], float] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("per-head fidelity trace contains a malformed row")
            identity = int(row["layer"]), int(row["kv_head"])
            error = float(row["error"])
            if identity in parsed or not math.isfinite(error) or error < 0:
                raise RuntimeError("per-head fidelity trace is invalid")
            parsed[identity] = error
        values[sample_id] = parsed
    return values


def _prototype_dispersion_limits(
    output_root: Path,
    run_id: str,
    *,
    safety_factor: float,
) -> tuple[float, float]:
    """Derive fail-closed K/V dispersion ceilings from calibration traces only."""

    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("prototype dispersion safety_factor must be finite and >= 1")
    maximum_key = 0.0
    maximum_value = 0.0
    sample_count = 0
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        record = json.loads(path.read_text())
        rows = record.get("prototype_diagnostics_by_kv_head")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"prototype calibration trace has no dispersion diagnostics: {path}")
        sample_count += 1
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("prototype dispersion trace contains a malformed row")
            key = float(row["maximum_key_dispersion"])
            value = float(row["maximum_value_dispersion"])
            if any(not math.isfinite(item) or item < 0 for item in (key, value)):
                raise RuntimeError("prototype dispersion trace is invalid")
            maximum_key = max(maximum_key, key)
            maximum_value = max(maximum_value, value)
    if sample_count < 1 or maximum_key <= 0 or maximum_value <= 0:
        raise RuntimeError("prototype calibration produced no positive dispersion evidence")
    return maximum_key * safety_factor, maximum_value * safety_factor


def _repair_calibration_observations(
    output_root: Path, run_id: str
) -> tuple[RepairCalibrationObservation, ...]:
    observations: list[RepairCalibrationObservation] = []
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        record = json.loads(path.read_text())
        sample_id = record.get("sample_id")
        events = record.get("repair_events")
        if not isinstance(sample_id, str) or not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                raise RuntimeError("repair calibration trace contains a malformed event")
            if event.get("risk_normalization") != "sum_to_one":
                raise RuntimeError("repair calibration requires normalized prototype risk")
            observations.append(
                RepairCalibrationObservation(
                    sample_id=sample_id,
                    step=int(event["trigger_step"]),
                    normalized_prototype_risk=float(event["maximum_prototype_risk"]),
                    next_token_probability_margin=float(event["next_token_probability_margin"]),
                    oracle_beneficial=event.get("quality_recovered") is True,
                    draft_kl_divergence=(
                        None
                        if event.get("draft_kl_divergence") is None
                        else float(event["draft_kl_divergence"])
                    ),
                )
            )
    if not observations:
        raise RuntimeError("oracle repair calibration produced no provisional-step evidence")
    return tuple(observations)


def _fidelity_error(frame: pd.DataFrame) -> pd.Series:
    return (
        1.0
        - frame["token_agreement"].astype(float)
        + frame["logit_kl"].astype(float).map(math.log1p)
        + frame["attention_output_error"].astype(float)
    )


def _paired_bootstrap_interval(
    differences: tuple[float, ...], *, seed: int, resamples: int = 2000
) -> tuple[float, float]:
    if not differences:
        raise ValueError("paired bootstrap requires differences")
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    return means[math.floor(0.025 * (resamples - 1))], means[math.ceil(0.975 * (resamples - 1))]


def _analyze_heldout(
    rows: pd.DataFrame, spec: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "active_kv_bytes",
        "task_score",
        "token_agreement",
        "logit_kl",
        "attention_output_error",
    }
    if rows[list(required)].isna().any().any():
        raise RuntimeError("held-out comparison contains missing primary measurements")
    compressed = rows[rows["requested_method"] != "full_kv"].copy()
    compressed["fidelity_error"] = _fidelity_error(compressed)
    summary = (
        compressed.groupby("requested_method", sort=True)
        .agg(
            sample_count=("sample_id", "nunique"),
            task_score=("task_score", "mean"),
            active_kv_bytes=("active_kv_bytes", "mean"),
            fidelity_error=("fidelity_error", "mean"),
            end_to_end_time=("end_to_end_time", "mean"),
            compression_time=("compression_time", "mean"),
            repair_count=("repair_count", "sum"),
        )
        .reset_index()
    )
    mosaic = summary[summary["requested_method"] == "mosaickv_full"].iloc[0]
    baselines = summary[summary["requested_method"] != "mosaickv_full"].copy()
    baselines["relative_byte_mismatch"] = (
        baselines["active_kv_bytes"] - mosaic["active_kv_bytes"]
    ).abs() / baselines["active_kv_bytes"]
    maximum_mismatch = float(spec["gate"]["maximum_relative_byte_mismatch"])
    matched = baselines[baselines["relative_byte_mismatch"] <= maximum_mismatch]
    if matched.empty:
        nearest = baselines.sort_values(["relative_byte_mismatch", "requested_method"]).iloc[0]
        improvement = float("nan")
    else:
        nearest = matched.sort_values(["fidelity_error", "requested_method"]).iloc[0]
        denominator = float(nearest["fidelity_error"])
        improvement = (
            (denominator - float(mosaic["fidelity_error"])) / denominator
            if denominator > 0
            else (0.0 if float(mosaic["fidelity_error"]) == 0 else -math.inf)
        )
    required_improvement = float(
        spec["gate"]["required_relative_improvement_over_nearest_competitor"]
    )
    best_baseline_score = float(baselines["task_score"].max())
    best_score_method = str(
        baselines.sort_values(["task_score", "requested_method"], ascending=[False, True]).iloc[0][
            "requested_method"
        ]
    )
    mosaic_task_rows = compressed[compressed["requested_method"] == "mosaickv_full"][
        ["sample_id", "task_score"]
    ].rename(columns={"task_score": "mosaic_score"})
    comparator_task_rows = compressed[compressed["requested_method"] == best_score_method][
        ["sample_id", "task_score"]
    ].rename(columns={"task_score": "baseline_score"})
    paired_scores = mosaic_task_rows.merge(
        comparator_task_rows, on="sample_id", validate="one_to_one"
    )
    score_differences = tuple(
        float(row.mosaic_score) - float(row.baseline_score)
        for row in paired_scores.itertuples(index=False)
    )
    score_ci = _paired_bootstrap_interval(
        score_differences,
        seed=int(spec["seed"]),
    )
    noninferiority_margin = float(spec["gate"].get("task_score_noninferiority_margin", 0.0))
    mosaic_rows = compressed[compressed["requested_method"] == "mosaickv_full"]
    genuine_three_tier_rows = int(
        (
            (mosaic_rows["effective_method"] == "mosaickv_full")
            & (mosaic_rows["residual_kv_bytes"].astype(float) > 0)
        ).sum()
    )
    require_genuine_three_tier = bool(
        spec["gate"].get("require_genuine_three_tier_mosaickv", False)
    )
    genuine_three_tier_pass = bool(
        not require_genuine_three_tier or genuine_three_tier_rows == len(mosaic_rows)
    )
    gates = {
        "all_rows_complete": True,
        "nearest_competitor": str(nearest["requested_method"]),
        "relative_byte_mismatch": float(nearest["relative_byte_mismatch"]),
        "relative_fidelity_improvement": improvement,
        "required_relative_fidelity_improvement": required_improvement,
        "mosaickv_task_score": float(mosaic["task_score"]),
        "best_baseline_task_score": best_baseline_score,
        "best_score_baseline": best_score_method,
        "task_score_difference_ci95": list(score_ci),
        "task_score_noninferiority_margin": noninferiority_margin,
        "byte_match_pass": float(nearest["relative_byte_mismatch"]) <= maximum_mismatch,
        "fidelity_improvement_pass": bool(improvement >= required_improvement),
        "task_score_pass": score_ci[0] >= -noninferiority_margin,
        "genuine_three_tier_rows": genuine_three_tier_rows,
        "required_genuine_three_tier_rows": len(mosaic_rows),
        "genuine_three_tier_pass": genuine_three_tier_pass,
    }
    gates["full_sweep_allowed"] = bool(
        gates["all_rows_complete"]
        and gates["byte_match_pass"]
        and gates["fidelity_improvement_pass"]
        and gates["task_score_pass"]
        and gates["genuine_three_tier_pass"]
    )
    return summary, gates


def run(
    spec_path: Path,
    *,
    phase: str = "all",
    requested_variant: str | None = None,
) -> int:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    project_root = spec_path.resolve().parents[2]
    repository = project_root.parent
    output_root = Path(
        os.environ.get("MOSAICKV_OUTPUT_ROOT_OVERRIDE", str(spec["output_root"]))
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    calibration_path = project_root / str(spec["calibration_ledger"])
    evaluation_path = project_root / str(spec["evaluation_ledger"])
    calibration, calibration_sha = _load_ledger(calibration_path)
    evaluation, evaluation_sha = _load_ledger(evaluation_path)
    validation_path = (
        None
        if spec.get("calibration_validation_ledger") is None
        else project_root / str(spec["calibration_validation_ledger"])
    )
    validation: dict[str, Any] | None = None
    validation_sha: str | None = None
    if validation_path is not None:
        validation, validation_sha = _load_ledger(validation_path)
    if set(calibration["sample_ids"]) & set(evaluation["sample_ids"]):
        raise RuntimeError("calibration and held-out ledgers overlap")
    if validation is not None:
        validation_ids = set(validation["sample_ids"])
        if validation_ids & set(calibration["sample_ids"]):
            raise RuntimeError("design calibration and internal validation ledgers overlap")
        if validation_ids & set(evaluation["sample_ids"]):
            raise RuntimeError("internal validation and held-out ledgers overlap")
    worker_phases = {
        "calibration-candidate",
        "calibration-exact-control",
        "internal-method",
        "internal-exact-control",
        "heldout-method",
    }
    if phase not in {"all", *worker_phases}:
        raise ValueError(f"unsupported Choice C phase: {phase}")
    candidates = cast("dict[str, dict[str, Any]]", spec["choice_c_candidates"])
    if phase in {"calibration-candidate", "calibration-exact-control"} and (
        requested_variant not in candidates
    ):
        raise ValueError(
            f"calibration candidate {requested_variant!r} is unavailable; "
            f"known variants: {', '.join(sorted(candidates))}"
        )
    methods = cast("list[str]", spec["methods"])
    if phase in {"internal-method", "heldout-method"} and requested_variant not in methods:
        raise ValueError(
            f"comparison method {requested_variant!r} is unavailable; "
            f"known methods: {', '.join(methods)}"
        )
    provenance = {
        **_git_state(repository),
        "classification": spec["classification"],
        "spec_path": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "calibration_ledger_sha256": calibration_sha,
        "evaluation_ledger_sha256": evaluation_sha,
        "calibration_validation_ledger_sha256": validation_sha,
    }
    baseline_overrides, baseline_profile_path, baseline_profile_sha = _baseline_retention_overrides(
        spec_path
    )
    provenance.update(
        {
            "baseline_budget_profile_path": baseline_profile_path,
            "baseline_budget_profile_sha256": baseline_profile_sha,
        }
    )
    provenance_name = (
        "provenance.json" if phase == "all" else f"provenance-{phase}-{requested_variant}.json"
    )
    (output_root / provenance_name).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    retention_ratio = float(spec["retention_ratio"])
    for candidate_name, parameters in candidates.items():
        group_size = int(cast("dict[str, Any]", parameters["prototypes"])["group_size"])
        minimum_ratio = 2.0 / (group_size + 1)
        if minimum_ratio > retention_ratio:
            raise RuntimeError(
                f"prototype geometry for {candidate_name!r} has minimum active ratio "
                f"{minimum_ratio:.6f} above requested {retention_ratio:.6f}"
            )
    first = _mosaic_config(
        candidates[min(candidates)],
        calibration,
        calibration_sha,
        retention_ratio=retention_ratio,
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    adapter = load_hf_adapter(
        first.model.id,
        revision=first.model.revision,
        model_kwargs=_model_kwargs(first, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    if phase == "calibration-candidate":
        assert requested_variant is not None
        config = _mosaic_config(
            candidates[requested_variant],
            calibration,
            calibration_sha,
            retention_ratio=retention_ratio,
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
            repair_policy=RepairPolicy.NONE,
        )
        _run(
            config,
            adapter,
            run_id=(f"choice-c-calibration-{requested_variant}-{_sha(spec_path)[:10]}"),
            output_root=output_root,
            ledger_path=calibration_path,
            ledger_sha=calibration_sha,
        )
        return 0
    if phase == "calibration-exact-control":
        assert requested_variant is not None
        config = _mosaic_config(
            candidates[requested_variant],
            calibration,
            calibration_sha,
            retention_ratio=retention_ratio,
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
            method_name="mosaickv_exact",
            repair_policy=RepairPolicy.NONE,
        )
        _run(
            config,
            adapter,
            run_id=f"choice-c-calibration-exact-control-v2-{_sha(spec_path)[:10]}",
            output_root=output_root,
            ledger_path=calibration_path,
            ledger_sha=calibration_sha,
        )
        return 0
    if phase in {"internal-method", "internal-exact-control", "heldout-method"}:
        configured_profile_path = os.environ.get(
            "MOSAICKV_FROZEN_PROFILE_OVERRIDE",
            spec.get("frozen_fidelity_profile"),
        )
        profile_path = (
            output_root / "frozen_fidelity_profile.json"
            if configured_profile_path is None
            else Path(str(configured_profile_path)).resolve()
        )
        if not profile_path.exists():
            raise RuntimeError("the frozen fidelity profile is unavailable")
        profile_record = json.loads(profile_path.read_text())
        parameters = profile_record.get("parameters")
        if not isinstance(parameters, dict):
            raise RuntimeError("the frozen fidelity profile has no parameter mapping")
        frozen_parameters: dict[str, Any] = parameters
        if phase in {"internal-method", "internal-exact-control"}:
            if validation is None or validation_path is None or validation_sha is None:
                raise RuntimeError("internal validation requires a validation ledger")
            if phase == "internal-method":
                assert requested_variant is not None
                comparison_ratio = float(baseline_overrides.get(requested_variant, retention_ratio))
                config = (
                    _mosaic_config(
                        frozen_parameters,
                        validation,
                        validation_sha,
                        retention_ratio=retention_ratio,
                        output_length=int(spec["output_length"]),
                        seed=int(spec["seed"]),
                    )
                    if requested_variant == "mosaickv_full"
                    else _baseline_config(
                        requested_variant,
                        validation,
                        validation_sha,
                        retention_ratio=comparison_ratio,
                        output_length=int(spec["output_length"]),
                        seed=int(spec["seed"]),
                    )
                )
                _run(
                    config,
                    adapter,
                    run_id=(
                        f"choice-c-internal-validation-{requested_variant}-{_sha(spec_path)[:10]}"
                    ),
                    output_root=output_root,
                    ledger_path=validation_path,
                    ledger_sha=validation_sha,
                )
                return 0
            _run(
                _mosaic_config(
                    frozen_parameters,
                    validation,
                    validation_sha,
                    retention_ratio=retention_ratio,
                    output_length=int(spec["output_length"]),
                    seed=int(spec["seed"]),
                    method_name="mosaickv_exact",
                    repair_policy=RepairPolicy.NONE,
                ),
                adapter,
                run_id=f"choice-c-internal-validation-exact-{_sha(spec_path)[:10]}",
                output_root=output_root,
                ledger_path=validation_path,
                ledger_sha=validation_sha,
            )
            return 0
        internal_gate_path = output_root / "internal_validation_gate.json"
        if not internal_gate_path.exists():
            raise RuntimeError("held-out workers require the completed internal gate")
        internal_gate = json.loads(internal_gate_path.read_text())
        if internal_gate.get("pass") is not True:
            raise RuntimeError("held-out access is forbidden because the internal gate failed")
        assert requested_variant is not None
        comparison_ratio = float(baseline_overrides.get(requested_variant, retention_ratio))
        config = (
            _mosaic_config(
                frozen_parameters,
                evaluation,
                evaluation_sha,
                retention_ratio=retention_ratio,
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
            )
            if requested_variant == "mosaickv_full"
            else _baseline_config(
                requested_variant,
                evaluation,
                evaluation_sha,
                retention_ratio=comparison_ratio,
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
            )
        )
        _run(
            config,
            adapter,
            run_id=f"choice-c-heldout-{requested_variant}-{_sha(spec_path)[:10]}",
            output_root=output_root,
            ledger_path=evaluation_path,
            ledger_sha=evaluation_sha,
        )
        return 0
    observations: list[FidelityObservation] = []
    calibration_frames: list[pd.DataFrame] = []
    calibration_by_variant: dict[str, pd.DataFrame] = {}
    for variant, parameters in sorted(candidates.items()):
        config = _mosaic_config(
            parameters,
            calibration,
            calibration_sha,
            retention_ratio=float(spec["retention_ratio"]),
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
            repair_policy=RepairPolicy.NONE,
        )
        run_id = f"choice-c-calibration-{variant}-{_sha(spec_path)[:10]}"
        frame = _run(
            config,
            adapter,
            run_id=run_id,
            output_root=output_root,
            ledger_path=calibration_path,
            ledger_sha=calibration_sha,
        )
        frame["variant"] = variant
        calibration_frames.append(frame)
        calibration_by_variant[variant] = frame
        regrets = _forecast_regret_by_sample(output_root, run_id)
        if set(regrets) != set(frame["sample_id"]):
            raise RuntimeError(f"calibration forecast regret is incomplete for {variant}")
        for row in frame.itertuples(index=False):
            observations.append(
                FidelityObservation(
                    variant=variant,
                    sample_id=str(row.sample_id),
                    active_kv_bytes=int(row.active_kv_bytes),
                    token_agreement=float(row.token_agreement),
                    logit_kl=float(row.logit_kl),
                    attention_output_error=float(row.attention_output_error),
                    forecast_regret=regrets[str(row.sample_id)],
                )
            )
        torch.cuda.empty_cache()
    calibration_rows = pd.concat(calibration_frames, ignore_index=True)
    calibration_rows.to_parquet(output_root / "calibration_rows.parquet", index=False)
    target_bytes = math.ceil(
        max(calibration_rows.groupby("variant")["active_kv_bytes"].mean().astype(float))
    )
    try:
        profile = select_fidelity_profile(
            tuple(observations),
            parameters_by_variant=candidates,
            calibration_dataset_id=str(calibration["dataset_id"]),
            calibration_dataset_revision=str(calibration["dataset_revision"]),
            calibration_split=str(calibration["split"]),
            evaluation_sample_ids=tuple(cast("list[str]", evaluation["sample_ids"])),
            target_active_kv_bytes=target_bytes,
            tail_weight=float(spec.get("calibration_rule", {}).get("tail_weight", 0.25)),
            bootstrap_resamples=int(
                spec.get("calibration_rule", {}).get("bootstrap_resamples", 200)
            ),
            bootstrap_seed=int(spec.get("calibration_rule", {}).get("bootstrap_seed", 2027)),
            minimum_selection_frequency=float(
                spec.get("calibration_rule", {}).get("minimum_selection_frequency", 0.0)
            ),
            minimum_calibration_samples=int(
                spec.get("calibration_rule", {}).get("minimum_samples", 2)
            ),
        )
    except CalibrationStabilityError as error:
        failure = {
            "status": "failed",
            "reason": "bootstrap_selection_instability",
            "best_selection_frequency": error.best_frequency,
            "required_selection_frequency": error.required_frequency,
            "full_sweep_allowed": False,
            "heldout_evaluation_started": False,
        }
        (output_root / "calibration_stability_gate.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        (output_root / "gate.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return 0

    selected_base_frame = calibration_by_variant[profile.selected_variant]
    # v1 reserved high-attention partial modality fragments before the dense
    # exact selector could enforce its whole-block mask-free invariant. Keep
    # that failed artifact immutable and use a new run identity after the
    # reservation eligibility correction.
    exact_run_id = f"choice-c-calibration-exact-control-v2-{_sha(spec_path)[:10]}"
    exact_config = _mosaic_config(
        profile.parameters,
        calibration,
        calibration_sha,
        retention_ratio=retention_ratio,
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
        method_name="mosaickv_exact",
        repair_policy=RepairPolicy.NONE,
    )
    exact_frame = _run(
        exact_config,
        adapter,
        run_id=exact_run_id,
        output_root=output_root,
        ledger_path=calibration_path,
        ledger_sha=calibration_sha,
    )
    base_run_id = f"choice-c-calibration-{profile.selected_variant}-{_sha(spec_path)[:10]}"
    exact_head_errors = _head_errors_by_sample(output_root, exact_run_id)
    prototype_head_errors = _head_errors_by_sample(output_root, base_run_id)
    if set(exact_head_errors) != set(prototype_head_errors):
        raise RuntimeError("exact/prototype head diagnostics cover different samples")
    head_observations: list[HeadFidelityObservation] = []
    for sample_id in sorted(exact_head_errors):
        if set(exact_head_errors[sample_id]) != set(prototype_head_errors[sample_id]):
            raise RuntimeError("exact/prototype head diagnostics cover different heads")
        for identity in sorted(exact_head_errors[sample_id]):
            head_observations.append(
                HeadFidelityObservation(
                    sample_id=sample_id,
                    layer=identity[0],
                    kv_head=identity[1],
                    exact_error=exact_head_errors[sample_id][identity],
                    prototype_error=prototype_head_errors[sample_id][identity],
                )
            )
    routing_spec = cast("dict[str, Any]", spec.get("head_routing", {}))
    base_prototypes = cast("dict[str, Any]", profile.parameters["prototypes"])
    routing = derive_adaptive_head_routing(
        tuple(head_observations),
        default_group_size=int(base_prototypes["group_size"]),
        sensitive_group_size=int(routing_spec.get("sensitive_group_size", 11)),
        sensitive_position_span=int(routing_spec.get("sensitive_position_span", 127)),
        damage_threshold=float(routing_spec.get("damage_threshold", 0.02)),
        exact_head_fraction=float(routing_spec.get("exact_head_fraction", 0.02)),
    )
    adaptive_parameters = json.loads(json.dumps(profile.parameters))
    adaptive_prototypes = cast("dict[str, Any]", adaptive_parameters["prototypes"])
    adaptive_prototypes.update(
        {
            "adaptive_grouping": True,
            "exact_head_routes": list(routing.exact_head_routes),
            "head_group_sizes": list(routing.head_group_sizes),
            "head_position_spans": list(routing.head_position_spans),
            "forecast_weight": float(routing_spec.get("forecast_weight", 0.5)),
        }
    )
    adaptive_repair = cast("dict[str, Any]", adaptive_parameters["repair"])
    adaptive_repair.update(
        {
            "normalize_prototype_risk": True,
            "candidate_ranking": RepairCandidateRanking.DECISION_INFLUENCE.value,
        }
    )
    adaptive_run_id = f"choice-c-calibration-adaptive-heads-{_sha(spec_path)[:10]}"
    adaptive_config = _mosaic_config(
        adaptive_parameters,
        calibration,
        calibration_sha,
        retention_ratio=retention_ratio,
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
        repair_policy=RepairPolicy.NONE,
    )
    adaptive_frame = _run(
        adaptive_config,
        adapter,
        run_id=adaptive_run_id,
        output_root=output_root,
        ledger_path=calibration_path,
        ledger_sha=calibration_sha,
    )
    adaptive_effective = _effective_method_by_sample(output_root, adaptive_run_id)
    adaptive_three_tier = bool(
        len(adaptive_effective) == len(adaptive_frame)
        and set(adaptive_effective.values()) == {"mosaickv_full"}
    )
    base_error = float(_fidelity_error(selected_base_frame).mean())
    adaptive_error = float(_fidelity_error(adaptive_frame).mean())
    exact_error = float(_fidelity_error(exact_frame).mean())
    adaptive_selected = adaptive_three_tier and adaptive_error < base_error
    frozen_parameters = adaptive_parameters if adaptive_selected else profile.parameters
    frozen_calibration_run_id = adaptive_run_id if adaptive_selected else base_run_id
    frozen_variant = (
        f"{profile.selected_variant}__adaptive_heads"
        if adaptive_selected
        else profile.selected_variant
    )
    prototype_calibration_pass = min(base_error, adaptive_error) < exact_error
    calibration_gate = {
        "base_variant": profile.selected_variant,
        "base_fidelity_error": base_error,
        "adaptive_fidelity_error": adaptive_error,
        "exact_control_fidelity_error": exact_error,
        "adaptive_selected": adaptive_selected,
        "adaptive_genuine_three_tier": adaptive_three_tier,
        "prototype_beats_exact_control": prototype_calibration_pass,
        "exact_head_routes": list(routing.exact_head_routes),
        "sensitive_head_count": len(routing.head_group_sizes),
    }
    dispersion_spec = cast("dict[str, Any]", spec.get("prototype_safety_calibration", {}))
    maximum_key_dispersion, maximum_value_dispersion = _prototype_dispersion_limits(
        output_root,
        frozen_calibration_run_id,
        safety_factor=float(dispersion_spec.get("safety_factor", 1.25)),
    )
    frozen_parameters = json.loads(json.dumps(frozen_parameters))
    frozen_prototypes = cast("dict[str, Any]", frozen_parameters["prototypes"])
    frozen_prototypes.update(
        {
            "max_key_dispersion": maximum_key_dispersion,
            "max_value_dispersion": maximum_value_dispersion,
        }
    )
    calibration_gate.update(
        {
            "maximum_key_dispersion": maximum_key_dispersion,
            "maximum_value_dispersion": maximum_value_dispersion,
            "dispersion_safety_factor": float(dispersion_spec.get("safety_factor", 1.25)),
        }
    )
    (output_root / "prototype_calibration_gate.json").write_text(
        json.dumps(calibration_gate, indent=2, sort_keys=True) + "\n"
    )
    if not prototype_calibration_pass:
        profile = replace(
            profile,
            selected_variant=frozen_variant,
            parameters=frozen_parameters,
        )
        profile.write(output_root / "frozen_fidelity_profile.json")
        failure = {
            "status": "failed",
            "reason": "prototype_did_not_beat_exact_control_on_calibration",
            "prototype_calibration_pass": False,
            "repair_calibration_started": False,
            "internal_validation_started": False,
            "heldout_evaluation_started": False,
            "full_sweep_allowed": False,
        }
        (output_root / "gate.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return 0

    oracle_parameters = json.loads(json.dumps(frozen_parameters))
    oracle_repair = cast("dict[str, Any]", oracle_parameters["repair"])
    oracle_repair.update(
        {
            "normalize_prototype_risk": True,
            "candidate_ranking": RepairCandidateRanking.DECISION_INFLUENCE.value,
        }
    )
    oracle_run_id = f"choice-c-calibration-oracle-repair-{_sha(spec_path)[:10]}"
    oracle_config = _mosaic_config(
        oracle_parameters,
        calibration,
        calibration_sha,
        retention_ratio=retention_ratio,
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
        repair_policy=RepairPolicy.ORACLE,
        repair_evaluation_only=True,
    )
    _run(
        oracle_config,
        adapter,
        run_id=oracle_run_id,
        output_root=output_root,
        ledger_path=calibration_path,
        ledger_sha=calibration_sha,
    )
    repair_spec = cast("dict[str, Any]", spec.get("repair_calibration", {}))
    repair_profile = calibrate_decision_repair(
        _repair_calibration_observations(output_root, oracle_run_id),
        evaluation_sample_ids=tuple(cast("list[str]", evaluation["sample_ids"])),
        target_trigger_rate=float(repair_spec.get("target_trigger_rate", 0.1)),
        minimum_precision=float(repair_spec.get("minimum_precision", 0.5)),
    )
    repair_profile.write(output_root / "frozen_repair_profile.json")
    final_parameters = json.loads(json.dumps(frozen_parameters))
    final_repair = cast("dict[str, Any]", final_parameters["repair"])
    final_repair.update(
        {
            "normalize_prototype_risk": True,
            "prototype_risk_threshold": repair_profile.prototype_risk_threshold,
            "decision_margin_threshold": repair_profile.decision_margin_threshold,
            "decision_high_risk_threshold": repair_profile.decision_high_risk_threshold,
            "draft_kl_threshold": repair_profile.draft_kl_threshold,
            "candidate_ranking": RepairCandidateRanking.DECISION_INFLUENCE.value,
            "policy": (
                RepairPolicy.DECISION_AWARE.value
                if repair_profile.enabled
                else RepairPolicy.NONE.value
            ),
        }
    )
    profile = replace(
        profile,
        selected_variant=frozen_variant,
        parameters=final_parameters,
    )
    profile.write(output_root / "frozen_fidelity_profile.json")
    print(
        f"choice-c frozen variant={profile.selected_variant} profile_sha={profile.sha256}",
        flush=True,
    )
    repair_gate = {
        "enabled": repair_profile.enabled,
        "draft_kl_threshold": repair_profile.draft_kl_threshold,
        "predicted_precision": repair_profile.predicted_precision,
        "predicted_recall": repair_profile.predicted_recall,
        "predicted_triggered_repairs": repair_profile.predicted_triggered_repairs,
        "predicted_beneficial_repairs": repair_profile.predicted_beneficial_repairs,
        "minimum_precision": float(repair_spec.get("minimum_precision", 0.5)),
        "target_trigger_rate": float(repair_spec.get("target_trigger_rate", 0.1)),
        "pass": bool(
            repair_profile.enabled
            and repair_profile.predicted_precision
            >= float(repair_spec.get("minimum_precision", 0.5))
        ),
    }
    (output_root / "repair_calibration_gate.json").write_text(
        json.dumps(repair_gate, indent=2, sort_keys=True) + "\n"
    )
    if not repair_gate["pass"]:
        failure = {
            "status": "failed",
            "reason": "decision_repair_calibration_gate_failed",
            "prototype_calibration_pass": prototype_calibration_pass,
            "repair_calibration_enabled": repair_profile.enabled,
            "repair_calibration_precision": repair_profile.predicted_precision,
            "repair_calibration_recall": repair_profile.predicted_recall,
            "repair_calibration_minimum_precision": repair_gate["minimum_precision"],
            "internal_validation_started": False,
            "heldout_evaluation_started": False,
            "full_sweep_allowed": False,
        }
        (output_root / "gate.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        report = [
            "# Choice C calibration gate",
            "",
            f"- Frozen variant: `{profile.selected_variant}` (`{profile.sha256}`)",
            "- Prototype beats exact calibration control: True",
            "- Decision-aware repair enabled: False",
            "- Calibrated repair precision/recall: "
            f"{repair_profile.predicted_precision:.4f}/"
            f"{repair_profile.predicted_recall:.4f}",
            "- Internal validation started: False",
            "- Held-out evaluation started: False",
            "- Full sweep allowed: **False**",
            "",
            "The preregistered repair gate failed. Calibration artifacts are retained, "
            "but the disjoint internal and held-out ledgers remain unconsumed.",
        ]
        (output_root / "gate_report.md").write_text("\n".join(report) + "\n")
        return 0

    internal_validation_required = bool(
        spec.get("calibration_rule", {}).get("require_internal_validation", False)
    )
    internal_validation_pass = not internal_validation_required
    internal_validation_report: dict[str, Any] = {
        "required": internal_validation_required,
        "available": validation is not None,
    }
    if validation is not None and validation_path is not None and validation_sha is not None:
        final_validation_run_id = f"choice-c-internal-validation-mosaickv-{_sha(spec_path)[:10]}"
        final_validation_frame = _run(
            _mosaic_config(
                profile.parameters,
                validation,
                validation_sha,
                retention_ratio=retention_ratio,
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
            ),
            adapter,
            run_id=final_validation_run_id,
            output_root=output_root,
            ledger_path=validation_path,
            ledger_sha=validation_sha,
        )
        exact_validation_run_id = f"choice-c-internal-validation-exact-{_sha(spec_path)[:10]}"
        exact_validation_frame = _run(
            _mosaic_config(
                profile.parameters,
                validation,
                validation_sha,
                retention_ratio=retention_ratio,
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
                method_name="mosaickv_exact",
                repair_policy=RepairPolicy.NONE,
            ),
            adapter,
            run_id=exact_validation_run_id,
            output_root=output_root,
            ledger_path=validation_path,
            ledger_sha=validation_sha,
        )
        effective = _effective_method_by_sample(output_root, final_validation_run_id)
        genuine = bool(
            len(effective) == len(final_validation_frame)
            and set(effective.values()) == {"mosaickv_full"}
        )
        mosaic_validation_error = float(_fidelity_error(final_validation_frame).mean())
        exact_validation_error = float(_fidelity_error(exact_validation_frame).mean())
        internal_validation_pass = bool(
            genuine and mosaic_validation_error < exact_validation_error
        )
        internal_validation_report.update(
            {
                "genuine_three_tier": genuine,
                "mosaickv_fidelity_error": mosaic_validation_error,
                "exact_control_fidelity_error": exact_validation_error,
                "pass": internal_validation_pass,
            }
        )
    elif internal_validation_required:
        internal_validation_report["pass"] = False
    (output_root / "internal_validation_gate.json").write_text(
        json.dumps(internal_validation_report, indent=2, sort_keys=True) + "\n"
    )

    heldout_frames: list[pd.DataFrame] = []
    for method in methods:
        config = (
            _mosaic_config(
                profile.parameters,
                evaluation,
                evaluation_sha,
                retention_ratio=float(spec["retention_ratio"]),
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
            )
            if method == "mosaickv_full"
            else _baseline_config(
                method,
                evaluation,
                evaluation_sha,
                retention_ratio=float(spec["retention_ratio"]),
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
            )
        )
        run_id = f"choice-c-heldout-{method}-{_sha(spec_path)[:10]}"
        frame = _run(
            config,
            adapter,
            run_id=run_id,
            output_root=output_root,
            ledger_path=evaluation_path,
            ledger_sha=evaluation_sha,
        )
        frame["requested_method"] = method
        effective_methods = _effective_method_by_sample(output_root, run_id)
        if set(effective_methods) != set(frame["sample_id"]):
            raise RuntimeError(f"effective-method trace is incomplete for {run_id}")
        frame["effective_method"] = frame["sample_id"].map(effective_methods)
        heldout_frames.append(frame)
        torch.cuda.empty_cache()
    heldout = pd.concat(heldout_frames, ignore_index=True)
    heldout.to_parquet(output_root / "heldout_rows.parquet", index=False)
    summary, gates = _analyze_heldout(heldout, spec)
    gates["prototype_calibration_pass"] = prototype_calibration_pass
    gates["repair_calibration_enabled"] = repair_profile.enabled
    gates["repair_calibration_precision"] = repair_profile.predicted_precision
    gates["repair_calibration_recall"] = repair_profile.predicted_recall
    gates["internal_validation_pass"] = internal_validation_pass
    gates["full_sweep_allowed"] = bool(
        gates["full_sweep_allowed"]
        and prototype_calibration_pass
        and repair_profile.enabled
        and repair_profile.predicted_precision >= float(repair_spec.get("minimum_precision", 0.5))
        and internal_validation_pass
    )
    summary.to_parquet(output_root / "heldout_summary.parquet", index=False)
    (output_root / "gate.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    report = [
        "# Choice C held-out gate",
        "",
        f"- Frozen variant: `{profile.selected_variant}` (`{profile.sha256}`)",
        f"- Nearest byte-matched competitor: `{gates['nearest_competitor']}`",
        f"- Relative byte mismatch: {100 * float(gates['relative_byte_mismatch']):.2f}%",
        "- Relative fidelity improvement: "
        f"{100 * float(gates['relative_fidelity_improvement']):.2f}%",
        f"- MosaicKV task score: {float(gates['mosaickv_task_score']):.6f}",
        f"- Best comparator task score: {float(gates['best_baseline_task_score']):.6f}",
        "- Genuine three-tier MosaicKV rows: "
        f"{gates['genuine_three_tier_rows']}/{gates['required_genuine_three_tier_rows']}",
        f"- Prototype beats exact calibration control: {prototype_calibration_pass}",
        "- Calibrated repair precision/recall: "
        f"{repair_profile.predicted_precision:.4f}/{repair_profile.predicted_recall:.4f}",
        f"- Disjoint internal validation pass: {internal_validation_pass}",
        f"- Full sweep allowed: **{gates['full_sweep_allowed']}**",
        "",
        "This preregistered small-sample result is an exploratory engineering gate, "
        "not a paper claim.",
    ]
    (output_root / "gate_report.md").write_text("\n".join(report) + "\n")
    # A negative scientific gate is a valid completed experiment. Runtime or
    # data-integrity failures raise above and remain non-zero; gate.json alone
    # controls whether a subsequent full sweep is scientifically permitted.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "calibration-candidate",
            "calibration-exact-control",
            "internal-method",
            "internal-exact-control",
            "heldout-method",
        ),
        default="all",
    )
    parser.add_argument("--variant")
    args = parser.parse_args()
    phases_requiring_variant = {
        "calibration-candidate",
        "calibration-exact-control",
        "internal-method",
        "heldout-method",
    }
    if args.phase in {"all", "internal-exact-control"} and args.variant is not None:
        parser.error(f"--variant is not valid with --phase {args.phase}")
    if args.phase in phases_requiring_variant and args.variant is None:
        parser.error(f"--phase {args.phase} requires --variant")
    return run(
        args.spec.resolve(),
        phase=args.phase,
        requested_variant=args.variant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
