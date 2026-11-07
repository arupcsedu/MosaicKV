"""Fail-closed rules for new comparisons; never relabel historical artifacts.

This is a validation boundary, not an experiment launcher or metric reducer.
Metric-aware reducers remain in evaluation.metrics. Failed/missing observations
block a complete comparison; they are never silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import RunConfig, canonical_config
from mosaickv.evaluation.tasks import default_task_registry

POLICY_FIELDS = ("forecasting", "graph", "utility", "selection", "prototypes", "residual", "repair")
CONTROL_FIELDS = (
    "model",
    "processor",
    "chat_template",
    "prompts",
    "media",
    "preprocessing",
    "tokenization",
    "generation",
    "precision",
    "backend",
    "attention",
    "block_policy",
    "mandatory_policy",
    "decode_growth_policy",
    "measurement",
    "gpu_controls",
    "dataset",
    "references",
    "evaluator",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_content_digest(source: Path) -> str:
    """Freeze code contents, independent of unrelated paper/scheduler edits.

    Commit SHA is additionally required at submission. This fingerprint permits
    only an identical frozen implementation, not a silent candidate retune.
    """
    paths = sorted((source / "mosaickv/src/mosaickv").rglob("*.py"))
    paths += [
        source / "mosaickv" / p
        for p in (
            "scripts/run_baseline_fidelity_review_worker.py",
            "scripts/prepare_native_metric_shared_bytes.py",
            "configs/paper_candidate_llava_v1.yaml",
        )
    ]
    return digest({str(p.relative_to(source)): file_digest(p) for p in paths})


def policy_digest(config: RunConfig) -> str:
    resolved = canonical_config(config)
    return digest({key: resolved[key] for key in POLICY_FIELDS})


def load_contract(path: Path) -> dict[str, Any]:
    contract = cast("dict[str, Any]", yaml.safe_load(path.read_text()))
    if contract.get("schema_version") != 2:
        raise ValueError("unrecognized comparison contract")
    resources = contract["resources"]
    if resources.get("max_global_a100s") not in {4, 6} or {
        key: value for key, value in resources.items() if key != "max_global_a100s"
    } != {
        "gpus_per_worker": 1,
        "cpus_per_worker": 12,
        "time_limit_seconds": 7200,
    }:
        raise ValueError(
            "require an approved four- or six-A100 global limit and "
            "one GPU/12 CPUs/two hours per worker"
        )
    if contract["execution_authorized"] or contract["fullkv_latency_comparison"]:
        raise ValueError("rules registration does not authorize execution or FullKV latency")
    for lane in contract["lanes"].values():
        for name, route in lane["tasks"].items():
            task = default_task_registry().resolve(name)
            if (route["dataset_id"], route["split"]) != (task.dataset_id, task.split):
                raise ValueError(f"registry route mismatch: {name}")
            revision = route.get("dataset_revision")
            if revision is not None and (
                len(revision) not in (40, 64) or any(c not in "0123456789abcdef" for c in revision)
            ):
                raise ValueError(f"unresolved revision: {name}")
    return contract


def validate_configuration(
    contract: Mapping[str, Any],
    lane: str,
    task: str,
    config: RunConfig,
) -> None:
    if config.output_preserving is not None:
        raise ValueError("output-preserving development candidate is not the frozen lane policy")
    rule = contract["lanes"][lane]
    route = rule["tasks"][task]
    if route["dataset_revision"] is None:
        raise ValueError("dataset revision unresolved: bind reviewed materialized ledger first")
    if (config.model.id, config.model.revision) != (rule["model_id"], rule["model_revision"]):
        raise ValueError("model checkpoint/revision mismatch")
    if (config.dataset.id, config.dataset.revision, config.dataset.split) != (
        route["dataset_id"],
        route["dataset_revision"],
        route["split"],
    ):
        raise ValueError("dataset route mismatch")
    if (
        config.execution.backend.value != "huggingface"
        or config.execution.attention_implementation != "eager"
        or config.execution.decode_attention_implementation not in (None, "eager")
        or config.model.precision.value != "fp16"
        or config.generation.do_sample
        or config.generation.temperature != 0
        or config.generation.top_p != 1
        or config.generation.output_length_policy != "fixed_max_new_tokens"
        or config.generation.max_new_tokens != route["output_tokens"]
        or config.execution.seed != contract["comparison"]["seed"]
        or config.cache.block_size != 1
        or config.cache.physical_budget_block_size != 1
        or config.cache.physical_budget_mandatory_policy != rule["mandatory_policy"]
    ):
        raise ValueError("execution/generation/cache controls mismatch")
    if config.method.value not in rule["methods"]:
        raise ValueError("method not registered")
    if config.method.value != "full_kv" and (
        config.cache.budget_unit.value != "bytes"
        or config.cache.budget_value not in [m * 2**20 for m in rule["caps_mib"]]
    ):
        raise ValueError("unregistered physical byte target")
    if config.method.value == "mosaickv_exact" and (
        policy_digest(config) != contract["candidate"]["policy_sha256"]
    ):
        raise ValueError("candidate policy changed after freeze")


def validate_ledger(
    path: Path,
    expected_sha: str,
    route: Mapping[str, Any],
    task: str,
) -> tuple[str, ...]:
    if file_digest(path) != expected_sha:
        raise ValueError("ledger hash mismatch")
    payload = json.loads(path.read_text())
    if payload.get("task") != task or any(
        payload.get(k) != route[k] for k in ("dataset_id", "dataset_revision", "split")
    ):
        raise ValueError("ledger route mismatch")
    ids = payload.get("sample_ids")
    if ids is None:
        ids = [s["sample_id"] for s in payload.get("samples", [])]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
        raise ValueError("empty or duplicate ledger identities")
    return tuple(ids)


def validate_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    methods: Sequence[str],
    cap_bytes: int,
    metric: str,
    expected_controls: Mapping[str, Mapping[str, str]],
    capacity_certificates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """One task/metric/cap, strictly paired; FullKV is reference-only.

    Input is an explicit derived comparison ledger, not a rewrite of raw JSONL.
    Storage is actual unique backing storage (padding/preallocation included).
    The capacity certificate must be independently produced by layout feasibility
    validation and hash-bound in the plan, never inferred from an underfilled row.
    """
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("invalid planned sample identities")
    if set(expected_controls) != set(sample_ids):
        raise ValueError("planned control fingerprints must cover every sample")
    if len(methods) != len(set(methods)) or "mosaickv_exact" not in methods:
        raise ValueError("duplicate methods or missing frozen candidate")
    compressed = [m for m in methods if m != "full_kv"]
    if len(compressed) < 2:
        raise ValueError("need at least two compressed methods")
    keyed = {}
    for row in rows:
        key = (row["method"], row["sample_id"])
        if key in keyed or key[0] not in methods or key[1] not in sample_ids:
            raise ValueError("duplicate or unexpected comparison row")
        keyed[key] = row
    expected = {(m, s) for m in methods for s in sample_ids}
    if set(keyed) != expected:
        raise ValueError("missing rows; preserve denominator and retry only missing/failed shards")
    cohorts = {(r["lane"], r["task"], r["metric_name"]) for r in rows}
    if len(cohorts) != 1 or next(iter(cohorts))[2] != metric:
        raise ValueError("do not mix lanes/tasks/metrics")
    for sid in sample_ids:
        paired = [keyed[m, sid] for m in methods]
        for row in paired:
            if (
                row.get("status") != "completed"
                or row.get("result_category") != "unified_comparison"
            ):
                raise ValueError("failed/historical row cannot enter new matched comparison")
            if row.get("control_hashes", {}).keys() != set(CONTROL_FIELDS):
                raise ValueError("missing controlled-comparison fingerprints")
            if any(
                not isinstance(h, str)
                or len(h) != 64
                or any(c not in "0123456789abcdef" for c in h)
                for h in row["control_hashes"].values()
            ):
                raise ValueError("invalid control fingerprint")
            if row["control_hashes"] != paired[0]["control_hashes"]:
                raise ValueError("model/input/generation/measurement controls differ")
            if row["control_hashes"] != expected_controls[sid]:
                raise ValueError("controls differ from the immutable plan")
        footprints = []
        for method in compressed:
            row = keyed[method, sid]
            if row.get("physical_byte_cap") != cap_bytes:
                raise ValueError("configured physical cap mismatch")
            initial, final, peak = (
                row.get(f"{phase}_active_kv_storage_bytes")
                for phase in ("initial", "final", "peak")
            )
            if any(type(v) is not int or v <= 0 for v in (initial, final, peak)):
                raise ValueError("missing measured initial/final/peak backing-storage bytes")
            initial, final, peak = cast("tuple[int, int, int]", (initial, final, peak))
            if initial > cap_bytes or final < initial or peak < final:
                raise ValueError("invalid append-only physical storage accounting")
            if row.get("largest_feasible_initial_bytes") != initial:
                raise ValueError("underfilled cache: not the largest feasible allocation")
            if not row.get("capacity_certificate_sha256"):
                raise ValueError("missing independently bound capacity certificate")
            certificate = capacity_certificates.get((method, sid), {})
            if (
                digest(certificate) != row["capacity_certificate_sha256"]
                or certificate.get("physical_byte_cap") != cap_bytes
                or certificate.get("largest_feasible_initial_bytes") != initial
                or certificate.get("boundary") != "unique_live_backing_storage"
                or certificate.get("status") != "passed"
            ):
                raise ValueError("capacity certificate does not validate this allocation")
            if row.get("decode_evictions") != 0 or row.get("used_fullkv_fallback") is not False:
                raise ValueError("non-append-only/fallback result is not comparable")
            footprints.append((initial, final, peak))
        if len(set(footprints)) != 1:
            raise ValueError("realized bytes differ; matching nominal ratios/caps is insufficient")
    return {
        "matched": True,
        "planned_samples": len(sample_ids),
        "completed_rows": len(rows),
        "failed_rows": 0,
        "metric_name": metric,
        "latency_methods": compressed,
        "fullkv_role": "quality_reference_only",
    }


def reuse_decision(existing: Mapping[str, Any] | None, planned: Mapping[str, Any]) -> str:
    """Classify, never delete/relabel or automatically rerun historical rows."""
    if existing is None:
        return "retry_missing"
    identity = (
        "sample_id",
        "method",
        "lane",
        "task",
        "metric_name",
        "control_hashes",
        "physical_byte_cap",
        "candidate_content_sha256",
        "baseline_commit",
    )
    if any(k not in planned or existing.get(k) != planned[k] for k in identity):
        return "preserve_incompatible_requires_new_plan"
    if existing.get("status") != "completed":
        return "retry_failed"
    if existing.get("comparison_validation_passed") is not True:
        return "preserve_unverified_requires_artifact_audit"
    return "reuse_completed"
