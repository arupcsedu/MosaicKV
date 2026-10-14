"""Frozen shared-byte development validation; never promotes a paper claim.

PrefixKV question scoring and append-only decoding are explicitly local variants.
All methods share the initial physical cap and append one identical-width KV
position per decode call. FullKV is a quality control, never a latency competitor.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from mosaickv.baselines.prefixkv import PrefixKVOfflineProfile
from mosaickv.config import LookMConfig, PrefixKVConfig
from mosaickv.types import MosaicKVMethod

CONTROL = "prefixkv_byte_calibrated_selection_only_reimpl"
CANDIDATE = "prefixkv_question_scored_selection_only_reimpl"
PREFIX_METHODS = ("full_kv", "prompt_attention_topk", CONTROL, CANDIDATE, "mosaickv_exact")
LOOKM_METHODS = ("full_kv", "prompt_attention_topk", "lookm_reimpl", "mosaickv_exact")


def matched_configs(controls, profiles=None, *, lane="prefixkv"):
    """Validate controls and bind existing frozen profiles; no fitting or tuning."""
    required = {"full_kv", "prompt_attention_topk", "mosaickv_exact"}
    if set(controls) != required or lane not in {"prefixkv", "lookm"}:
        raise ValueError("unsupported lane or controls")
    template = controls["prompt_attention_topk"]
    for name, cfg in controls.items():
        if cfg.method.value != name:
            raise ValueError("mislabeled control")
        for field in ("model", "dataset", "generation", "execution"):
            if getattr(cfg, field) != getattr(template, field):
                raise ValueError(f"control {field} mismatch")
        if name != "full_kv" and cfg.cache != template.cache:
            raise ValueError("cache policy mismatch")
        if cfg.prototypes.enabled or cfg.residual.enabled or cfg.repair.enabled:
            raise ValueError("only exact online controls are admitted")
    if (
        template.cache.block_size != 1
        or template.cache.physical_budget_mandatory_policy != "first_and_last"
        or template.execution.attention_implementation != "eager"
        or template.execution.backend.value != "huggingface"
        or template.generation.do_sample
    ):
        raise ValueError("require deterministic HF eager, one-token blocks and first/last")
    result = dict(controls)
    if lane == "lookm":
        if profiles:
            raise ValueError("LOOK-M does not consume PrefixKV profiles")
        result["lookm_reimpl"] = replace(
            template,
            method=MosaicKVMethod.LOOKM_REIMPL,
            lookm=LookMConfig(enabled=True, recent_ratio=0.495, important_ratio=0.495),
        )
        return {name: result[name] for name in LOOKM_METHODS}
    if set(profiles or {}) != {"control", "c_rank"}:
        raise ValueError("require the frozen control and C-only profiles")
    identity = None
    for arm, label, alpha in (("control", CONTROL, 0.0), ("c_rank", CANDIDATE, 0.5)):
        path = Path(profiles[arm])
        profile = PrefixKVOfflineProfile.from_json_object(json.loads(path.read_text()))
        observed = (
            profile.calibration_dataset_id,
            profile.calibration_dataset_revision,
            profile.calibration_split,
            profile.calibration_sample_ids,
            profile.calibration_seed,
            profile.calibration_bytes_per_layer_position,
        )
        if identity is not None and observed != identity:
            raise ValueError("calibration identity mismatch")
        identity = observed
        if (
            profile.model_id != template.model.id
            or profile.model_revision != template.model.revision
            or profile.calibration_byte_cap != template.cache.budget_value
            or profile.question_score_weight != alpha
            or profile.search_policy != "upstream_protected_cdf"
            or profile.prompt_reservation_policy != "boundaries"
            or (profile.start_size, profile.protect_size) != (1, 1)
        ):
            raise ValueError("frozen C-only/control profile contract differs")
        result[label] = replace(
            template,
            method=MosaicKVMethod.PREFIXKV_REIMPL,
            prefixkv=PrefixKVConfig(
                enabled=True,
                profile_path=str(path),
                rescale_offline_profile_to_byte_budget=True,
                profile_reference_retention_ratio=profile.target_retention_ratio,
                hf_cache_transport="ragged_llama_eager_v1",
                decode_boundary_policy="protected",
                decode_policy="append_only",
                question_score_weight=alpha,
            ),
        )
    return {name: result[name] for name in PREFIX_METHODS}


def select_disjoint_ids(parent_ids, excluded_ids, *, validation_count=64, larger_count=256):
    """ID-only selection, independent stages; not a globally untouched claim."""
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate parent IDs")
    if validation_count < 1 or larger_count < 1:
        raise ValueError("positive stage sizes required")
    excluded = set(excluded_ids)
    available = [x for x in parent_ids if x not in excluded]
    if len(available) < validation_count + larger_count:
        raise ValueError("insufficient disjoint IDs")
    return available[:validation_count], available[
        validation_count : validation_count + larger_count
    ]


def paired_statistics(left, right, ids, *, seed=20271014, resamples=10000):
    import numpy as np

    if set(left) != set(ids) or set(right) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("paired identities must match exactly")
    delta = np.array([left[s]["task_score"] - right[s]["task_score"] for s in ids])
    a = np.array([left[s]["end_to_end_time"] for s in ids])
    b = np.array([right[s]["end_to_end_time"] for s in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    return {
        "paired_score_delta": float(delta.mean()),
        "paired_score_ci95": np.quantile(delta[idx].mean(1), [0.025, 0.975]).tolist(),
        "answer_agreement": float(np.mean([left[s]["answer"] == right[s]["answer"] for s in ids])),
        "latency_ratio_of_paired_means": float(a.mean() / b.mean()),
        "latency_ratio_ci95": np.quantile(a[idx].mean(1) / b[idx].mean(1), [0.025, 0.975]).tolist(),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def summarize(records, ids, *, methods=PREFIX_METHODS, cap=40 * 2**20, output_tokens=64):
    """Fail closed on incomplete rows, invalid timings, unequal initial/final bytes.

    Promotion requires observed score parity AND the existing five-point paired
    noninferiority bound. It is not evidence of exact population quality equality.
    """
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("invalid planned identities")
    by_method = {m: {} for m in methods}
    errors = []
    for record in records:
        method = record["variant"]
        if method not in by_method:
            raise ValueError("unexpected method")
        for row in record["rows"]:
            sid = row["sample_id"]
            if sid not in ids or sid in by_method[method]:
                raise ValueError("unexpected or duplicate sample")
            by_method[method][sid] = row
    for method, rows in by_method.items():
        if set(rows) != set(ids):
            errors.append(f"incomplete: {method}")
        for row in rows.values():
            values = [
                row.get(k)
                for k in ("task_score", "end_to_end_time", "compression_time", "decode_time")
            ]
            if (
                row.get("status") != "completed"
                or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values
                )
                or not row.get("end_to_end_time", 0)
                or row.get("generated_tokens") != output_tokens
            ):
                errors.append(f"invalid row: {method}")
            if method != "full_kv" and (
                row.get("active_kv_bytes") != cap
                or row.get("residual_kv_bytes") != 0
                or row.get("repair_count") != 0
                or row.get("decode_evictions") != 0
                or not row.get("trace_validated", False)
            ):
                errors.append(f"cache/trace mismatch: {method}")
    if not errors:
        for sid in ids:
            final = [
                by_method[m][sid].get("final_active_kv_bytes") for m in methods if m != "full_kv"
            ]
            if None in final or len(set(final)) != 1 or final[0] < cap:
                errors.append("unequal final physical bytes")
            input_hashes = {by_method[m][sid].get("input_controls_sha256") for m in methods}
            cache_hashes = {
                by_method[m][sid].get("cache_controls_sha256") for m in methods if m != "full_kv"
            }
            if None in input_hashes or len(input_hashes) != 1:
                errors.append("model/input/generation controls mismatch")
            if None in cache_hashes or len(cache_hashes) != 1:
                errors.append("physical cache policies mismatch")
            if len({by_method[m][sid].get("reference") for m in methods}) != 1:
                errors.append("reference annotation mismatch")
    result = {
        "gates_passed": False,
        "errors": sorted(set(errors)),
        "planned_samples": len(ids),
        "counts": {m: len(r) for m, r in by_method.items()},
        "methods": {},
        "fullkv_latency_excluded": True,
        "paper_claim_authorized": False,
    }
    if errors:
        return result
    for method, rows in by_method.items():
        result["methods"][method] = {
            "task_score": sum(r["task_score"] for r in rows.values()) / len(ids),
            "active_bytes": sorted({r["active_kv_bytes"] for r in rows.values()}),
        }
        if method != "full_kv":
            result["methods"][method]["vs_topk"] = paired_statistics(
                rows, by_method["prompt_attention_topk"], ids
            )
    candidate = CANDIDATE if CANDIDATE in methods else "lookm_reimpl"
    pair = result["methods"][candidate]["vs_topk"]
    informative = all(
        result["methods"][m]["task_score"] > 0 for m in ("full_kv", "prompt_attention_topk")
    )
    result["gates_passed"] = bool(
        informative
        and pair["paired_score_delta"] >= -1e-12
        and pair["paired_score_ci95"][0] >= -0.05
    )
    result["quality_informative"] = informative
    result["memory_gate"] = True
    result["latency_informational_only"] = True
    return result
