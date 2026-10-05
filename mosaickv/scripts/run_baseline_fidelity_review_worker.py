#!/usr/bin/env python3
"""Run one paired published-baseline fidelity shard on an A100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from scripts.run_paper_candidate_v1_preflight_worker import _control_configs, _record, _settle

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import (
    CacheConfig,
    ForecastingConfig,
    GraphConfig,
    LookMConfig,
    ModelConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    VLCacheConfig,
    load_config,
)
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import (
    BudgetUnit,
    ForecastMode,
    MosaicKVMethod,
    Precision,
    PrefixKVProfileMode,
    RepairPolicy,
)

CLASSIFICATION = "canonical_published_baseline_fidelity_diagnostic"
FAMILY_METHOD = {"prefixkv": "prefixkv_reimpl", "vl_cache": "vl_cache_reimpl"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast("dict[str, Any]", value)


def _retention_key(value: float) -> str:
    return f"{value:.2f}"


def review_configs(
    template: Any,
    *,
    repository: Path,
    family: str,
    task: str,
    model_id: str,
    model_revision: str,
    seed: int,
    output_length: int,
    retention_ratio: float,
    profile_forget_ratio: str | None = None,
) -> dict[str, Any]:
    """Build exact paired controls for one model, task, and retention ratio."""

    model = ModelConfig(model_id, model_revision, Precision.FP16)
    base = replace(template, model=model)
    prompt = _control_configs(base, task=task, seed=seed, output_length=output_length)[
        "prompt_attention_topk"
    ]
    cache = CacheConfig(
        budget_value=2**63 - 1,
        budget_unit=BudgetUnit.BYTES,
        retention_ratio=retention_ratio,
        block_size=1,
        accounting_spec_sha=prompt.cache.accounting_spec_sha,
        physical_budget_block_size=1,
        physical_budget_mandatory_policy="method",
        exact_gather_mode=prompt.cache.exact_gather_mode,
        packing_mode=prompt.cache.packing_mode,
        total_storage_budget_bytes=None,
    )
    prompt = replace(prompt, model=model, cache=cache)
    full = replace(
        prompt,
        method=MosaicKVMethod.FULL_KV,
        cache=replace(cache, retention_ratio=1.0),
    )
    common = {
        "prototypes": replace(prompt.prototypes, enabled=False),
        "residual": ResidualConfig(enabled=False),
        "repair": RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
        "lookm": LookMConfig(enabled=False),
    }
    if family == "prefixkv":
        if profile_forget_ratio is None:
            raise ValueError("PrefixKV requires a pinned profile ratio")
        profile = (
            repository
            / "third_party/PrefixKV/confs"
            / f"prefixkv_llava-v1.5-7b_{profile_forget_ratio}.json"
        )
        method = replace(
            prompt,
            method=MosaicKVMethod.PREFIXKV_REIMPL,
            cache=cache,
            prefixkv=PrefixKVConfig(
                enabled=True,
                profile_mode=PrefixKVProfileMode.OFFLINE_PROFILE,
                profile_path=str(profile),
                start_size=1,
                protect_size=1,
                eviction_distance=-25,
            ),
            **common,
        )
    elif family == "vl_cache":
        method = replace(
            prompt,
            method=MosaicKVMethod.VL_CACHE_REIMPL,
            cache=cache,
            vl_cache=VLCacheConfig(
                enabled=True,
                sparsity_threshold=0.01,
                min_layer_retention=0.01,
                max_layer_retention=1.0,
                recent_window_fraction=0.1,
            ),
            **common,
        )
    else:
        raise ValueError(f"unsupported fidelity family: {family}")
    if prompt.model != method.model or prompt.dataset != method.dataset:
        raise RuntimeError("paired baseline configurations changed model or dataset")
    if prompt.execution != method.execution or prompt.generation != method.generation:
        raise RuntimeError("paired baseline configurations changed decoding controls")
    if prompt.cache != method.cache:
        raise RuntimeError("paired baseline configurations changed the cache contract")
    return {"full_kv": full, "prompt_attention_topk": prompt, FAMILY_METHOD[family]: method}


def byte_capped_configs(
    template: Any,
    *,
    repository: Path,
    family: str,
    task: str,
    model_id: str,
    model_revision: str,
    seed: int,
    output_length: int,
    byte_cap: int,
    profile_forget_ratio: str | None = None,
    profile_reference_retention_ratio: float | None = None,
) -> dict[str, Any]:
    """Build a physical-byte-matched FullKV/Top-K/baseline/MosaicKV set.

    The profile ratio is only the source shape for PrefixKV.  It is never a
    comparison budget: all compressed methods receive the same hard byte cap
    and block/packing contract.
    """

    if byte_cap < 1:
        raise ValueError("byte_cap must be positive")
    controls = review_configs(
        template,
        repository=repository,
        family=family,
        task=task,
        model_id=model_id,
        model_revision=model_revision,
        seed=seed,
        output_length=output_length,
        retention_ratio=0.99,
        profile_forget_ratio=profile_forget_ratio,
    )
    baseline_name = FAMILY_METHOD[family]
    byte_cache = replace(
        controls["prompt_attention_topk"].cache,
        budget_value=byte_cap,
        budget_unit=BudgetUnit.BYTES,
        retention_ratio=0.99,
        block_size=1,
        physical_budget_block_size=1,
        physical_budget_mandatory_policy="last_token",
        exact_gather_mode="source_views",
        packing_mode="per_head_reference",
        total_storage_budget_bytes=None,
    )
    prompt = replace(controls["prompt_attention_topk"], cache=byte_cache)
    baseline = replace(controls[baseline_name], cache=byte_cache)
    if family == "prefixkv":
        if profile_reference_retention_ratio is None:
            raise ValueError("byte-capped PrefixKV requires a profile reference retention ratio")
        baseline = replace(
            baseline,
            prefixkv=replace(
                baseline.prefixkv,
                rescale_offline_profile_to_byte_budget=True,
                profile_reference_retention_ratio=profile_reference_retention_ratio,
            ),
        )
    mosaic = replace(
        template,
        model=prompt.model,
        dataset=prompt.dataset,
        execution=prompt.execution,
        generation=prompt.generation,
        cache=byte_cache,
        method=MosaicKVMethod.MOSAICKV_EXACT,
        forecasting=ForecastingConfig(
            enabled=True,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
        ),
        graph=GraphConfig(enabled=True),
        selection=SelectionConfig(enabled=True),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
        lookm=LookMConfig(enabled=False),
        prefixkv=PrefixKVConfig(enabled=False),
        vl_cache=VLCacheConfig(enabled=False),
    )
    for config in (prompt, baseline, mosaic):
        if config.cache != byte_cache:
            raise RuntimeError("byte-matched configurations diverged in cache policy")
        if config.model != prompt.model or config.dataset != prompt.dataset:
            raise RuntimeError("byte-matched configurations diverged in model or dataset")
        if config.execution != prompt.execution or config.generation != prompt.generation:
            raise RuntimeError("byte-matched configurations diverged in execution controls")
    return {
        "full_kv": controls["full_kv"],
        "prompt_attention_topk": prompt,
        baseline_name: baseline,
        "mosaickv_exact": mosaic,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--job-spec-sha256", required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    job_path = args.job_spec.resolve()
    if _sha(job_path) != args.job_spec_sha256:
        raise RuntimeError("job specification hash changed")
    job = _load_json(job_path)
    source_sha, dirty = _snapshot(repository)
    if dirty or source_sha != job["source_git_sha"]:
        raise RuntimeError("baseline-fidelity worker requires its clean immutable source SHA")
    if job.get("classification") != CLASSIFICATION:
        raise RuntimeError("worker received the wrong classification")
    protocol_path = Path(str(job["protocol_path"])).resolve()
    if _sha(protocol_path) != job["protocol_sha256"]:
        raise RuntimeError("protocol hash changed")
    ledger_path = Path(str(job["sample_ledger_path"])).resolve()
    if _sha(ledger_path) != job["sample_ledger_sha256"]:
        raise RuntimeError("sample ledger hash changed")
    ledger = _load_json(ledger_path)
    sample_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != int(job["sample_count"]) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("sample identities changed")

    family = str(job["family"])
    task = str(job["task"])
    method_name = FAMILY_METHOD[family]
    methods = tuple(str(value) for value in cast("list[str]", job["methods"]))
    if methods != ("full_kv", "prompt_attention_topk", method_name):
        raise RuntimeError("method order changed")
    output_root = Path(str(job["output_root"])).resolve()
    shard_root = output_root / family / task / f"{int(job['shard_index']):02d}"
    completion = shard_root / "job_complete.json"
    if completion.exists():
        previous = _load_json(completion)
        if (
            previous.get("status") == "completed"
            and previous.get("job_spec_sha256") == args.job_spec_sha256
        ):
            return 0
        raise RuntimeError("existing completion marker has a different identity")
    shard_root.mkdir(parents=True, exist_ok=True)

    protocol = _load_json(protocol_path) if protocol_path.suffix == ".json" else None
    if protocol is None:
        import yaml

        protocol = cast("dict[str, Any]", yaml.safe_load(protocol_path.read_text()))
    family_contract = cast("dict[str, Any]", cast("dict[str, Any]", protocol["families"])[family])
    profile_map = cast(
        "dict[str, str]", family_contract.get("profile_forget_ratio_by_retention", {})
    )
    template = load_config(repository / "mosaickv/configs/paper_candidate_llava_v1.yaml")
    before = asdict(capture_gpu_environment(torch))
    adapter = load_hf_adapter(
        str(job["model_id"]),
        revision=str(job["model_revision"]),
        model_kwargs=_model_kwargs(
            replace(
                template,
                model=ModelConfig(str(job["model_id"]), str(job["model_revision"]), Precision.FP16),
            ),
            local_files_only=True,
        ),
        processor_kwargs={"local_files_only": True},
    )
    records: list[dict[str, Any]] = []
    byte_checks: list[dict[str, Any]] = []
    namespace = f"baseline-fidelity-{source_sha[:10]}-{family}-{task}-{int(job['shard_index']):02d}"
    try:
        full_configs = review_configs(
            template,
            repository=repository,
            family=family,
            task=task,
            model_id=str(job["model_id"]),
            model_revision=str(job["model_revision"]),
            seed=int(job["selection_seed"]),
            output_length=int(job["output_length"]),
            retention_ratio=float(cast("list[float]", job["retention_ratios"])[0]),
            profile_forget_ratio=profile_map.get(
                _retention_key(float(cast("list[float]", job["retention_ratios"])[0]))
            ),
        )
        records.append(
            _record(
                full_configs["full_kv"],
                adapter,
                task=task,
                run_id=f"{namespace}-full-kv",
                output_root=shard_root,
                ledger_path=ledger_path,
                ledger_sha=str(job["sample_ledger_sha256"]),
                sample_ids=sample_ids,
                phase=CLASSIFICATION,
                repetition=0,
                order=0,
            )
        )
        _settle(adapter)
        for ratio_index, raw_ratio in enumerate(cast("list[float]", job["retention_ratios"])):
            ratio = float(raw_ratio)
            configs = review_configs(
                template,
                repository=repository,
                family=family,
                task=task,
                model_id=str(job["model_id"]),
                model_revision=str(job["model_revision"]),
                seed=int(job["selection_seed"]),
                output_length=int(job["output_length"]),
                retention_ratio=ratio,
                profile_forget_ratio=profile_map.get(_retention_key(ratio)),
            )
            paired: dict[str, dict[str, dict[str, Any]]] = {}
            order: tuple[str, ...] = ("prompt_attention_topk", method_name)
            if ratio_index % 2:
                order = tuple(reversed(order))
            for method in order:
                record = _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"{namespace}-r{ratio:.2f}-{method}",
                    output_root=shard_root,
                    ledger_path=ledger_path,
                    ledger_sha=str(job["sample_ledger_sha256"]),
                    sample_ids=sample_ids,
                    phase=CLASSIFICATION,
                    repetition=0,
                    order=len(records),
                )
                record["retention_ratio"] = ratio
                record["profile_forget_ratio"] = profile_map.get(_retention_key(ratio))
                records.append(record)
                paired[method] = {
                    str(row["sample_id"]): row
                    for row in cast("list[dict[str, Any]]", record["rows"])
                }
                _settle(adapter)
            for sample_id in sample_ids:
                prompt_bytes = int(paired["prompt_attention_topk"][sample_id]["active_kv_bytes"])
                method_bytes = int(paired[method_name][sample_id]["active_kv_bytes"])
                byte_checks.append(
                    {
                        "sample_id": sample_id,
                        "retention_ratio": ratio,
                        "prompt_active_kv_bytes": prompt_bytes,
                        "method_active_kv_bytes": method_bytes,
                        "difference_bytes": method_bytes - prompt_bytes,
                        "exact_byte_match": method_bytes == prompt_bytes,
                    }
                )
    finally:
        after = asdict(capture_gpu_environment(torch))
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if any(not row["exact_byte_match"] for row in byte_checks):
        raise RuntimeError("paired method and Prompt Top-K did not realize identical active bytes")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "classification": CLASSIFICATION,
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "job_spec": str(job_path),
        "job_spec_sha256": args.job_spec_sha256,
        "protocol_sha256": job["protocol_sha256"],
        "sample_ledger_sha256": job["sample_ledger_sha256"],
        "family": family,
        "task": task,
        "model_id": job["model_id"],
        "model_revision": job["model_revision"],
        "records": records,
        "byte_match_checks": byte_checks,
        "before_model_load": before,
        "after_measurement": after,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    }
    with completion.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
