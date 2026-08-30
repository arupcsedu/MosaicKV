#!/usr/bin/env python3
"""Create immutable task-routed HF selector-planning configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import (
    ForecastingConfig,
    GraphConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    canonical_config_json,
    load_config,
)
from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_MODEL_ID,
    PAPER_CANDIDATE_TASK_ROUTES,
)
from mosaickv.types import ForecastMode, MosaicKVMethod, RepairPolicy

CLASSIFICATION = "fixed_preregistered_paper_quality_memory"
CONTINUATION_CLASSIFICATION = "fixed_preregistered_paper_quality_memory_continuation"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planning_configs(
    template: Any,
    *,
    task: str,
    seed: int,
    planner_output_length: int,
) -> dict[str, Any]:
    """Return plan-only Prompt Top-K and exact MosaicKV configurations."""

    if task not in PAPER_CANDIDATE_TASK_ROUTES:
        raise ValueError(f"unsupported paper task: {task!r}")
    if template.model.id != PAPER_CANDIDATE_MODEL_ID:
        raise ValueError("paper configuration requires the pinned LLaVA-1.5-7B model")
    if template.method is not MosaicKVMethod.MOSAICKV_FULL:
        raise ValueError("candidate template must be mosaickv_full")
    if planner_output_length < 1:
        raise ValueError("planner_output_length must be positive")
    route = PAPER_CANDIDATE_TASK_ROUTES[task]
    common = {
        "dataset": route,
        "execution": replace(template.execution, seed=seed),
        # Selector construction occurs at prefill. One measured token avoids
        # spending paper GPUs on an unused HF decode while the primary vLLM
        # arms retain the protocol's common 64-token output length.
        "generation": replace(
            template.generation,
            max_new_tokens=planner_output_length,
        ),
    }
    prompt = replace(
        template,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            total_storage_budget_bytes=None,
        ),
        forecasting=ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
        ),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
        **common,
    )
    exact = replace(
        template,
        method=MosaicKVMethod.MOSAICKV_EXACT,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            total_storage_budget_bytes=None,
        ),
        prototypes=replace(template.prototypes, enabled=False),
        residual=replace(template.residual, enabled=False),
        repair=replace(
            template.repair,
            enabled=False,
            evaluation_only=False,
            policy=RepairPolicy.NONE,
            counterfactual_draft_steps=0,
            counterfactual_route_heads=False,
            draft_kl_threshold=None,
            draft_require_token_mismatch=False,
            max_blocks_per_step=0,
        ),
        **common,
    )
    return {"prompt_attention_topk": prompt, "mosaickv_exact": exact}


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable planning artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_state(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--planner-output-length", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    source_sha, dirty = _source_state(repository)
    if dirty:
        raise RuntimeError("paper planning requires a clean worktree")
    protocol_path = args.protocol.resolve()
    template_path = args.template.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a YAML object")
    if protocol.get("classification") not in {
        CLASSIFICATION,
        CONTINUATION_CLASSIFICATION,
    }:
        raise ValueError("wrong paper protocol classification")
    if protocol.get("execution_authorized") is not True:
        raise PermissionError("paper quality-memory execution is not authorized")
    candidate = cast("dict[str, Any]", protocol.get("candidate", {}))
    if candidate.get("config_sha256") != _sha(template_path):
        raise ValueError("candidate template SHA-256 differs from the protocol")
    experiment = cast("dict[str, Any]", protocol.get("experiment", {}))
    if args.task not in cast("list[str]", experiment.get("tasks", [])):
        raise ValueError("task is absent from the paper protocol")
    if args.seed != experiment.get("selection_seed"):
        raise ValueError("selection seed differs from the paper protocol")

    template = load_config(template_path)
    configs = planning_configs(
        template,
        task=args.task,
        seed=args.seed,
        planner_output_length=args.planner_output_length,
    )
    output_root = args.output_root.resolve()
    records: dict[str, Any] = {}
    for method, config in configs.items():
        encoded = yaml.safe_dump(json.loads(canonical_config_json(config)), sort_keys=True).encode(
            "utf-8"
        )
        path = output_root / f"{method}.yaml"
        _write_once(path, encoded)
        records[method] = {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}
    descriptor = {
        "schema_version": 1,
        "classification": protocol["classification"],
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "candidate_template": str(template_path),
        "candidate_template_sha256": _sha(template_path),
        "task": args.task,
        "selection_seed": args.seed,
        "planner_output_length": args.planner_output_length,
        "primary_generation_output_length": int(experiment["output_length"]),
        "configs": records,
    }
    descriptor_path = output_root / "descriptor.json"
    encoded_descriptor = (json.dumps(descriptor, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_once(descriptor_path, encoded_descriptor)
    print(json.dumps(descriptor, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
