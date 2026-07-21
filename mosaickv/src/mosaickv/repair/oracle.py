"""Evaluation-only oracle trigger and recovery labeling for residual repair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mosaickv.config import RepairConfig
from mosaickv.repair.core import ReDecodeCallback, _repair_decode_step
from mosaickv.repair.types import RepairCacheState, RepairStepResult
from mosaickv.types import RepairPolicy


def evaluate_repair_decode_step(
    state: RepairCacheState,
    config: RepairConfig,
    *,
    step_index: int,
    provisional_logits: Any,
    prototype_attention_mass: Mapping[int, float],
    re_decode: ReDecodeCallback,
    reference_token_id: int,
    oracle_should_repair: bool | None = None,
    draft_distribution: Any | None = None,
) -> RepairStepResult:
    """Run repair with evaluation-only reference labels and optional oracle trigger."""

    if not config.evaluation_only:
        raise ValueError("repair evaluation API requires repair.evaluation_only=true")
    if config.policy is RepairPolicy.ORACLE and oracle_should_repair is None:
        raise ValueError("oracle policy requires oracle_should_repair")
    if config.policy is not RepairPolicy.ORACLE and oracle_should_repair is not None:
        raise ValueError("oracle_should_repair is only valid for the oracle policy")
    return _repair_decode_step(
        state,
        config,
        step_index=step_index,
        provisional_logits=provisional_logits,
        prototype_attention_mass=prototype_attention_mass,
        re_decode=re_decode,
        draft_distribution=draft_distribution,
        oracle_should_repair=oracle_should_repair,
        reference_token_id=reference_token_id,
    )


__all__ = ["evaluate_repair_decode_step"]
