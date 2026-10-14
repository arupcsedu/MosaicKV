"""Explicit factorial ablations; no dataset selection, submission, or tuning.

Comparisons are grouped by mandatory policy. A group cannot reuse controls
from another group, even when their byte caps are equal.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from mosaickv.baselines.prefixkv import PrefixKVOfflineProfile
from mosaickv.config import PrefixKVConfig, RunConfig
from mosaickv.types import MosaicKVMethod


@dataclass(frozen=True, slots=True)
class EvidenceAblation:
    name: str
    sink_neutral: bool
    question_protected: bool
    question_weight: float

    @property
    def profile_policy(self) -> str:
        return "question_and_boundaries" if self.question_protected else "boundaries"

    @property
    def cache_policy(self) -> str:
        return "question_and_boundaries" if self.question_protected else "first_and_last"


# Alpha is a prospective fixed hypothesis, not fitted from prior evaluation labels.
# Separate C, AC, BC, and ABC rows distinguish ranking from reservation effects.
EVIDENCE_ABLATIONS = (
    EvidenceAblation("control", False, False, 0.0),
    EvidenceAblation("a_sink", True, False, 0.0),
    EvidenceAblation("b_protect", False, True, 0.0),
    EvidenceAblation("c_rank", False, False, 0.5),
    EvidenceAblation("ab_sink_protect", True, True, 0.0),
    EvidenceAblation("ac_sink_rank", True, False, 0.5),
    EvidenceAblation("bc_protect_rank", False, True, 0.5),
    EvidenceAblation("abc_combined", True, True, 0.5),
)


def evidence_ablation_configs(
    controls: Mapping[str, RunConfig],
    profile_paths: Mapping[str, Path],
) -> dict[str, dict[str, RunConfig]]:
    """Require matched controls and frozen profiles for every preregistered arm."""
    expected = {"full_kv", "prompt_attention_topk", "mosaickv_exact"}
    if set(controls) != expected:
        raise ValueError("require FullKV, Prompt Top-K and MosaicKV exact controls")
    template = controls["prompt_attention_topk"]
    for name, cfg in controls.items():
        if cfg.method.value != name:
            raise ValueError("mislabeled control method")
        for key in ("model", "dataset", "execution", "generation"):
            if getattr(cfg, key) != getattr(template, key):
                raise ValueError(f"control {key} differs")
        if name != "full_kv" and cfg.cache != template.cache:
            raise ValueError("compressed control byte/block policies differ")
    if set(profile_paths) != {a.name for a in EVIDENCE_ABLATIONS}:
        raise ValueError("require separate frozen profiles for all eight ablations")
    groups: dict[str, dict[str, RunConfig]] = {}
    for policy in ("first_and_last", "question_and_boundaries"):
        groups[policy] = {
            name: replace(cfg, cache=replace(cfg.cache, physical_budget_mandatory_policy=policy))
            for name, cfg in controls.items()
        }
    calibration_identity = None
    evidence_hash = None
    for arm in EVIDENCE_ABLATIONS:
        path = profile_paths[arm.name]
        profile = PrefixKVOfflineProfile.from_json_object(json.loads(path.read_text()))
        identity = (
            profile.calibration_dataset_id,
            profile.calibration_dataset_revision,
            profile.calibration_split,
            profile.calibration_sample_ids,
            profile.calibration_seed,
            profile.calibration_bytes_per_layer_position,
        )
        if calibration_identity is not None and identity != calibration_identity:
            raise ValueError("ablation profiles use different calibration identities")
        calibration_identity = identity
        if profile.evidence_variant:
            if evidence_hash is not None and evidence_hash != profile.calibration_evidence_sha256:
                raise ValueError("ablation profiles use different captured evidence")
            evidence_hash = profile.calibration_evidence_sha256
        if (
            profile.calibration_byte_cap != template.cache.budget_value
            or profile.model_id != template.model.id
            or profile.model_revision != template.model.revision
            or profile.prompt_reservation_policy != arm.profile_policy
            or profile.question_score_weight != arm.question_weight
            or profile.search_policy
            != ("optional_cdf" if arm.sink_neutral else "upstream_protected_cdf")
            or profile.start_size != 1
            or profile.protect_size != 1
        ):
            raise ValueError(f"frozen profile does not match ablation {arm.name}")
        config = replace(
            groups[arm.cache_policy]["prompt_attention_topk"],
            method=MosaicKVMethod.PREFIXKV_REIMPL,
            prefixkv=PrefixKVConfig(
                enabled=True,
                profile_path=str(path),
                rescale_offline_profile_to_byte_budget=True,
                profile_reference_retention_ratio=profile.target_retention_ratio,
                hf_cache_transport="ragged_llama_eager_v1",
                decode_boundary_policy="protected",
                decode_policy="effective_profile_ratio",
                exclude_protected_from_calibration=arm.sink_neutral,
                question_score_weight=arm.question_weight,
            ),
        )
        groups[arm.cache_policy][arm.name] = config
    return groups
