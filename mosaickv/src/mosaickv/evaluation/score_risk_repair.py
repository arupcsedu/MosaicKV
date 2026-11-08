"""Actual prototype-risk repair before PV/token commitment, no parent oracle.

The mutable shared lease still contains current K. A trigger promotes the real
pinned residuals, updates K there, and replays the same full-head score operator
once. The subsequent ordinary V fill reads the promoted values. No full K clone,
extra vocabulary readout, second prefill, or generated-history update is made.
This is NOT the old live-parent guarded policy; offline quality must be tested.
"""

import math
from contextlib import ExitStack
from unittest.mock import patch


def should_repair(*, mass, dispersion, threshold, repaired, disabled, force):
    if (
        any(not math.isfinite(v) for v in (mass, dispersion, threshold))
        or not 0 <= mass <= 1.001
        or dispersion < 0
        or threshold <= 0
        or any(type(b) is not bool for b in (repaired, disabled, force))
    ):
        raise ValueError("finite measured prototype risk and explicit policy required")
    return not repaired and not disabled and (force or mass * dispersion >= threshold)


class ScoreRiskRepair:
    def __init__(self, overlay, policy, report):
        if policy.repair_execution != "score_risk" or policy.layers != (31,):
            raise ValueError("registered isolated score-risk policy required")
        self.overlay, self.policy, self.report = overlay, policy, report
        self.stack, self.steps = ExitStack(), 0
        self.maximum_mass = self.maximum_risk = 0.0
        self.events = []

    def apply(self, binding, query, key, mask, scaling, scores):
        self.steps += 1
        overlay = self.overlay
        if overlay.repair_count:
            # Once promoted there is no approximate prototype left to assess.
            # Avoid a pointless device-to-host risk synchronization per token.
            return scores
        mass = float(scores[0, overlay.head, 0].index_select(0, overlay.positions).float().sum())
        risk = mass * overlay.dispersion
        self.maximum_mass, self.maximum_risk = (
            max(self.maximum_mass, mass),
            max(self.maximum_risk, risk),
        )
        disabled = self.policy.component_mode in ("no_repair", "exact_members", "parent_reserved")
        trigger = should_repair(
            mass=mass,
            dispersion=overlay.dispersion,
            threshold=self.policy.prototype_risk_threshold,
            repaired=bool(overlay.repair_count),
            disabled=disabled,
            force=bool(self.policy.force_repair or self.policy.component_mode == "forced_repair"),
        )
        if not trigger:
            return scores
        if query.shape != (1, 32, 1, 128) or key.ndim != 4 or key.shape[:2] != (1, 32):
            raise ValueError("original full-head single-token BF16 lease required")
        event = overlay.promote()
        key[0, overlay.head].index_copy_(0, overlay.positions, overlay.promotion[0])
        calls = binding.calls
        # Reuse the very same original score operator/shape. Do not increment
        # Binding.calls: doing so corrupts layer ordering and scheduler metadata.
        revised = binding.core.replay_pending_scores(query, mask, key.shape[-2])
        if binding.calls != calls:
            raise RuntimeError("repair changed the committed attention-update count")
        detail = dict(
            event,
            trigger_step=self.steps,
            trigger_reason="controlled_score_risk"
            if self.policy.force_repair
            else "prototype_risk",
            stress_control=bool(self.policy.force_repair),
            prototype_cluster_anchor=overlay.anchor,
            restored_modality="image",
            prototype_attention_mass=mass,
            prototype_attention_mass_times_dispersion=risk,
            original_logical_positions_preserved=True,
            attention_scores_recomputed_once=True,
            whole_model_redecode_performed=False,
            extra_KV_snapshot_bytes=0,
            provisional_vocabulary_logits_computed=False,
            logit_change=None,
            token_change=None,
            benchmark_quality_recovered=None,
            unmeasured_vocabulary_changes_not_fabricated=True,
        )
        self.events.append(detail)
        self.report["repair_events"].append(detail)
        return revised

    def __enter__(self):
        from mosaickv.evaluation.uniform_core_transport import Binding

        scores = Binding.scores
        owner = self

        def observe(binding, query, key, mask, scaling):
            result = scores(binding, query, key, mask, scaling)
            if (binding.calls - 1) % 32 == 31:
                result = owner.apply(binding, query, key, mask, scaling, result)
            return result

        self.stack.enter_context(patch.object(Binding, "scores", observe))
        return self

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.report["score_risk_repair"] = dict(
                enabled=True,
                committed_updates_observed=self.steps,
                actual_repair_events=len(self.events),
                maximum_prototype_mass=self.maximum_mass,
                maximum_prototype_risk=self.maximum_risk,
                risk_statistics_scope="before_first_promotion_only",
                extra_parent_KV_snapshot_bytes=0,
                parent_counterfactual_readouts=0,
                fullkv_or_label_or_oracle_access=False,
                separate_decode_probe=False,
                online_parent_token_guarantee=False,
                offline_unchanged_fidelity_gate_required=True,
                all_six_mechanisms_active=True,
                exact_whole_request_peak_certified=False,
            )
