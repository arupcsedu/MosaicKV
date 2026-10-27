"""V9: shared-anchor sensitivity-ranked exact substitutions, consumed inputs only."""

from __future__ import annotations

import time
from typing import Any

from mosaickv.backends.hf_downstream_sensitivity import DownstreamCapture
from mosaickv.backends.hf_prefix_outputs import frozen_query_prediction
from mosaickv.backends.hf_runtime import PrefillDiagnosticComplete
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.draft_head_diagnostic import DraftConditionedHeadDiagnostic
from mosaickv.evaluation.readout_audit import ReadoutAuditPolicy, rank_correlation
from mosaickv.selection.draft_head_substitution import draft_substitution_pool


def downstream_screen(rows):
    policy, guard = ReadoutAuditPolicy(), DraftAllocationGuardConfig()
    if len(rows) < 2 or any(r["prefix"] != 1 for r in rows):
        return dict(passed=False, reason="insufficient_common_first_prefix_observations")
    actual = [r["actual_KL_gain"] for r in rows]
    rho = rank_correlation([r["predicted_gain"] for r in rows], actual)
    reconstruction = rank_correlation([r["reconstruction_gain"] for r in rows], actual)
    positive = [r for r in rows if r["predicted_gain"] > 0]
    precision = (
        sum(
            r["actual_KL_gain"]
            > guard.absolute_tolerance + guard.relative_tolerance * abs(r["anchor_KL"])
            for r in positive
        )
        / len(positive)
        if positive
        else None
    )
    passed = (
        rho is not None
        and reconstruction is not None
        and rho >= policy.minimum_rank_correlation
        and rho >= reconstruction + policy.minimum_rank_gain_over_reconstruction
        and len(positive) >= policy.minimum_positive_predictions
        and precision is not None
        and precision >= policy.minimum_positive_precision
    )
    return dict(
        passed=bool(passed),
        rank_correlation=rho,
        reconstruction_rank_correlation=reconstruction,
        positive_predictions=len(positive),
        positive_precision=precision,
        descriptive_only=True,
        benchmark_quality_validated=False,
    )


class DownstreamHeadDiagnostic(DraftConditionedHeadDiagnostic):
    def __init__(self, *args: Any, jvp_proposal: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.decode_observer = DownstreamCapture(self.adapter)
        self.jvp_proposal = jvp_proposal
        self.downstream_record: dict[str, Any] = {}
        self.saved_sensitivities: dict[str, Any] = {}

    def _construct_pool(self, full_state, anchor):
        import torch

        observer = self.decode_observer
        p = self.jvp_proposal
        predicted = frozen_query_prediction(
            full_state, anchor, p, self.saved_evidence, observer.layers[p.layer].self_attn.o_proj
        )
        direction = torch.zeros(
            (len(observer.layers), predicted["predicted_projected_change_fp64"].shape[-1])
        )
        direction[p.layer] = predicted["predicted_projected_change_fp64"][0]
        try:
            gradients, record = observer.gradients(
                self.saved_logits["fullkv"], self.saved_logits["anchor"], jvp_direction=direction
            )
        except Exception as error:
            self.downstream_record = dict(
                observer.progress,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                snapshot_KV_bytes=observer.storage_bytes,
                seconds=time.perf_counter()
                - observer.progress.get("started_monotonic", time.perf_counter()),
            )
            raise
        self.saved_sensitivities = dict(gradients=gradients)
        head_gradients = []
        for i, (layer, data, source) in enumerate(
            zip(observer.layers, self.saved_evidence, full_state.layers, strict=True)
        ):
            weight = layer.self_attn.o_proj.weight.detach()
            query_count = data["target"].shape[1]
            heads, dimension = source.kv_heads, source.value.shape[-1]
            groups = layer.self_attn.q_proj.out_features // (heads * dimension)
            if query_count != 32 * groups:
                raise ValueError("downstream pool requires the unchanged 32-prefix evidence")
            out = torch.zeros_like(data["target"], device=weight.device)
            for step in (1, 4):
                g = gradients[step][i].to(weight.device, torch.float32) @ weight.float()
                # The original pool averages both Q and D. Correct both factors
                # to obtain the sum over D, averaged over exactly two prefixes.
                out[:, (step - 1) * groups : step * groups] = g.reshape(
                    heads, groups, dimension
                ) * (query_count * dimension / 2)
            head_gradients.append(out)
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)
        start = time.perf_counter()
        pool, trace = draft_substitution_pool(
            full_state,
            anchor,
            self.saved_evidence,
            limit=self.policy.pool_size,
            policy=self.pool_policy,
            output_gradients=tuple(head_gradients),
        )
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)
        record.update(
            pool_seconds=time.perf_counter() - start,
            capture_seconds=observer.capture_seconds,
            source_contract=observer.recorder.source_contract,
            fixed_derivative_prefixes=[1, 4],
            original_weights_frozen=True,
            candidate_forward_calls_for_ranking=0,
        )
        self.downstream_record = record
        self.saved_sensitivities["output_gradients"] = tuple(
            x.detach().cpu() for x in head_gradients
        )
        return pool, trace

    def _observe_exact_anchor(self, *args):
        try:
            super()._observe_exact_anchor(*args)
        except PrefillDiagnosticComplete as complete:
            record = complete.record
            trace = {(r["layer"], r["kv_head"]): r for r in record["pool_trace"]}
            observations = []
            for event in record["selection"]["events"]:
                raw = record["proposals"][event["proposal_id"]]
                surrogate = trace[raw["layer"], raw["head"]]
                first = event["steps"][0]
                observations.append(
                    dict(
                        proposal_id=event["proposal_id"],
                        prefix=1,
                        predicted_gain=surrogate["predicted_gain"],
                        reconstruction_gain=surrogate["draft_output_error_before"]
                        - surrogate["draft_output_error_after"],
                        actual_KL_gain=first["anchor"]["logit_kl"] - first["candidate"]["logit_kl"],
                        anchor_KL=first["anchor"]["logit_kl"],
                    )
                )
            screen = downstream_screen(observations)
            record.update(
                classification="V9_consumed_input_downstream_sensitivity_not_benchmark_quality",
                downstream_sensitivity=self.downstream_record,
                ranking_observations=observations,
                ranking_screen=screen,
                proposer="one_head_equal_capacity_shared_anchor_KL_VJP",
                original_guard_unchanged=True,
                diagnostic_passed=bool(record["diagnostic_passed"] and screen["passed"]),
                negative_action="stop_substitution_strategy_no_automatic_successor",
                gpu_campaign_authorized=False,
            )
            raise PrefillDiagnosticComplete(record) from None
