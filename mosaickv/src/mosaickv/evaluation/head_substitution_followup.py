"""V7.1 null controls and full-horizon replay of immutable V7 survivors."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from mosaickv.backends.hf_allocation_guard import (
    decide_draft_stability,
    draft_row_causes,
    packed_positions_sha,
)
from mosaickv.backends.hf_head_substitution import HeadProbeSession
from mosaickv.backends.hf_runtime import PrefillDiagnosticComplete
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.head_substitution_diagnostic import HeadSubstitutionDiagnostic
from mosaickv.selection.draft_shortlist import meaningful_improvement
from mosaickv.selection.head_substitution import HeadSubstitution, pack_substitution


def clone_anchor(anchor: Any) -> Any:
    return replace(
        anchor,
        snapshot=replace(
            anchor.snapshot,
            layers=tuple(
                replace(layer, key=layer.key.clone(), value=layer.value.clone())
                for layer in anchor.snapshot.layers
            ),
        ),
    )


def repack_anchor(full_state: Any, anchor: Any) -> Any:
    """Gather exactly the original slot order, without a substitution or re-sort."""
    import torch

    physical = {p: i for i, p in enumerate(full_state.logical_positions.physical_to_logical)}
    layers = []
    for i, (source, old) in enumerate(zip(full_state.layers, anchor.snapshot.layers, strict=True)):
        heads: list[list[tuple[int, int]]] = [[] for _ in range(source.kv_heads)]
        for row in anchor.slot_records:
            if row["layer"] == i:
                heads[row["kv_head"]].append((row["slot"], physical[row["logical_position"]]))
        positions = [[p for _, p in sorted(row)] for row in heads]
        if any(len(row) != old.key.shape[2] for row in positions):
            raise ValueError("null repack must preserve every physical head slot")
        indices = torch.tensor(positions, device=source.key.device).unsqueeze(0).unsqueeze(-1)
        key = source.key.gather(
            2, indices.expand(1, source.kv_heads, old.key.shape[2], source.key.shape[-1])
        )
        values = indices.to(source.value.device).expand(
            1, source.kv_heads, old.value.shape[2], source.value.shape[-1]
        )
        layers.append(replace(old, key=key, value=source.value.gather(2, values)))
    return replace(anchor, snapshot=replace(anchor.snapshot, layers=tuple(layers)))


def audit_null_control(
    rows: list[dict[str, Any]], policy: DraftAllocationGuardConfig
) -> dict[str, Any]:
    decision = decide_draft_stability(rows, policy)
    exact = all(
        row["candidate"]["bitwise_equal_to_anchor_logits"]
        and row["candidate"]["anchor_candidate_maximum_logit_difference"] == 0
        and row["candidate"]["anchor_candidate_kl_fp64"] == 0
        and all(
            row["anchor"][key] == row["candidate"][key]
            for key in (
                "logit_kl",
                "maximum_logit_difference",
                "greedy_token_agreement",
                "active_bytes",
            )
        )
        for row in rows
    )
    return dict(
        passed=bool(exact and decision["accepted"]),
        bitwise_anchor_logit_identity=bool(exact),
        registered_guard=decision,
    )


class HeadSubstitutionFollowup(HeadSubstitutionDiagnostic):
    def __init__(self, *args: Any, followup: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if set(followup) != {"anchor_positions_sha256", "survivors"} or len(
            followup["survivors"]
        ) not in (0, 3):
            raise ValueError("follow-up accepts only the frozen zero/three survivor descriptors")
        self.followup = followup
        self.saved_logits: dict[str, dict[int, Any]] = {}

    def _observe_exact_anchor(
        self, prefill: Any, full_state: Any, anchor: Any, attention: Any, budget: Any
    ) -> None:
        import torch

        if packed_positions_sha(anchor) != self.followup["anchor_positions_sha256"]:
            raise ValueError("follow-up did not reconstruct the immutable V7 anchor")

        def observe(role: str, step: int, logits: Any) -> None:
            self.saved_logits.setdefault(role, {})[step] = logits.detach().cpu().clone()

        session = HeadProbeSession(
            self.adapter,
            prefill,
            anchor,
            budget,
            self.policy.guard,
            sink_tokens=self.config.bounded_decode.sink_tokens,
            capture_fp64=True,
            logits_observer=observe,
        )
        session.sync()
        started = time.perf_counter()
        controls: dict[str, Any] = {}
        extensions: list[dict[str, Any]] = []
        controls_passed = False
        try:
            for name, packed in (
                ("clone", clone_anchor(anchor)),
                ("repack", repack_anchor(full_state, anchor)),
            ):
                tensors_equal = all(
                    torch.equal(a.key, b.key) and torch.equal(a.value, b.value)
                    for a, b in zip(anchor.snapshot.layers, packed.snapshot.layers, strict=True)
                )
                if not tensors_equal or packed_positions_sha(packed) != packed_positions_sha(
                    anchor
                ):
                    raise ValueError(
                        "null control changed original K/V values or logical positions"
                    )
                # Controls collect the entire horizon even on an identity failure;
                # they cannot be accepted or used to relax the registered guard.
                stream = session.probe(packed, label=f"null_{name}")
                try:
                    rows = list(stream)
                finally:
                    stream.close()
                controls[name] = dict(
                    steps=rows,
                    tensors_bitwise_equal=True,
                    **audit_null_control(rows, self.policy.guard),
                )
            controls_passed = all(value["passed"] for value in controls.values())
            if controls_passed:
                for spec in self.followup["survivors"]:
                    proposal = HeadSubstitution(**spec["proposal"])
                    packed = pack_substitution(full_state, anchor, proposal)
                    if packed_positions_sha(packed) != spec["positions_sha256"]:
                        raise ValueError("replayed survivor differs from the frozen V7 plan")
                    stream = session.probe(packed, label=spec["proposal_id"])
                    rows = []
                    try:
                        for row in stream:
                            rows.append(row)
                            if draft_row_causes(row, self.policy.guard):
                                break
                    finally:
                        stream.close()
                    decision = decide_draft_stability(
                        rows, self.policy.guard, allow_early_rejection=True
                    )
                    keys = (
                        "logit_kl",
                        "maximum_logit_difference",
                        "greedy_token_agreement",
                        "active_bytes",
                    )
                    replayed = len(rows) >= 4 and all(
                        row[who][key] == old[who][key]
                        for row, old in zip(rows[:4], spec["short_steps"], strict=True)
                        for who in ("anchor", "candidate")
                        for key in keys
                    )
                    extensions.append(
                        dict(
                            proposal_id=spec["proposal_id"],
                            proposal=spec["proposal"],
                            steps=rows,
                            registered_guard=decision,
                            original_four_prefixes_replayed=bool(replayed),
                            accepted=bool(decision["accepted"] and replayed),
                            meaningful=bool(
                                decision["accepted"]
                                and replayed
                                and meaningful_improvement(rows, self.policy.guard)
                            ),
                        )
                    )
        finally:
            storage = session.finish()
        session.sync()
        all_failed = (
            bool(extensions)
            and len(extensions) == len(self.followup["survivors"])
            and all(
                not event["registered_guard"]["accepted"]
                and event["original_four_prefixes_replayed"]
                for event in extensions
            )
        )
        record = dict(
            classification="retrospective_V7_1_consumed_input_numerical_and_survivor_diagnostic_not_quality_evidence",
            controls=controls,
            controls_passed=controls_passed,
            extensions=extensions,
            existing_pool_exhausted=all_failed,
            draft_conditioned_design_review_indicated=bool(controls_passed and all_failed),
            original_guard_unchanged=True,
            fp64_KL_is_shadow_only=True,
            model_precision_unchanged=True,
            active_cap_bytes=budget.cap_bytes,
            initial_prompt_storage_bytes=budget.prompt_budget_bytes,
            generated_reserve_bytes=budget.reserved_generated_bytes,
            padding_bytes=0,
            benchmark_labels_accessed=False,
            selection_modified_by_probe=False,
            used_fullkv_fallback=False,
            diagnostic_wall_seconds=time.perf_counter() - started,
            **storage,
            fresh_cohort_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
        )
        raise PrefillDiagnosticComplete(record)
