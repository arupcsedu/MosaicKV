"""Instrument frozen V8 proposals, never generate or promote a new proposal."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from mosaickv.backends.hf_allocation_guard import packed_positions_sha
from mosaickv.backends.hf_head_substitution import HeadProbeSession
from mosaickv.backends.hf_prefix_outputs import PrefixOutputRecorder, frozen_query_prediction
from mosaickv.backends.hf_runtime import PrefillDiagnosticComplete
from mosaickv.evaluation.head_substitution_diagnostic import (
    HeadDiagnosticPolicy,
    HeadSubstitutionDiagnostic,
)
from mosaickv.evaluation.head_substitution_followup import (
    audit_null_control,
    clone_anchor,
    repack_anchor,
)
from mosaickv.selection.head_substitution import HeadSubstitution, pack_substitution


def frozen_trajectories(record):
    """Fail closed on missing/unresolved proposals; replay each observed prefix once."""
    import hashlib
    import json

    if json.dumps(record["policy"], sort_keys=True) != json.dumps(
        asdict(HeadDiagnosticPolicy()), sort_keys=True
    ):
        raise ValueError("frozen V8 guard/pool policy changed")
    if record["selection"]["accepted"] or not record["controls_passed"]:
        raise ValueError("this replay requires the negative, controlled V8 diagnostic")
    rows = {}
    for event in record["selection"]["events"]:
        key = event["proposal_id"]
        result = rows.setdefault(key, {})
        for row in event["steps"]:
            if row["step"] in result and result[row["step"]] != row:
                raise ValueError("conflicting frozen prefix records")
            result[row["step"]] = row
    if not rows or len(rows) > 16 or set(rows) != set(record["proposals"]):
        raise ValueError("incomplete or oversized frozen proposal set")
    trajectories = []
    for key, steps in rows.items():
        raw = record["proposals"][key]
        if hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest() != key:
            raise ValueError("frozen proposal identity changed")
        if set(steps) != set(range(1, max(steps) + 1)) or max(steps) > 32:
            raise ValueError("frozen prefix trajectory is incomplete")
        trajectories.append((key, HeadSubstitution(**raw), steps))
    return tuple(trajectories)


class PrefixOutputReplay(HeadSubstitutionDiagnostic):
    def __init__(
        self,
        *args: Any,
        frozen_record: dict[str, Any],
        reference_logits: dict[str, Any],
        draft_evidence: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frozen = frozen_record
        self.trajectories = frozen_trajectories(frozen_record)
        self.reference_logits = reference_logits
        self.draft_evidence = draft_evidence
        self.saved_logits: dict[str, dict[int, Any]] = {}
        self.saved_outputs: dict[str, dict[int, Any]] = {}
        self.saved_predictions: dict[str, Any] = {}

    def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
        import torch

        expected = self.frozen["anchor_positions_sha256"]
        if packed_positions_sha(anchor) != expected:
            raise ValueError("output replay must use the frozen V8 anchor")
        recorder = PrefixOutputRecorder(self.adapter)
        self.saved_outputs = recorder.records

        def observe(role, step, logits):
            self.saved_logits.setdefault(role, {})[step] = logits.detach().cpu().clone()
            if not torch.equal(self.saved_logits[role][step], self.reference_logits[role][step]):
                raise RuntimeError(f"instrumentation changed frozen logits: {role} prefix {step}")

        session = HeadProbeSession(
            self.adapter,
            prefill,
            anchor,
            budget,
            self.policy.guard,
            sink_tokens=self.config.bounded_decode.sink_tokens,
            capture_fp64=True,
            capture_draft_attentions=True,
            logits_observer=observe,
            decode_observer=recorder,
        )
        session.sync()
        started = time.perf_counter()
        controls, replay = {}, []
        try:
            for name, packed in (
                ("clone", clone_anchor(anchor)),
                ("repack", repack_anchor(full_state, anchor)),
            ):
                stream = session.probe(packed, label=f"null_{name}")
                try:
                    rows = list(stream)
                finally:
                    stream.close()
                controls[name] = dict(steps=rows, **audit_null_control(rows, self.policy.guard))
            if not all(c["passed"] for c in controls.values()):
                raise RuntimeError("instrumented clone/repack control failed")
            for key, proposal, steps in self.trajectories:
                packed = pack_substitution(full_state, anchor, proposal)
                if packed_positions_sha(packed) != self.frozen["candidate_positions_sha256"][key]:
                    raise ValueError("instrumented replay changed frozen proposal positions")
                self.saved_predictions[key] = frozen_query_prediction(
                    full_state,
                    anchor,
                    proposal,
                    self.draft_evidence,
                    recorder.layers[proposal.layer].self_attn.o_proj,
                )
                stream = session.probe(packed, label=key)
                observed = []
                try:
                    for row in stream:
                        old = steps[row["step"]]
                        for role in ("anchor", "candidate"):
                            for field in (
                                "logit_kl",
                                "maximum_logit_difference",
                                "greedy_token_agreement",
                                "active_bytes",
                                "logit_kl_fp64",
                            ):
                                if row[role][field] != old[role][field]:
                                    raise RuntimeError("frozen registered/shadow guard changed")
                        observed.append(row)
                        if row["step"] == len(steps):
                            break
                finally:
                    stream.close()
                replay.append(dict(proposal_id=key, proposal=asdict(proposal), steps=observed))
        finally:
            storage = session.finish()
        if set(self.saved_logits) != set(self.reference_logits) or any(
            set(self.saved_logits[role]) != set(self.reference_logits[role])
            for role in self.saved_logits
        ):
            raise RuntimeError("frozen prefix coverage changed")
        null_outputs_equal = all(
            torch.equal(record[field], self.saved_outputs["anchor"][step][layer][field])
            for role in ("null_clone", "null_repack")
            for step, layers in self.saved_outputs[role].items()
            for layer, record in enumerate(layers)
            for field in recorder.FIELDS
        )
        if not null_outputs_equal:
            raise RuntimeError("null-control intermediate outputs are not bitwise identical")
        session.sync()
        raise PrefillDiagnosticComplete(
            dict(
                classification="V8_1_consumed_input_frozen_proposal_output_replay_not_quality_evidence",
                instrumentation_passed=True,
                instrumented_logits_bitwise_replayed=True,
                intermediate_null_controls_bitwise_equal=True,
                frozen_proposals_replayed=len(replay),
                frozen_proposal_prefixes=sum(len(row["steps"]) for row in replay),
                controls=controls,
                controls_passed=True,
                replay=replay,
                policy=asdict(self.policy),
                active_cap_bytes=budget.cap_bytes,
                initial_prompt_storage_bytes=budget.prompt_budget_bytes,
                generated_reserve_bytes=budget.reserved_generated_bytes,
                padding_bytes=0,
                output_capture_storage_bytes=recorder.storage_bytes,
                captured_fields=list(recorder.FIELDS),
                captured_layers=len(recorder.layers),
                installed_source_contract=recorder.source_contract,
                original_guard_unchanged=True,
                fp64_KL_is_shadow_only=True,
                proposal_construction_revised=False,
                accepted_proposals=0,
                benchmark_labels_accessed=False,
                used_fullkv_fallback=False,
                second_prefill_performed=False,
                fresh_cohort_authorized=False,
                calibration_authorized=False,
                paper_campaign_authorized=False,
                diagnostic_wall_seconds=time.perf_counter() - started,
                **storage,
            )
        )
