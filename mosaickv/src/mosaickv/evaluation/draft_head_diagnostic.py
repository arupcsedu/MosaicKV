"""V8 draft-conditioned one-head diagnostic on two consumed inputs only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

from mosaickv.backends.hf_allocation_guard import packed_positions_sha
from mosaickv.backends.hf_draft_head_evidence import DraftAttentionRecorder
from mosaickv.backends.hf_head_substitution import HeadProbeSession
from mosaickv.backends.hf_runtime import PrefillDiagnosticComplete
from mosaickv.evaluation.head_substitution_diagnostic import HeadSubstitutionDiagnostic
from mosaickv.evaluation.head_substitution_followup import (
    audit_null_control,
    clone_anchor,
    repack_anchor,
)
from mosaickv.selection.draft_head_substitution import DraftHeadPoolPolicy, draft_substitution_pool
from mosaickv.selection.draft_shortlist import draft_shortlist, validate_shortlist_trace
from mosaickv.selection.head_substitution import pack_substitution


class DraftConditionedHeadDiagnostic(HeadSubstitutionDiagnostic):
    def __init__(
        self,
        *args: Any,
        anchor_positions_sha256: str,
        reference_logits: dict[str, Any],
        pool_policy: DraftHeadPoolPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.expected_positions = anchor_positions_sha256
        self.reference_logits = reference_logits
        self.pool_policy = DraftHeadPoolPolicy() if pool_policy is None else pool_policy
        self.saved_logits: dict[str, dict[int, Any]] = {}
        self.saved_evidence: tuple[dict[str, Any], ...] = ()
        self.decode_observer: Any = None

    def _construct_pool(self, full_state, anchor):
        return draft_substitution_pool(
            full_state,
            anchor,
            self.saved_evidence,
            limit=self.policy.pool_size,
            policy=self.pool_policy,
        )

    def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
        import torch

        if packed_positions_sha(anchor) != self.expected_positions:
            raise ValueError("V8 must reuse the unchanged V7/V7.1 anchor")
        recorder = DraftAttentionRecorder(
            self.adapter, full_state, horizon=self.policy.guard.horizon
        )

        def observe(role, step, logits):
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
            capture_draft_attentions=True,
            draft_observer=recorder.observe,
            decode_observer=self.decode_observer,
        )
        session.sync()
        started = time.perf_counter()
        controls, hashes, proposals = {}, {}, {}
        selection = dict(
            accepted=False, meaningful=False, winner=None, events=[], candidate_updates=0
        )
        pool_trace = []
        capture_seconds = pool_seconds = packing_seconds = 0.0
        replayed = False
        try:
            for name, packed in (
                ("clone", clone_anchor(anchor)),
                ("repack", repack_anchor(full_state, anchor)),
            ):
                if packed_positions_sha(packed) != self.expected_positions or not all(
                    torch.equal(a.key, b.key) and torch.equal(a.value, b.value)
                    for a, b in zip(anchor.snapshot.layers, packed.snapshot.layers, strict=True)
                ):
                    raise ValueError("V8 null control changed K/V or logical positions")
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
                if name == "clone":
                    session.sync()
                    capture_seconds = time.perf_counter() - started
            controls_passed = all(c["passed"] for c in controls.values())
            replayed = all(
                set(self.saved_logits[role]) == set(range(1, 33))
                and set(self.reference_logits[role]) == set(range(1, 33))
                and all(
                    torch.equal(t, self.reference_logits[role][step])
                    for step, t in self.saved_logits[role].items()
                )
                for role in ("fullkv", "anchor")
            )
            self.saved_evidence = recorder.result()
            if controls_passed and replayed:
                session.sync()
                t0 = time.perf_counter()
                pool, pool_trace = self._construct_pool(full_state, anchor)
                session.sync()
                pool_seconds = time.perf_counter() - t0

                def identity(p):
                    return hashlib.sha256(
                        json.dumps(asdict(p), sort_keys=True).encode()
                    ).hexdigest()

                def probe(p):
                    nonlocal packing_seconds
                    session.sync()
                    t0 = time.perf_counter()
                    packed = pack_substitution(full_state, anchor, p)
                    session.sync()
                    packing_seconds += time.perf_counter() - t0
                    key = identity(p)
                    hashes[key] = packed_positions_sha(packed)
                    proposals[key] = asdict(p)
                    yield from session.probe(packed, label=key)

                if pool:
                    selection = draft_shortlist(
                        tuple((identity(p), p) for p in pool),
                        probe,
                        self.policy.guard,
                        short_horizon=self.policy.short_horizon,
                        shortlist_size=self.policy.shortlist_size,
                        synchronize=session.sync,
                    )
                    validate_shortlist_trace(selection)
            if session.candidate_updates != 64 + selection["candidate_updates"]:
                raise RuntimeError("V8 control/proposal probe accounting mismatch")
        finally:
            storage = session.finish()
            self.last_probe_storage = storage
        session.sync()
        record = dict(
            classification="V8_consumed_input_draft_conditioned_diagnostic_not_quality_evidence",
            diagnostic_passed=bool(
                controls_passed and replayed and selection["accepted"] and selection["meaningful"]
            ),
            controls=controls,
            controls_passed=controls_passed,
            original_reference_and_anchor_32_prefixes_replayed=bool(replayed),
            policy=asdict(self.policy),
            proposal_policy=asdict(self.pool_policy),
            selection=selection,
            proposals=proposals,
            candidate_positions_sha256=hashes,
            pool_trace=pool_trace,
            pool_reason="draft_output_improving_proposals"
            if pool_trace
            else "no_draft_output_improving_proposals"
            if controls_passed and replayed
            else "controls_or_reference_replay_failed_no_proposals_tested",
            evidence_provenance="actual_eager_FullKV_greedy_draft_attention_not_prompt_only",
            evidence_prefixes=len(recorder.steps),
            evidence_storage_bytes=recorder.storage_bytes,
            draft_evidence_capture_seconds=recorder.capture_seconds,
            original_guard_unchanged=True,
            fp64_KL_is_shadow_only=True,
            one_head_replacement_only=True,
            layer_capacities_changed=False,
            anchor_positions_sha256=self.expected_positions,
            selected_positions_sha256=self.expected_positions,
            active_cap_bytes=budget.cap_bytes,
            initial_prompt_storage_bytes=budget.prompt_budget_bytes,
            generated_reserve_bytes=budget.reserved_generated_bytes,
            padding_bytes=0,
            control_and_evidence_seconds=capture_seconds,
            pool_seconds=pool_seconds,
            packing_seconds=packing_seconds,
            diagnostic_wall_seconds=time.perf_counter() - started,
            **storage,
            benchmark_labels_accessed=False,
            selection_modified_by_probe=False,
            used_fullkv_fallback=False,
            second_prefill_performed=False,
            fresh_cohort_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
        )
        raise PrefillDiagnosticComplete(record)
