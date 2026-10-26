"""V7 consumed-identity mechanistic diagnostic, not quality evidence."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from mosaickv.backends.hf_allocation_guard import packed_positions_sha
from mosaickv.backends.hf_head_substitution import HeadProbeSession
from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, PrefillDiagnosticComplete
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.selection.draft_shortlist import draft_shortlist, validate_shortlist_trace
from mosaickv.selection.head_substitution import pack_substitution, substitution_pool


@dataclass(frozen=True)
class HeadDiagnosticPolicy:
    pool_size: int = 16
    short_horizon: int = 4
    shortlist_size: int = 2
    guard: DraftAllocationGuardConfig = field(default_factory=DraftAllocationGuardConfig)

    def __post_init__(self):
        if (
            type(self.pool_size) is not int
            or not 1 <= self.pool_size <= 16
            or type(self.short_horizon) is not int
            or not 1 <= self.short_horizon <= 4
            or type(self.shortlist_size) is not int
            or not 1 <= self.shortlist_size <= 2
            or self.guard != DraftAllocationGuardConfig()
        ):
            raise ValueError(
                "head diagnostic must preserve the V6 horizon/thresholds and bounded probes"
            )


class HeadSubstitutionDiagnostic(HuggingFaceMosaicKVModel):
    def __init__(self, *args: Any, policy: HeadDiagnosticPolicy | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if (
            self.config.bounded_decode is None
            or self.config.bounded_decode.layer_allocation != "uniform"
            or self.config.method.value != "mosaickv_exact"
            or self.config.cache.block_size != 1
            or self.config.cache.retention_ratio == 1.0
        ):
            raise ValueError("one-head diagnostic requires the unmodified compressed exact anchor")
        self.policy = HeadDiagnosticPolicy() if policy is None else policy

    def diagnose(self, request):
        if request.generation_kwargs:
            raise ValueError(
                "one-head diagnostic accepts only prompt/media, never targets or references"
            )
        try:
            self.generate(request)
        except PrefillDiagnosticComplete as complete:
            return complete.record
        raise RuntimeError("one-head diagnostic did not intercept the exact anchor")

    def _observe_exact_anchor(self, prefill, full_state, anchor, attention, budget):
        import torch

        if attention is None:
            raise ValueError("one-head diagnostic lacks source prompt attention")

        def sync():
            if self.adapter.device.type == "cuda":
                torch.cuda.synchronize(self.adapter.device)

        sync()
        start = time.perf_counter()
        pool = substitution_pool(
            full_state,
            attention,
            anchor,
            prompt_bytes=budget.prompt_budget_bytes,
            limit=self.policy.pool_size,
        )
        sync()
        pool_seconds = time.perf_counter() - start
        session = HeadProbeSession(
            self.adapter,
            prefill,
            anchor,
            budget,
            self.policy.guard,
            sink_tokens=self.config.bounded_decode.sink_tokens,
        )
        proposal_records = {}
        hashes = {}
        packing_seconds = 0.0

        def identity(proposal):
            return hashlib.sha256(json.dumps(asdict(proposal), sort_keys=True).encode()).hexdigest()

        def probe(proposal):
            nonlocal packing_seconds
            sync()
            t0 = time.perf_counter()
            packed = pack_substitution(full_state, anchor, proposal)
            sync()
            packing_seconds += time.perf_counter() - t0
            key = identity(proposal)
            proposal_records[key] = asdict(proposal)
            hashes[key] = packed_positions_sha(packed)
            yield from session.probe(packed)

        try:
            selection = (
                draft_shortlist(
                    tuple((identity(p), p) for p in pool),
                    probe,
                    self.policy.guard,
                    short_horizon=self.policy.short_horizon,
                    shortlist_size=self.policy.shortlist_size,
                    synchronize=sync,
                )
                if pool
                else dict(
                    accepted=False,
                    meaningful=False,
                    winner=None,
                    events=[],
                    shortlist=[],
                    reason="no_feasible_substitution",
                )
            )
        finally:
            storage = session.finish()
        if pool:
            validate_shortlist_trace(selection)
            if selection["candidate_updates"] != storage["candidate_updates"]:
                raise RuntimeError("one-head diagnostic probe count mismatch")
        sync()
        record = dict(
            classification="consumed_identity_label_blind_mechanistic_diagnostic_not_quality_or_latency_evidence",
            policy=asdict(self.policy),
            selection=selection,
            proposals=proposal_records,
            candidate_positions_sha256=hashes,
            anchor_positions_sha256=packed_positions_sha(anchor),
            selected_positions_sha256=hashes[selection["winner"]]
            if selection["winner"] is not None
            else packed_positions_sha(anchor),
            initial_prompt_storage_bytes=budget.prompt_budget_bytes,
            active_cap_bytes=budget.cap_bytes,
            generated_reserve_bytes=budget.reserved_generated_bytes,
            layer_capacities=[int(layer.key.shape[-2]) for layer in anchor.snapshot.layers],
            one_head_replacement_only=True,
            layer_capacities_changed=False,
            padding_bytes=0,
            pool_seconds=pool_seconds,
            packing_seconds=packing_seconds,
            diagnostic_wall_seconds=time.perf_counter() - start,
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
