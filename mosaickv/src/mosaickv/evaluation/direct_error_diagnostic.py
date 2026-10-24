"""Evaluation-only FullKV-draft logit audit; never feeds online selection."""

from __future__ import annotations

import math
import time
from typing import Any, cast

from mosaickv.backends.hf_runtime import (
    HuggingFaceMosaicKVModel,
    PrefillDiagnosticComplete,
    _clone_decode_state,
    _state_from_packed,
)
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.measurements.cache_lifecycle import CacheLifecycle


class DirectErrorLogitDiagnostic(HuggingFaceMosaicKVModel):
    def __init__(self, *args: Any, horizon: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(horizon) is not int or not 1 <= horizon <= 8:
            raise ValueError("diagnostic horizon must be in [1, 8]")
        if (
            self.config.bounded_decode is None
            or self.config.bounded_decode.layer_allocation != "direct_error_exchange_v1"
        ):
            raise ValueError("diagnostic accepts the frozen V4 allocation only")
        self.horizon = horizon

    def diagnose(self, request: EvaluationRequest) -> dict[str, Any]:
        if request.generation_kwargs:
            raise ValueError(
                "diagnostic accepts prompt/media only, never targets or reference labels"
            )
        try:
            self.generate(request)
        except PrefillDiagnosticComplete as completed:
            return cast("dict[str, Any]", completed.record)
        raise RuntimeError("diagnostic failed to intercept cache planning")

    def _observe_evidence_allocation(
        self,
        prefill: Any,
        anchor: Any,
        candidate: Any,
        budget: Any,
        allocation_trace: dict[str, Any],
    ) -> None:
        import torch

        from mosaickv.adapters.huggingface.bounded_history_cache import install_bounded_history

        if budget is None or budget.generated_positions < self.horizon:
            raise ValueError("diagnostic horizon exceeds preregistered generated reserve")
        assert self.config.bounded_decode is not None
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)
        started = time.perf_counter()
        originals = [
            tensor
            for pair in self.adapter._legacy_layers(prefill.state.past_key_values)[0]
            for tensor in pair
        ]
        originals += [
            tensor
            for packed in (anchor, candidate)
            for layer in packed.snapshot.layers
            for tensor in (layer.key, layer.value)
        ]
        saved = [tensor.detach().clone() for tensor in originals]
        logical_before = (
            prefill.state.active_cache_length,
            prefill.state.logical_sequence_length,
            prefill.state.next_decode_position,
        )
        reference = _clone_decode_state(self.adapter, prefill.state)
        states = {}
        for name, packed in (("anchor", anchor), ("candidate", candidate)):
            state = _clone_decode_state(
                self.adapter, _state_from_packed(self.adapter, prefill.state, packed)
            )
            install_bounded_history(
                self.adapter,
                state,
                generated_capacity=budget.generated_positions,
                sink_tokens=min(
                    self.config.bounded_decode.sink_tokens, budget.generated_positions - 1
                ),
                cap_bytes=budget.cap_bytes,
            )
            states[name] = state
        initial = {
            name: CacheLifecycle.count(self.adapter._legacy_layers(state.past_key_values)[0])[1]
            for name, state in states.items()
        }
        token = prefill.next_token_id.detach().clone()
        prefix, steps = [], []
        for index in range(self.horizon):
            prefix.append(int(token.item()))
            full = self.adapter.decode_one_token(token, reference, capture_queries=False)
            reference = full.state
            logp = torch.log_softmax(full.logits.float(), dim=-1)
            p = logp.exp()
            row: dict[str, Any] = {"step": index + 1}
            for name, state in states.items():
                output = self.adapter.decode_one_token(token, state, capture_queries=False)
                states[name] = output.state
                logq = torch.log_softmax(output.logits.float(), dim=-1)
                row[name] = dict(
                    logit_kl=float((p * (logp - logq)).sum(-1).mean().item()),
                    maximum_logit_difference=float(
                        (full.logits.float() - output.logits.float()).abs().max().item()
                    ),
                    greedy_token_agreement=bool(
                        torch.equal(full.next_token_id, output.next_token_id)
                    ),
                    active_bytes=CacheLifecycle.count(
                        self.adapter._legacy_layers(output.state.past_key_values)[0]
                    )[1],
                )
                if not all(
                    math.isfinite(row[name][key])
                    for key in ("logit_kl", "maximum_logit_difference")
                ):
                    raise RuntimeError("isolated probe produced non-finite logits")
                if row[name]["active_bytes"] != budget.cap_bytes:
                    raise RuntimeError("isolated probe violated compressed storage cap")
            steps.append(row)
            token = full.next_token_id
        if not all(torch.equal(a, b) for a, b in zip(originals, saved, strict=True)):
            raise RuntimeError("diagnostic mutated source or measured-run cache storage")
        if logical_before != (
            prefill.state.active_cache_length,
            prefill.state.logical_sequence_length,
            prefill.state.next_decode_position,
        ):
            raise RuntimeError("diagnostic changed original logical positions")
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)
        record = dict(
            classification="evaluation_only_label_blind_V4_logit_diagnostic_not_quality_or_latency_evidence",
            horizon=self.horizon,
            trajectory="forced_FullKV_greedy_draft_prefix_not_benchmark_targets",
            benchmark_labels_accessed=False,
            selection_modified_by_probe=False,
            fullkv_prefill_reused=True,
            second_prefill_performed=False,
            measured_source_cache_unchanged=True,
            logical_positions_unchanged=True,
            initial_compressed_storage_bytes=initial,
            fullkv_draft_prefix=prefix,
            allocation=allocation_trace,
            steps=steps,
            diagnostic_wall_seconds=time.perf_counter() - started,
            promotion_authorized=False,
        )
        raise PrefillDiagnosticComplete(record)
