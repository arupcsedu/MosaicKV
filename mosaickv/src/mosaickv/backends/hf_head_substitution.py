"""Isolated model probes for the V7 one-head diagnostic, never benchmark labels."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from mosaickv.measurements.cache_lifecycle import CacheLifecycle


class HeadProbeSession:
    def __init__(
        self, adapter: Any, prefill: Any, anchor: Any, budget: Any, policy: Any, *, sink_tokens: int
    ) -> None:
        if budget is None or budget.generated_positions < 1:
            raise ValueError("one-head diagnostic requires a prompt/history byte budget")
        self.adapter, self.prefill, self.anchor = adapter, prefill, anchor
        self.budget, self.policy, self.sink_tokens = budget, policy, sink_tokens
        self.reference = self.anchor_state = None
        self.evidence: list[Any] = []
        self.candidate_updates = 0
        self.state_preparation_seconds = 0.0
        self.private_probe_peak_bytes = 0
        self.live: dict[int, Any] = {}
        self.next_id = 0
        if (
            self.bytes(tuple((layer.key, layer.value) for layer in anchor.snapshot.layers))
            != budget.prompt_budget_bytes
        ):
            raise ValueError("one-head anchor violates exact physical byte cap")
        self.tensors = [
            tensor
            for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]
            for tensor in pair
        ]
        self.tensors += [t for layer in anchor.snapshot.layers for t in (layer.key, layer.value)]
        self.saved = [tensor.detach().clone() for tensor in self.tensors]
        self.logical = (
            prefill.state.active_cache_length,
            prefill.state.logical_sequence_length,
            prefill.state.next_decode_position,
        )

    @staticmethod
    def bytes(layers):
        return CacheLifecycle.count(layers)[1]

    def sync(self):
        if self.adapter.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.adapter.device)

    def record_storage(self):
        states = [self.reference, self.anchor_state, *self.live.values()]
        size = sum(
            self.bytes(self.adapter._legacy_layers(state.past_key_values)[0])
            for state in states
            if state is not None
        )
        self.private_probe_peak_bytes = max(self.private_probe_peak_bytes, size)

    def private(self, packed):
        from mosaickv.adapters.huggingface.bounded_history_cache import install_bounded_history
        from mosaickv.backends.hf_runtime import _clone_decode_state, _state_from_packed

        self.sync()
        start = time.perf_counter()
        state = _clone_decode_state(
            self.adapter, _state_from_packed(self.adapter, self.prefill.state, packed)
        )
        install_bounded_history(
            self.adapter,
            state,
            generated_capacity=self.budget.generated_positions,
            sink_tokens=min(self.sink_tokens, self.budget.generated_positions - 1),
            cap_bytes=self.budget.cap_bytes,
        )
        self.sync()
        self.state_preparation_seconds += time.perf_counter() - start
        return state

    def measure(self, full, out):
        import torch

        active = self.bytes(self.adapter._legacy_layers(out.state.past_key_values)[0])
        if active != self.budget.cap_bytes:
            raise RuntimeError("one-head probe violated exact live storage cap")
        logp, logq = (
            torch.log_softmax(full.logits.float(), -1),
            torch.log_softmax(out.logits.float(), -1),
        )
        return dict(
            logit_kl=float((logp.exp() * (logp - logq)).sum(-1).mean().item()),
            maximum_logit_difference=float(
                (full.logits.float() - out.logits.float()).abs().max().item()
            ),
            greedy_token_agreement=bool(torch.equal(full.next_token_id, out.next_token_id)),
            active_bytes=active,
        )

    def prefix(self, step):
        from mosaickv.backends.hf_runtime import _clone_decode_state

        if len(self.evidence) < step:
            if self.reference is None:
                self.reference = _clone_decode_state(self.adapter, self.prefill.state)
                self.adapter.preallocate_decode_cache(self.reference, self.policy.horizon)
                self.anchor_state = self.private(self.anchor)
            token = (
                self.prefill.next_token_id.detach().clone()
                if not self.evidence
                else self.evidence[-1][1].next_token_id
            )
            full = self.adapter.decode_one_token(token, self.reference, capture_queries=False)
            self.reference = full.state
            original = self.adapter.decode_one_token(
                token, self.anchor_state, capture_queries=False
            )
            self.anchor_state = original.state
            frozen = SimpleNamespace(
                logits=full.logits.detach().clone(),
                next_token_id=full.next_token_id.detach().clone(),
            )
            self.evidence.append((token.detach().clone(), frozen, self.measure(full, original)))
            self.record_storage()
        return self.evidence[step - 1]

    def probe(self, packed):
        identity = self.next_id
        self.next_id += 1
        state = self.private(packed)
        self.live[identity] = state
        self.record_storage()
        try:
            for step in range(1, self.policy.horizon + 1):
                token, full, original = self.prefix(step)
                out = self.adapter.decode_one_token(token, state, capture_queries=False)
                state = out.state
                self.live[identity] = state
                self.candidate_updates += 1
                self.record_storage()
                yield dict(
                    step=step,
                    checkpoint=step in self.policy.checkpoints,
                    anchor=original,
                    candidate=self.measure(full, out),
                )
        finally:
            self.live.pop(identity, None)

    def finish(self):
        import torch

        if self.live:
            raise RuntimeError("one-head diagnostic leaked private probe states")
        if not all(torch.equal(a, b) for a, b in zip(self.tensors, self.saved, strict=True)):
            raise RuntimeError("one-head probes mutated original cache tensors")
        if self.logical != (
            self.prefill.state.active_cache_length,
            self.prefill.state.logical_sequence_length,
            self.prefill.state.next_decode_position,
        ):
            raise RuntimeError("one-head probes mutated original logical positions")
        return dict(
            candidate_updates=self.candidate_updates,
            reference_updates=len(self.evidence),
            anchor_updates=len(self.evidence),
            probe_updates=self.candidate_updates + 2 * len(self.evidence),
            state_preparation_seconds=self.state_preparation_seconds,
            measured_private_probe_peak_KV_bytes=self.private_probe_peak_bytes,
            source_verification_copy_bytes=sum(t.numel() * t.element_size() for t in self.saved),
            measured_source_cache_unchanged=True,
            logical_positions_unchanged=True,
            live_private_states_after_diagnostic=len(self.live),
            fullkv_prefills=0,
        )
