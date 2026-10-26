"""FullKV greedy-draft attention evidence, never a second prefill or labels."""

from __future__ import annotations

import time
from typing import Any


class DraftAttentionRecorder:
    def __init__(self, adapter: Any, full_state: Any, *, horizon: int) -> None:
        self.adapter, self.full_state, self.horizon = adapter, full_state, horizon
        self.steps: list[tuple[dict[str, Any], ...]] = []
        self.capture_seconds = 0.0

    def observe(self, step: int, full: Any, anchor_state: Any) -> None:
        import torch

        device = self.full_state.layers[0].value.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if step != len(self.steps) + 1 or step > self.horizon:
            raise ValueError("draft attention prefixes must be complete and ordered")
        layers = self.adapter._legacy_layers(full.state.past_key_values)[0]
        bounded = anchor_state.past_key_values.layers
        if len(full.attention_weights) != len(self.full_state.layers):
            raise ValueError("FullKV eager draft did not expose every layer's attention")
        records = []
        for source, (_, values), probabilities, history in zip(
            self.full_state.layers, layers, full.attention_weights, bounded, strict=True
        ):
            n, heads = source.sequence_length, source.kv_heads
            if (
                probabilities.ndim != 4
                or probabilities.shape[0] != 1
                or probabilities.shape[2] != 1
                or probabilities.shape[1] % heads
                or probabilities.shape[-1] != n + step
                or values.shape[2] != n + step
                or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
            ):
                raise ValueError("invalid FullKV single-token draft attention geometry/values")
            groups = probabilities.shape[1] // heads
            p = probabilities[0, :, 0].float().reshape(heads, groups, n + step)
            total = p.sum(-1, keepdim=True)
            if not bool((total > 0).all()):
                raise ValueError("draft attention has a zero-mass query")
            # Normalize rounding in captured probabilities only for the surrogate.
            # The actual model logits and registered FP32 guard are untouched.
            p = p / total
            v = values[0].float()
            indices = [
                n + position - self.full_state.next_decode_position
                for position in history.generated_logical_positions
            ]
            if (
                not indices
                or len(set(indices)) != len(indices)
                or any(not n <= position < n + step for position in indices)
            ):
                raise ValueError("draft generated history has invalid original positions")
            kept = p[:, :, indices]
            records.append(
                dict(
                    prompt_probabilities=p[:, :, :n].detach().cpu().clone(),
                    target=torch.bmm(p, v).detach().cpu(),
                    generated_numerator=torch.bmm(kept, v[:, indices]).detach().cpu(),
                    generated_mass=kept.sum(-1).detach().cpu(),
                    retained_generated_physical_indices=tuple(indices),
                )
            )
        self.steps.append(tuple(records))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.capture_seconds += time.perf_counter() - started

    def result(self) -> tuple[dict[str, Any], ...]:
        import torch

        if len(self.steps) != self.horizon:
            raise ValueError("draft proposal construction needs the complete registered horizon")
        return tuple(
            {
                key: torch.stack([step[layer][key] for step in self.steps], dim=1).flatten(1, 2)
                for key in (
                    "prompt_probabilities",
                    "target",
                    "generated_numerator",
                    "generated_mass",
                )
            }
            for layer in range(len(self.full_state.layers))
        )

    @property
    def storage_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for step in self.steps
            for layer in step
            for t in layer.values()
            if hasattr(t, "numel")
        )
