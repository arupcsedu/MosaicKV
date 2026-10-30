"""Immutable exact 16-token history chunks plus one reusable raw tail."""

from typing import Any

import torch

from mosaickv.evaluation.exact_storage import ExactLayer, encode
from mosaickv.evaluation.lossless_bf16 import PackedBF16, unpack_into
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity


class SealedHistoryLayer(ExactLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sealed: list[tuple[PackedBF16, ...]] = []
        self.tail_used = 0

    def owned_tensors(self) -> tuple[Any, ...]:
        return (
            *super().owned_tensors(),
            *(t for pair in self.sealed for b in pair for t in b.buffers),
        )

    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        from contextlib import nullcontext

        del cache_kwargs
        expected = (*self.prompt_shape[:2], 1, self.prompt_shape[-1])
        if (
            self.leased
            or key_states.shape != expected
            or value_states.shape != expected
            or key_states.dtype != torch.bfloat16
            or value_states.dtype != torch.bfloat16
            or self.history_length >= self.maximum_decode_updates
        ):
            raise ValueError("released lease and bounded one-token BF16 history required")
        if not self.history_capacity:
            capacity = min(16, self.maximum_decode_updates)
            newer = tuple(
                torch.empty(
                    (*expected[:2], capacity, expected[-1]),
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                for _ in range(2)
            )
            self.observe(newer)
            self.history, self.history_capacity = newer, capacity
            self.history_bytes = backing_bytes(newer)
            self.resident_bytes += self.history_bytes
        if self.tail_used == self.history_capacity:
            mode = (
                ScratchAudit(
                    excluded=self.weights | {identity(t) for t in self.owned_tensors()},
                    observe=self.observe,
                )
                if self.audit
                else None
            )
            with mode if mode is not None else nullcontext():
                pair = tuple(encode(t) for t in self.history)
            if mode is not None:
                self.workspace_operations += mode.operations
            # Both CPU/GPU raw fallbacks clone: never alias a tail we overwrite.
            tail_owners = {identity(t) for t in self.history}
            if any(identity(t) in tail_owners for b in pair for t in b.buffers):
                raise RuntimeError("sealed history aliases mutable tail")
            self.sealed.append(pair)
            additional = backing_bytes(tuple(t for b in pair for t in b.buffers))
            self.history_bytes += additional
            self.resident_bytes += additional
            self.tail_used = 0
            self.observe(())
        for target, value in zip(self.history, (key_states, value_states), strict=True):
            target[..., self.tail_used : self.tail_used + 1, :].copy_(value)
        self.tail_used += 1
        self.history_length += 1
        self.leased = tuple(
            torch.empty(
                (*expected[:2], self.get_seq_length(), expected[-1]),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            for _ in range(2)
        )
        self.lease_bytes = backing_bytes(self.leased)
        self.observe(())
        self.materialize()
        return self.leased

    def fill_history(self) -> None:
        offset = self.prompt_length
        for pair in self.sealed:
            count = pair[0].shape[-2]
            for body, target in zip(pair, self.leased, strict=True):
                unpack_into(body, target[..., offset : offset + count, :])
            offset += count
        for tail, target in zip(self.history, self.leased, strict=True):
            target[..., offset : offset + self.tail_used, :].copy_(tail[..., : self.tail_used, :])
        if offset + self.tail_used != self.get_seq_length():
            raise RuntimeError("sealed history lost or duplicated a token")
