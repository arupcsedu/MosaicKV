"""Online-draft history repair diagnostic, not a paper-ready compressor.

Reference states come from this request's FULLKV model rollout, never labels.
They replace only GENERATED K/V while the actual input prefix matches that
rollout. Prompt selections and K/V are untouched. CPU residuals are charged.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class HistoryReference:
    kv: object
    tokens: object
    logical_start: int

    def __post_init__(self):
        if (
            self.kv.device.type != "cpu"
            or self.tokens.device.type != "cpu"
            or self.kv.dtype != torch.bfloat16
            or self.tokens.dtype != torch.long
            or self.kv.ndim != 6
            or self.kv.shape[2] != 2
            or self.kv.shape[-2:] != (1, 128)
            or self.tokens.shape != self.kv.shape[:1]
            or not 1 <= len(self.tokens) <= 128
            or not self.kv.is_contiguous()
            or not self.tokens.is_contiguous()
            or not torch.isfinite(self.kv).all()
            or (self.tokens < 0).any()
            or type(self.logical_start) is not int
            or self.logical_start < 1
        ):
            raise ValueError("CPU BF16 T,L,2,H,1,128 history and original token positions required")

    @property
    def cpu_bytes(self):
        return backing_bytes((self.kv, self.tokens))


class HistoryRepairTransport:
    def __init__(self, adapter, cache, reference, *, enabled):
        if type(enabled) is not bool or not isinstance(reference, HistoryReference):
            raise ValueError("explicit reference and repair-enable boolean required")
        self.adapter, self.cache, self.reference = adapter, cache, reference
        self.enabled = enabled
        self.stack = ExitStack()
        self.steps = self.repaired_steps = 0
        self.matched = True
        self.current = False
        self.active = False
        self.seen = set()
        self.first_input_mismatch = None
        self.events = []

    def __enter__(self):
        device = self.cache.layers[0].representation.private()[0].device
        r = self.reference
        if len(self.cache.layers) != r.kv.shape[1] or any(
            layer.prompt_shape[1] != r.kv.shape[3] for layer in self.cache.layers
        ):
            raise ValueError("reference and request geometry differ")
        if device.type == "cuda" and not r.kv.is_pinned():
            raise ValueError("GPU history reference must be pinned CPU memory")
        self.stage = torch.zeros(r.kv.shape[1:], device=device, dtype=torch.bfloat16)
        self.cache.shared_tensors = (*self.cache.shared_tensors, self.stage)
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.cache.initial_bytes += backing_bytes((self.stage,))
        self.cache.observe()
        original = self.adapter.decode_one_token

        def decode(token, state, **kwargs):
            if self.active or state.past_key_values is not self.cache:
                raise RuntimeError("nested or foreign history-repair request")
            if state.next_decode_position != r.logical_start + self.steps:
                raise RuntimeError("history repair logical position changed")
            self.active, self.current = True, False
            self.seen.clear()
            if self.steps < len(r.tokens):
                equal = int(token.item()) == int(r.tokens[self.steps])
                if self.matched and not equal:
                    self.first_input_mismatch = self.steps
                self.matched = self.matched and equal
                self.current = self.enabled and self.matched
                if self.current:
                    if device.type == "cuda":
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                    self.stage.copy_(r.kv[self.steps], non_blocking=device.type == "cuda")
                    if device.type == "cuda":
                        end.record()
                        self.events.append((start, end))
            try:
                output = original(token, state, **kwargs)
                if self.seen != set(range(len(self.cache.layers))):
                    raise RuntimeError("incomplete or duplicate repaired-layer update")
                self.repaired_steps += int(self.current)
                self.steps += 1
                return output
            finally:
                self.active = False
                self.current = False

        try:
            self.stack.enter_context(patch.object(self.adapter, "decode_one_token", decode))
            for i, layer in enumerate(self.cache.layers):
                update = layer.update

                def replace(key, value, cache_kwargs=None, *, index=i, target=layer, fn=update):
                    if not self.active or index in self.seen or target.history_length != self.steps:
                        raise RuntimeError("out-of-order history repair update")
                    self.seen.add(index)
                    if self.current:
                        expected = (1, *self.stage[index, 0].shape)
                        if key.shape != expected or value.shape != expected:
                            raise RuntimeError("generated reference K/V shape changed")
                        key = self.stage[index, 0][None]
                        value = self.stage[index, 1][None]
                    return fn(key, value, cache_kwargs)

                self.stack.enter_context(patch.object(layer, "update", replace))
        except BaseException:
            self.stack.close()
            raise
        return self

    def record(self):
        # GPU caller synchronizes the entire decode before requesting timings.
        return dict(
            enabled=self.enabled,
            online_reference_horizon=len(self.reference.tokens),
            cpu_residual_storage_bytes=self.reference.cpu_bytes,
            gpu_staging_storage_bytes=backing_bytes((self.stage,)),
            repaired_token_steps=self.repaired_steps,
            repaired_layer_head_blocks=self.repaired_steps
            * self.reference.kv.shape[1]
            * self.reference.kv.shape[3],
            repaired_bytes=self.repaired_steps * self.stage.numel() * self.stage.element_size(),
            first_input_prefix_mismatch=self.first_input_mismatch,
            stopped_permanently_after_prefix_mismatch=True,
            transfer_cuda_ms=sum(a.elapsed_time(b) for a, b in self.events),
            original_prompt_KV_and_memberships_unchanged=True,
            no_fullkv_fallback_or_draft_token_emission=True,
            host_reference_retained_and_fully_charged=True,
        )

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
