"""Full-prompt shared-input precision VARIANT, not the exact-selection method.

Store one quantized normalized input shared by K/V projections. Reconstruct
each original position, retaining the old per-head protected rows bitwise.
No model weights are modified, no RoPE phases are merged, no token is omitted.
"""

from contextlib import ExitStack
from dataclasses import dataclass

import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.exact_storage import decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.protected_projection_source import protected_positions


class SourceCapture:
    """One temporary original projection input per layer, same device, one prefill."""

    def __init__(self, adapter):
        self.adapter, self.stack, self.records = adapter, ExitStack(), {}

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):

                def capture(_module, args, kwargs, *, index=i):
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    if (
                        x is None
                        or index in self.records
                        or x.dtype != torch.bfloat16
                        or x.ndim != 3
                        or x.shape[0] != 1
                    ):
                        raise ValueError("one batch-one BF16 prefill source per layer required")
                    self.records[index] = x.detach().clone().contiguous()

                handle = block.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@dataclass(frozen=True)
class SharedSourceFormat:
    source: object
    shape: tuple
    row_positions: object
    exact: tuple
    cos: object
    sin: object
    union: object
    slots: object
    kind: str = "evicastkv_full_prompt_shared_source_precision_variant"

    def private(self):
        return (
            *self.source.tensors(),
            self.row_positions,
            *(t for body in self.exact for t in body.buffers),
        )

    def shared(self):
        return self.cos, self.sin, self.union, self.slots

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @torch.inference_mode()
    def fill(self, attention, targets, *, direct):
        del direct
        n, h, d = self.shape[2], self.shape[1], self.shape[3]
        if attention is None or len(targets) != 2 or any(
            t.ndim != 4
            or t.shape[:2] != (1, h)
            or t.shape[-1] != d
            or t.shape[-2] < n
            or t.dtype != torch.bfloat16
            or t.device != self.source.codes.device
            for t in targets
        ):
            raise ValueError("actual frozen attention and complete layer lease required")
        if targets[0].device.type == "cuda":
            from mosaickv.evaluation.shared_source_kernel import project

            project(self, attention, targets)
            from mosaickv.evaluation.head_local_precision_kernel import scatter

            for body, target in zip(self.exact, targets, strict=True):
                scatter(body, self.row_positions, target, prompt_length=n)
        else:
            x = unpack_cpu(self.source).squeeze(1)
            k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
            v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
            rotated = torch.cat((-k[..., d // 2 :], k[..., : d // 2]), -1)
            k = k * self.cos[:, None] + rotated * self.sin[:, None]
            for value, target, body in zip((k, v), targets, self.exact, strict=True):
                target[..., :n, :].copy_(value)
                ids = self.row_positions.long()
                target[0, ids // n, ids % n] = decoded(body)[0, 0]

    def materialize(self, attention):
        out = tuple(
            torch.empty(self.shape, device=self.source.codes.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, out, direct=True)
        return out


@torch.inference_mode()
def construct(
    normalized, original_pair, memberships, text, phases, identity, *, exclude_global_exact=False
):
    if type(exclude_global_exact) is not bool or len(original_pair) != 2:
        raise ValueError("explicit source variant and actual K/V pair required")
    key, value = original_pair
    if (
        key.shape != value.shape
        or key.ndim != 4
        or key.shape[0] != 1
        or key.shape[-1] != 128
        or key.dtype != torch.bfloat16
        or value.dtype != key.dtype
    ):
        raise ValueError("batch-one full-MHA BF16 original cache required")
    _, h, n, d = key.shape
    if (
        len(memberships) != h
        or n >= 32768
        or normalized.shape != (1, n, h * d)
        or normalized.device != key.device
        or normalized.dtype != key.dtype
    ):
        raise ValueError("full normalized projection input, not GQA, required")
    cos, sin = phases
    union, slots = identity
    if (
        cos.shape != (1, n, d)
        or sin.shape != cos.shape
        or union.shape != (n,)
        or slots.shape != (h, n)
        or any(t.device != key.device for t in (value, cos, sin, union, slots))
        or cos.dtype != torch.bfloat16
        or sin.dtype != cos.dtype
        or union.dtype != torch.int32
        or slots.dtype != torch.int16
        or not torch.equal(union, torch.arange(n, device=key.device, dtype=torch.int32))
        or not torch.equal(slots, union.to(torch.int16)[None].expand(h, n))
        or not all(torch.isfinite(t).all() for t in (normalized, key, value, cos, sin))
        or not torch.equal(cos[..., :64], cos[..., 64:])
        or not torch.equal(sin[..., :64], sin[..., 64:])
    ):
        raise ValueError("shared original phase and full identity positions required")
    mask = protected_positions(memberships, text, length=n).to(key.device)
    rows = mask.flatten().nonzero().flatten().to(torch.int32)
    if exclude_global_exact:
        source = pack_unprotected(normalized[:, None], mask.all(0)[None], bits=3, axis=2)
    else:
        source = pack(normalized[:, None], bits=3, axis=2)
    head, position = rows.long() // n, rows.long() % n
    exact = tuple(
        encode(t[0, head, position].reshape(1, 1, len(rows), d).contiguous()) for t in original_pair
    )
    result = SharedSourceFormat(source, tuple(key.shape), rows, exact, cos, sin, union, slots)
    forbidden = {t.untyped_storage().data_ptr() for t in (*original_pair, normalized)}
    if any(t.numel() and t.untyped_storage().data_ptr() in forbidden for t in result.private()):
        raise RuntimeError("encoded source or exact rows alias an original full cache/input owner")
    return result
