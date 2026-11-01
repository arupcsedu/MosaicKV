"""Consumed-input precision/capacity diagnostic, not official KVQuant.

Re-express ACTUAL post-RoPE keys (including a baseline's merged keys) in their
anchor position's inverse phase, quantize, and rotate back at the ORIGINAL
positions. This inverse is approximate after BF16 rounding, not an assertion
that the model stored pre-RoPE keys. All exact overlays remain post-RoPE.
"""

from contextlib import ExitStack
from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.text_source_storage import TextSourceFormat


class PhaseCapture:
    """Capture one shared original phase, with no model/cache mutation."""

    def __init__(self, adapter):
        self.adapter, self.stack = adapter, ExitStack()
        self.phases, self.identity = None, None
        self.seen = set()

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):

                def capture(_module, _args, kwargs, *, index=i):
                    phase = kwargs.get("position_embeddings")
                    if phase is None or index in self.seen:
                        raise ValueError("one original phase invocation per prefill layer required")
                    self.seen.add(index)
                    identity = tuple(
                        (x.data_ptr(), tuple(x.shape), tuple(x.stride())) for x in phase
                    )
                    if self.phases is None:
                        self.phases = tuple(x.detach().clone().contiguous() for x in phase)
                        self.identity = identity
                    elif identity != self.identity:
                        raise ValueError("shared LLaVA original phases required")

                hook = block.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
                self.stack.callback(hook.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def selected_phases(positions, cos, sin):
    if (
        positions.ndim != 2
        or positions.dtype != torch.int32
        or cos.ndim != 3
        or cos.shape != sin.shape
        or cos.shape[0] != 1
        or cos.shape[-1] != 128
        or cos.dtype != torch.bfloat16
        or sin.dtype != cos.dtype
        or positions.device != cos.device
        or sin.device != cos.device
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or not torch.isfinite(cos).all()
        or not torch.isfinite(sin).all()
        or not torch.equal(cos[..., :64], cos[..., 64:])
        or not torch.equal(sin[..., :64], sin[..., 64:])
        or (cos.float().square() + sin.float().square()).min() < 0.5
        or (positions < 0).any()
        or (positions >= cos.shape[1]).any()
        or (positions[:, 1:] <= positions[:, :-1]).any()
    ):
        raise ValueError("monotonic original positions and finite shared BF16 LLaVA phase required")
    return tuple(x[0][positions.long()][None] for x in (cos, sin))


def inverse_phase(keys, positions, cos, sin):
    c, s = selected_phases(positions, cos, sin)
    if keys.shape != (1, *positions.shape, 128) or keys.dtype != torch.bfloat16:
        raise ValueError("same-shape BF16 post-RoPE source keys required")
    x = keys.float()
    rotated = torch.cat((-x[..., 64:], x[..., :64]), -1)
    return (
        (x * c.float() - rotated * s.float()) / (c.float().square() + s.float().square())
    ).bfloat16()


def rotate_cpu(target, positions, cos, sin):
    """CPU oracle. Slice only prompt rows; generated history is untouched."""
    c, s = selected_phases(positions, cos, sin)
    n = positions.shape[1]
    x = target[..., :n, :].clone()
    rotated = torch.cat((-x[..., 64:], x[..., :64]), -1)
    target[..., :n, :].copy_(x * c + rotated * s)


@dataclass(frozen=True)
class PhaseFormat:
    base: HeadLocalFormat
    positions: object
    cos: object
    sin: object
    kind: str = "common_inverse_phase_prompt_precision_diagnostic"

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), self.positions)

    def shared(self):
        return self.cos, self.sin

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, destinations, *, direct):
        del attention, direct
        # Never scatter post-RoPE scalar patches before rotation: doing so would
        # also corrupt their unpatched partner features. Decode raw bodies first.
        for body, target in zip(self.base.base.pair, destinations, strict=True):
            if target.device.type == "cuda":
                from mosaickv.evaluation.precision_tier_kernel import decode

                decode(body, target, self.base.base.exact_map, body.codes, overlay=False)
            else:
                from mosaickv.evaluation.precision_tier import unpack_cpu

                target[..., : self.shape[-2], :].copy_(unpack_cpu(body))
        key = destinations[0]
        if key.device.type == "cuda":
            from mosaickv.evaluation.phase_capacity_kernel import rotate

            rotate(key, self.positions, self.cos, self.sin)
        else:
            rotate_cpu(key, self.positions, self.cos, self.sin)
        from mosaickv.evaluation.protected_exact import scatter_into

        for body, target in zip(self.base.base.exact_values, destinations, strict=True):
            scatter_into(body, self.base.base.exact_indices, target)
        if self.base.protected:
            for body, target in zip(self.base.protected, destinations, strict=True):
                if target.device.type == "cuda":
                    from mosaickv.evaluation.head_local_precision_kernel import scatter

                    scatter(body, self.base.flat_tokens, target, prompt_length=self.shape[-2])
                else:
                    from mosaickv.evaluation.exact_storage import decoded

                    ids = self.base.flat_tokens.long()
                    target[0, ids // self.shape[-2], ids % self.shape[-2]] = decoded(body)[0, 0]
        for patch, target in zip(
            self.base.base.corrections,
            destinations if self.base.base.corrections else (),
            strict=True,
        ):
            if target.device.type == "cuda":
                from mosaickv.evaluation.dual_projection_kernel import apply_patch

                apply_patch(patch, target[..., : self.shape[-2], :])
            else:
                ids = patch.indices.long()
                n, d = self.shape[-2:]
                target[0, ids // (n * d), ids // d % n, ids % d] = patch.values

    def materialize(self, attention):
        out = tuple(
            torch.empty(self.shape, device=self.positions.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, out, direct=True)
        return out


@torch.inference_mode()
def recode(packet, original_pair, positions, phases, *, key_bits, value_bits, pre_rotary):
    """Same selected/merged ACTUAL K/V; original exact protections stay intact.

    Retention-1 must bypass this function and use the original raw format.
    A TextSourceFormat continues to apply its actual post-RoPE text last.
    """
    if type(pre_rotary) is not bool or key_bits not in (2, 3) or value_bits not in (2, 3):
        raise ValueError("explicit registered two/three-bit diagnostic required")
    outer = packet if isinstance(packet, TextSourceFormat) else None
    local = outer.base if outer is not None else packet
    if not isinstance(local, HeadLocalFormat) or len(original_pair) != 2:
        raise ValueError("common head-local packet and actual selected pair required")
    ids = torch.tensor(positions, dtype=torch.int32, device=original_pair[0].device)
    cos, sin = phases
    selected_phases(ids, cos, sin)
    if any(tuple(t.shape) != local.shape or t.dtype != torch.bfloat16 for t in original_pair):
        raise ValueError("actual same-shape BF16 selected/merged K/V required")
    key = inverse_phase(original_pair[0], ids, cos, sin) if pre_rotary else original_pair[0]
    bodies = (pack(key, bits=key_bits, axis=2), pack(original_pair[1], bits=value_bits, axis=3))
    rewritten = replace(local, base=replace(local.base, pair=bodies))
    result = PhaseFormat(rewritten, ids, cos, sin) if pre_rotary else rewritten
    return replace(outer, base=result) if outer is not None else result
