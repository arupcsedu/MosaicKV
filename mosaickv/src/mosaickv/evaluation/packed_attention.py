"""Explicitly experimental lossless-codec attention boundary and CPU oracle."""

from dataclasses import dataclass
from typing import Any

from mosaickv.evaluation.lossless_bf16 import PackedBF16, unpack_cpu
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class AttentionPacket:
    payload: PackedBF16
    checkpoints: Any

    def tensors(self) -> tuple[Any, ...]:
        return (*self.payload.buffers, self.checkpoints)

    @property
    def storage_bytes(self) -> int:
        return backing_bytes(self.tensors())


def packet(payload: PackedBF16) -> AttentionPacket:
    import torch

    if (
        len(payload.shape) != 4
        or payload.shape[0] != 1
        or payload.shape[-1] != 128
        or any(type(x) is not int or x <= 0 for x in payload.shape)
    ):
        raise ValueError("positive batch-one BHN128 packet required")
    if payload.shared_literal_window:
        raise ValueError("shared literal-window packets are not audited for this path")
    device = payload.buffers[0].device
    if payload.raw:
        checkpoints = torch.empty(0, dtype=torch.int16, device=device)
    elif device.type == "cuda":
        from mosaickv.evaluation.encoded_projection_kernel import checkpoints as build

        checkpoints = build(payload)
    else:
        # Independent decoded-exponent counting is a CPU correctness oracle.
        codes = payload.buffers[1].reshape(-1, 3).to(torch.int32)
        words = codes[:, 0] | (codes[:, 1] << 8) | (codes[:, 2] << 16)
        values = ((words[:, None] >> (3 * torch.arange(8))) & 7).reshape(-1, 1024)
        offsets = (values == 7).reshape(-1, 8, 128).sum(-1)
        checkpoints = (offsets.cumsum(-1) - offsets).to(torch.int16).reshape(-1)
    return AttentionPacket(payload, checkpoints)


def attend(
    query: Any,
    key: AttentionPacket,
    value: AttentionPacket,
    history: tuple[Any, Any],
    *,
    used: int,
    scaling: float,
) -> tuple[Any, Any]:
    import math

    import torch

    shape = key.payload.shape
    if (
        value.payload.shape != shape
        or query.shape != (1, shape[1], 1, 128)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or type(used) is not int
        or used < 0
        or len(history) != 2
        or not math.isfinite(scaling)
        or scaling <= 0
    ):
        raise ValueError("finite positive scaling, one BF16 MHA query and matched packets required")
    if any(t.device != query.device for p in (key, value) for t in p.tensors()):
        raise ValueError("packets must share the query device")
    for t in history:
        if (
            t.ndim != 4
            or t.shape[:2] != shape[:2]
            or t.shape[-1] != 128
            or t.shape[-2] < used
            or t.dtype != query.dtype
            or t.device != query.device
            or not t.is_contiguous()
            or t.shape != history[0].shape
        ):
            raise ValueError("compatible contiguous BF16 history and bounded used length required")
    if query.device.type == "cuda":
        from mosaickv.evaluation.packed_attention_kernel import attention

        return attention(query, key, value, history, used, scaling)
    keys = torch.cat((unpack_cpu(key.payload), history[0][..., :used, :]), -2)
    values = torch.cat((unpack_cpu(value.payload), history[1][..., :used, :]), -2)
    scores = query @ keys.transpose(-2, -1) * scaling
    probabilities = torch.softmax(scores, -1, dtype=torch.float32).to(query.dtype)
    return (probabilities @ values).transpose(1, 2).contiguous(), probabilities
