"""Lossless exact-text storage inside the explicitly lossy precision variant."""

from typing import Any

import torch

from mosaickv.evaluation.exact_storage import decoded


def scatter_into(body: Any, positions: Any, target: Any) -> None:
    if (
        len(body.shape) != 4
        or body.shape[:2] != tuple(target.shape[:2])
        or body.shape[-1] != target.shape[-1]
        or body.shape[-2] != positions.numel()
        or body.shared_literal_window
        or positions.dtype != torch.int32
        or target.dtype != torch.bfloat16
        or target.stride(-1) != 1
        or target.stride(-2) != target.shape[-1]
    ):
        raise ValueError(
            "independent protected BF16 packet and original strided positions required"
        )
    if target.device.type == "cpu":
        if not bool(((positions >= 0) & (positions < target.shape[-2])).all()):
            raise ValueError("protected positions out of bounds")
        target[..., positions.long(), :] = decoded(body)
    else:
        from mosaickv.evaluation.protected_exact_kernel import scatter

        scatter(body, positions, target)


def gpu_controls() -> list[dict[str, Any]]:
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode

    words = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    data = words.view(torch.bfloat16).reshape(1, 2, 256, 128)
    positions = torch.arange(256, device="cuda", dtype=torch.int32) * 2 + 3
    records = []
    for raw in (False, True):
        body = (
            PackedBF16(tuple(data.shape), True, (data.clone(),))
            if raw
            else encode(data, raw_fallback=False)
        )
        target = torch.full((1, 2, 521, 128), -5, device="cuda", dtype=torch.bfloat16)
        scatter_into(body, positions, target)
        if not torch.equal(
            target[..., positions.long(), :].view(torch.int16), data.view(torch.int16)
        ):
            raise RuntimeError("protected exact scatter changed an original BF16 bit")
        mask = torch.ones(521, device="cuda", dtype=torch.bool)
        mask[positions.long()] = False
        if not bool((target[..., mask, :] == -5).all()):
            raise RuntimeError("protected scatter overwrote non-protected source/history positions")
        records.append(dict(raw=raw, all_65536_patterns_bitwise=True, nonprotected_unchanged=True))
    return records
