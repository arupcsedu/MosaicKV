"""Higher-memory causal control: original BF16 K/V at identical memberships."""

import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16


@torch.inference_mode()
def selected_original(source, memberships):
    if len(source) != len(memberships) or not source:
        raise ValueError("complete original layers and memberships required")
    result = []
    for (key, value), heads in zip(source, memberships, strict=True):
        if (
            key.shape != value.shape
            or key.dtype != torch.bfloat16
            or value.dtype != key.dtype
            or key.ndim != 4
            or key.shape[0] != 1
            or len(heads) != key.shape[1]
            or len({len(h) for h in heads}) != 1
            or any(
                not h or tuple(h) != tuple(sorted(set(h))) or h[0] < 0 or h[-1] != key.shape[-2] - 1
                for h in heads
            )
        ):
            raise ValueError(
                "sorted unpadded BF16 memberships preserving mandatory last token required"
            )
        ids = torch.tensor(heads, device=key.device, dtype=torch.long)[None, :, :, None]
        ids = ids.expand(1, key.shape[1], len(heads[0]), key.shape[-1])
        pairs = tuple(t.gather(2, ids).contiguous() for t in (key, value))
        result.append(
            ExactFormat(
                "kv",
                tuple(pairs[0].shape),
                pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in pairs),
            )
        )
    return tuple(result)
