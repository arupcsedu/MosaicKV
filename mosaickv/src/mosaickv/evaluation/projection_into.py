"""Original-shape source projection directly into a strided layer lease."""

from typing import Any, cast


def reconstruct_into(archive: Any, attention: Any, destinations: tuple[Any, ...]) -> None:
    import torch

    from mosaickv.evaluation.lossless_bf16 import unpack_into

    heads, kept = archive.selected.shape
    dimension, width = attention.head_dim, archive.source.shape[-1]
    if (
        width != heads * dimension
        or attention.k_proj.out_features != width
        or attention.v_proj.out_features != width
        or len(destinations) != 2
        or any(tuple(t.shape) != (1, heads, kept, dimension) for t in destinations)
    ):
        raise ValueError("full MHA and complete original-capacity destination pair required")
    with torch.inference_mode():
        decoded = torch.empty(
            archive.source.shape, dtype=torch.bfloat16, device=archive.union.device
        )
        unpack_into(archive.source, decoded)
        canvas = torch.zeros(
            (1, archive.original_length, width), dtype=torch.bfloat16, device=archive.union.device
        )
        canvas.index_copy_(1, archive.union, decoded.squeeze(1))
        del decoded
        shape = (1, archive.original_length, heads, dimension)
        key = attention.k_proj(canvas).view(shape).transpose(1, 2)
        index = archive.selected[None, :, :, None].expand(1, heads, kept, dimension)
        if key.device.type == "cuda":
            from mosaickv.evaluation.projection_rope_kernel import rope_gather

            rope_gather(key, archive.cos, archive.sin, archive.selected, destinations[0])
        else:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            unused, rotated = cast(Any, apply_rotary_pos_emb)(
                key[:, :1], key, archive.cos, archive.sin
            )
            torch.gather(rotated, 2, index, out=destinations[0])
            del unused, rotated
        del key
        value = attention.v_proj(canvas).view(shape).transpose(1, 2)
        del canvas
        torch.gather(value, 2, index, out=destinations[1])
