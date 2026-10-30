"""Label-free feasibility of exact K/V regeneration from shared projection input.

Not a cache backend, not low-rank fitting, and not a benchmark-quality method.
The full original GEMM shape is retained to test bitwise, not merely algebraic,
equivalence. Original selected positions and RoPE phases are unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_into


class ProjectionCapture:
    """Read-only hooks for one prefill in the installed Llama attention class."""

    def __init__(self, adapter: Any) -> None:
        self.adapter, self.stack = adapter, ExitStack()
        self.records: dict[int, dict[str, Any]] = {}

    def __enter__(self) -> ProjectionCapture:
        import transformers

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("projection capture audited only for Transformers 4.55.2")
        for index, layer in enumerate(self.adapter._language_layers()):
            module = layer.self_attn
            if (
                type(module).__name__ != "LlamaAttention"
                or type(module).__module__ != "transformers.models.llama.modeling_llama"
            ):
                raise NotImplementedError("projection capture requires pinned LlamaAttention")

            def before(_module: Any, args: Any, kwargs: Any, *, offset: int = index) -> None:
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                rotary = kwargs.get("position_embeddings")
                if offset in self.records or hidden is None or rotary is None:
                    raise ValueError("exactly one explicit full prefill per layer required")
                if hidden.ndim != 3 or hidden.shape[0] != 1 or not hidden.is_contiguous():
                    raise ValueError("contiguous batch-one normalized projection source required")
                self.records[offset] = dict(
                    normalized=hidden.detach().clone(),
                    cos=rotary[0].detach(),
                    sin=rotary[1].detach(),
                    pointer=hidden.data_ptr(),
                    k_calls=0,
                    v_calls=0,
                )

            handle = module.register_forward_pre_hook(before, with_kwargs=True)
            self.stack.callback(handle.remove)
            for name in ("k", "v"):

                def verify(
                    _module: Any, args: Any, *, offset: int = index, kind: str = name
                ) -> None:
                    record = self.records[offset]
                    if len(args) != 1 or args[0].data_ptr() != record["pointer"]:
                        raise ValueError(
                            "K/V projections did not consume the same normalized input"
                        )
                    record[f"{kind}_calls"] += 1

                handle = getattr(module, f"{name}_proj").register_forward_pre_hook(verify)
                self.stack.callback(handle.remove)
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stack.__exit__(*args)

    def validate(self) -> None:
        if len(self.records) != len(self.adapter._language_layers()) or any(
            r["k_calls"] != 1 or r["v_calls"] != 1 for r in self.records.values()
        ):
            raise ValueError("incomplete shared-source projection capture")


@dataclass(frozen=True)
class ProjectionArchive:
    source: PackedBF16
    union: Any
    selected: Any
    original_length: int
    cos: Any
    sin: Any

    def storage_tensors(self) -> tuple[Any, ...]:
        return (*self.source.buffers, self.union, self.selected, self.cos, self.sin)


def build_archive(record: dict[str, Any], positions: Any, *, encoded: bool) -> ProjectionArchive:
    import torch

    normalized = record["normalized"]
    if normalized.dtype != torch.bfloat16 or normalized.ndim != 3 or normalized.shape[0] != 1:
        raise ValueError("BF16 batch-one normalized input required")
    if (
        not positions
        or len({len(h) for h in positions}) != 1
        or any(
            not h or list(sorted(set(h))) != list(h) or h[0] < 0 or h[-1] >= normalized.shape[1]
            for h in positions
        )
    ):
        raise ValueError("uniform, sorted unique original per-head positions required")
    union = torch.tensor(
        sorted({p for head in positions for p in head}), dtype=torch.int64, device=normalized.device
    )
    if all(list(head) == list(positions[0]) for head in positions):
        # One true owner, not H duplicate maps. Stride-zero head views retain
        # the same ordinary gather semantics and are charged by backing.
        selected = union.unsqueeze(0).expand(len(positions), -1)
    else:
        selected = torch.tensor(positions, dtype=torch.int64, device=normalized.device)
    hidden = normalized.index_select(1, union).unsqueeze(1).contiguous()
    if encoded:
        if hidden.device.type == "cuda":
            from mosaickv.evaluation.lossless_bf16_kernel import encode

            body = encode(hidden)
        else:
            body = pack_cpu(hidden)
    else:
        body = PackedBF16(tuple(hidden.shape), True, (hidden,))
    return ProjectionArchive(
        body, union, selected, normalized.shape[1], record["cos"], record["sin"]
    )


def reconstruct_archive(archive: ProjectionArchive, attention: Any) -> tuple[Any, Any]:
    """Original full-shape frozen projections, installed RoPE, then exact gather."""
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    heads, kept = archive.selected.shape
    dimension = attention.head_dim
    width = archive.source.shape[-1]
    if (
        width != heads * dimension
        or attention.k_proj.out_features != width
        or attention.v_proj.out_features != width
    ):
        raise NotImplementedError("first feasibility path requires full multi-head attention")
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
        value = attention.v_proj(canvas).view(shape).transpose(1, 2)
        del canvas
        # The installed function supports different Q/K head counts (GQA).
        # Its unused Q result needs only one head; the K branch and its original
        # shape/phase/arithmetic are unchanged. Do not retain that unused output.
        unused_q, key = apply_rotary_pos_emb(key[:, :1], key, archive.cos, archive.sin)
        del unused_q
        index = archive.selected[None, :, :, None].expand(1, heads, kept, dimension)
        return key.gather(2, index), value.gather(2, index)
