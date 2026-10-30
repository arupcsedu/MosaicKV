"""Opt-in, bit-exact source-sealed history for bounded 128-token diagnostics.

Raw K/V tails are retained. Their normalized-input tails and actual rotary
phases are additional, explicitly charged storage, not free capture tensors.
"""

from contextlib import ExitStack, nullcontext
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import ExactCache
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity
from mosaickv.evaluation.sealed_history import SealedHistoryLayer


class SourceHistoryLayer(SealedHistoryLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.normalized_tail: Any = None
        self.source_sealed: list[Any] = []
        self.source_pending = False

    def owned_tensors(self) -> tuple[Any, ...]:
        return (
            *super().owned_tensors(),
            *((self.normalized_tail,) if self.normalized_tail is not None else ()),
            *(t for format in self.source_sealed for t in format.tensors()),
        )

    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        if not self.source_pending or self.tail_used == 16:
            raise ValueError("one captured source and coordinated sealing required")
        result = super().update(key_states, value_states, cache_kwargs)
        self.source_pending = False
        return result

    def fill_history(self) -> None:
        offset = self.prompt_length
        for format in self.source_sealed:
            end = offset + format.shape[-2]
            targets = tuple(t[..., offset:end, :] for t in self.leased)
            format.fill(self.attention, targets, direct=True)
            offset = end
        for tail, target in zip(self.history, self.leased, strict=True):
            target[..., offset : offset + self.tail_used, :].copy_(tail[..., : self.tail_used, :])
        if offset + self.tail_used != self.get_seq_length():
            raise RuntimeError("source history dropped or duplicated a token")


class SourceHistoryCache(ExactCache):
    def __init__(self, *, audit_workspace: bool) -> None:
        super().__init__(audit_workspace=audit_workspace)
        self.shared_tensors: tuple[Any, ...] = ()
        self.source_history_seals = 0
        self.phase_tail: Any = None
        self.phase_identity: Any = None
        self.history_position_chunks: list[Any] = []

    def source_input(self, index: int, hidden: Any, rotary: Any, positions: Any) -> None:
        layer = self.layers[index]
        if (
            not isinstance(layer, SourceHistoryLayer)
            or layer.leased
            or layer.source_pending
            or layer.tail_used >= 16
            or hidden.dtype != torch.bfloat16
            or hidden.shape != (1, 1, layer.prompt_shape[1] * 128)
            or not hidden.is_contiguous()
            or positions.numel() != 1
            or any(t.shape != (1, 1, 128) or t.dtype != torch.bfloat16 for t in rotary)
        ):
            raise ValueError("released one-token full-MHA source/phase transaction required")
        token_identity = tuple(
            (t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in (*rotary, positions)
        )
        slot = layer.tail_used
        if index == 0:
            if self.phase_tail is None:
                phase = tuple(
                    torch.empty((1, 16, 128), dtype=t.dtype, device=t.device) for t in rotary
                )
                logical = torch.empty(16, dtype=positions.dtype, device=positions.device)
                self.observe((*phase, logical))
                self.phase_tail = (*phase, logical)
                self.shared_tensors = (*self.shared_tensors, *self.phase_tail)
                self.shared_bytes = backing_bytes(self.shared_tensors)
            self.phase_identity = token_identity
            for source, target in zip(rotary, self.phase_tail[:2], strict=True):
                target[:, slot : slot + 1].copy_(source)
            self.phase_tail[2][slot : slot + 1].copy_(positions.reshape(-1))
        elif token_identity != self.phase_identity:
            raise ValueError("all layers must use the identical original rotary/position tensors")
        if layer.normalized_tail is None:
            normal = torch.empty(
                (1, 16, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device
            )
            self.observe((normal,))
            layer.normalized_tail = normal
            layer.resident_bytes += backing_bytes((normal,))
        layer.normalized_tail[:, slot : slot + 1].copy_(hidden)
        layer.source_pending = True
        self.observe()

    def storage_record(self) -> dict[str, Any]:
        record = super().storage_record()
        record.update(
            source_history=True,
            source_history_seals=self.source_history_seals,
            source_history_normalized_tail_bytes=backing_bytes(
                tuple(
                    layer.normalized_tail
                    for layer in self.layers
                    if layer.normalized_tail is not None
                )
            ),
            generated_history_storage=[
                dict(sealed_chunks=len(layer.source_sealed), raw_tail_used=layer.tail_used)
                for layer in self.layers
            ],
            source_history_original_positions=[t.tolist() for t in self.history_position_chunks],
            source_history_constructor_tensor_results_observed=self.audit_workspace,
            whole_request_peak_certified=False,
        )
        return record


class SourceInputTransport:
    def __init__(self, adapter: Any, cache: SourceHistoryCache) -> None:
        self.adapter, self.cache, self.stack = adapter, cache, ExitStack()

    def __enter__(self) -> "SourceInputTransport":
        import transformers

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("source history audited for Transformers 4.55.2 only")
        try:
            for index, layer in enumerate(self.adapter._language_layers()):
                module = layer.self_attn
                if (
                    type(module).__name__ != "LlamaAttention"
                    or type(module).__module__ != "transformers.models.llama.modeling_llama"
                ):
                    raise NotImplementedError("pinned LlamaAttention required")

                def before(_module, args, kwargs, *, offset=index):
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    if (
                        kwargs.get("past_key_value") is not self.cache
                        or hidden is None
                        or kwargs.get("position_embeddings") is None
                        or kwargs.get("position_ids") is None
                    ):
                        raise ValueError(
                            "request-local cache, original source/phase/positions required"
                        )
                    self.cache.source_input(
                        offset, hidden, kwargs["position_embeddings"], kwargs["position_ids"]
                    )

                handle = module.register_forward_pre_hook(before, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stack.__exit__(*args)


def build_chunk(layer: Any, cos: Any, sin: Any) -> Any:
    from mosaickv.evaluation.dual_projection import build_dual
    from mosaickv.evaluation.encoded_projection import build
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    union = torch.arange(16, device=cos.device, dtype=torch.int32)
    selected = union[None].expand(layer.prompt_shape[1], -1)
    archive = ProjectionArchive(
        encode(layer.normalized_tail[:, None].contiguous()), union, selected, 16, cos, sin
    )
    base = ExactFormat("source", tuple(layer.history[0].shape), archive=archive)
    dual, _ = build_dual(base, layer.attention, layer.history)
    candidate, _ = build(dual, layer.attention)
    return candidate


def seal_source_full_tails(cache: SourceHistoryCache) -> bool:
    layers = cache.layers
    if (
        not layers
        or not all(isinstance(layer, SourceHistoryLayer) for layer in layers)
        or len({layer.history_length for layer in layers}) != 1
        or any(layer.leased or layer.source_pending for layer in layers)
    ):
        raise ValueError("complete synchronized released source-history layers required")
    if not any(layer.tail_used == 16 for layer in layers):
        return False
    if not all(layer.tail_used == 16 for layer in layers) or cache.phase_tail is None:
        raise ValueError("all history tails must be complete before source sealing")
    excluded = {identity(t) for layer in layers for t in layer.owned_tensors()}
    excluded.update(identity(t) for t in cache.shared_tensors)
    excluded.update(owner for layer in layers for owner in layer.weights)
    pending: list[Any] = []

    def observe(extras=()):
        cache.observe((*pending, *extras))

    mode = ScratchAudit(excluded=excluded, observe=observe) if cache.audit_workspace else None
    formats = []
    with mode if mode is not None else nullcontext():
        cos, sin, positions = tuple(t.clone() for t in cache.phase_tail)
        if not torch.equal(positions[1:], positions[:-1] + 1) or (
            cache.history_position_chunks
            and int(positions[0]) != int(cache.history_position_chunks[-1][-1]) + 1
        ):
            raise ValueError("original logical history positions must remain consecutive")
        pending.extend((cos, sin, positions))
        for layer in layers:
            format = build_chunk(layer, cos, sin)
            if any(identity(t) in excluded for t in format.private()):
                raise RuntimeError("immutable source history aliases a mutable existing owner")
            formats.append(format)
            pending.extend(format.tensors())
            observe()
    # Only commit after every layer's original K/V reconstruction has passed.
    cache.shared_tensors = (*cache.shared_tensors, cos, sin, positions)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    cache.history_position_chunks.append(positions)
    for layer, format in zip(layers, formats, strict=True):
        layer.source_sealed.append(format)
        layer.resident_bytes += backing_bytes(format.private())
        layer.tail_used = 0
    cache.source_history_seals += 1
    if mode is not None:
        layers[0].workspace_operations += mode.operations
    pending.clear()
    cache.observe()
    return True
