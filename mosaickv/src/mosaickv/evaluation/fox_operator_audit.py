"""Read-only operator localization; instrumented diagnostics, never latency data."""

from __future__ import annotations

from contextlib import contextmanager


def difference(reference, candidate):
    import torch

    if reference.shape != candidate.shape:
        raise ValueError("operator query slices have different shapes")
    a, b = reference.double(), candidate.double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite operator capture")
    error = (a - b).abs()
    absolute = float(error.max())
    norm = max(1.0, float(a.abs().max()))
    return dict(
        absolute_linf=absolute,
        reference_linf=float(a.abs().max()),
        normalized_linf=absolute / norm,
        rmse=float(error.square().mean().sqrt()),
        bitwise_equal=bool(torch.equal(reference, candidate)),
        shape=list(reference.shape),
    )


class OperatorCapture:
    """Capture only the last query row, not prompt/media/reference contents."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.values = {}
        self.order = []

    def clear(self):
        self.values.clear()
        self.order.clear()

    def save(self, name, tensor):
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"unsupported operator output: {name}")
        if name in self.values:
            raise ValueError(f"operator executed more than once: {name}")
        self.values[name] = tensor[:, -1:, :].detach().cpu().clone()
        self.order.append(name)

    @contextmanager
    def install(self):
        handles = []

        def hook(name, *, output=True):
            def record(module, inputs, result):
                value = result if output else inputs[0]
                if isinstance(value, tuple):
                    value = value[0]
                self.save(name, value)

            return record

        try:
            layers = tuple(self.adapter._language_layers())
            for index, layer in enumerate(layers):
                for suffix in (
                    "input_layernorm",
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "post_attention_layernorm",
                    "mlp.down_proj",
                ):
                    module = layer
                    for part in suffix.split("."):
                        module = getattr(module, part)
                    name = f"layer.{index}.{suffix}"
                    handles.append(
                        module.register_forward_hook(hook(name + ".input", output=False))
                    )
                    handles.append(module.register_forward_hook(hook(name + ".output")))
                handles.append(layer.register_forward_hook(hook(f"layer.{index}.output")))
            owners = [
                module
                for module in self.adapter.model.modules()
                if getattr(module, "layers", None) is not None
                and tuple(module.layers) == layers
                and hasattr(module, "norm")
            ]
            if len(owners) != 1:
                raise NotImplementedError("expected one Llama/Mistral decoder final norm")
            for name, module in (
                ("final_norm", owners[0].norm),
                ("lm_head", self.adapter.model.get_output_embeddings()),
            ):
                handles.append(module.register_forward_hook(hook(name + ".input", output=False)))
                handles.append(module.register_forward_hook(hook(name + ".output")))
            yield self
        finally:
            for handle in handles:
                handle.remove()


def compare_captures(reference: OperatorCapture, candidate: OperatorCapture):
    if reference.order != candidate.order:
        raise ValueError("different operator execution paths")
    rows = [
        dict(operator=name, **difference(reference.values[name], candidate.values[name]))
        for name in reference.order
    ]
    first = next((r["operator"] for r in rows if not r["bitwise_equal"]), None)
    return dict(first_nonidentical_operator=first, operators=rows)


def padding_localization(adapter, original, first_token, *, steps=16):
    import torch

    from mosaickv.evaluation.fox_numerics import _step, independent_state

    layers = adapter._legacy_layers(original.past_key_values)[0]
    positions = {
        (i, h): tuple(range(k.shape[2]))
        for i, (k, _) in enumerate(layers)
        for h in range(k.shape[1])
    }
    full, masks = independent_state(adapter, original, positions, original_layout=True)
    padded, pmasks = independent_state(
        adapter,
        original,
        positions,
        original_layout=True,
        padding=max(1, original.active_cache_length // 3),
    )
    token = torch.tensor([[int(first_token)]], device=adapter.device, dtype=torch.long)
    records = []
    for step in range(steps):
        history = []
        a_layers = adapter._legacy_layers(full.past_key_values)[0]
        b_layers = adapter._legacy_layers(padded.past_key_values)[0]
        n = original.active_cache_length
        extra = max(1, n // 3)
        for layer, ((ak, av), (bk, bv)) in enumerate(zip(a_layers, b_layers, strict=True)):
            ids = [*range(n), *range(n + extra, n + extra + step)]
            history.append(
                dict(
                    layer=layer,
                    key_bitwise_equal=bool(torch.equal(ak, bk[:, :, ids])),
                    value_bitwise_equal=bool(torch.equal(av, bv[:, :, ids])),
                )
            )
        a, b = OperatorCapture(adapter), OperatorCapture(adapter)
        with a.install():
            left = _step(adapter, token, full, masks)
        with b.install():
            right = _step(adapter, token, padded, pmasks)
        if full.next_decode_position != padded.next_decode_position:
            raise AssertionError("logical positions diverged")
        records.append(
            dict(
                step=step,
                logical_next_position=full.next_decode_position,
                full_physical_length=full.active_cache_length,
                padded_physical_length=padded.active_cache_length,
                logits=difference(left.logits.cpu(), right.logits.cpu()),
                cached_valid_history=history,
                **compare_captures(a, b),
            )
        )
        token = left.next_token_id
    return records


def boundary_localization(adapter, prepared):
    from mosaickv.evaluation.fox_boundary import hold_back_last_text_token

    prefix, held = hold_back_last_text_token(prepared)
    a, b = OperatorCapture(adapter), OperatorCapture(adapter)
    with a.install():
        full = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    state = adapter.prefill(prefix, capture_queries=False, capture_attentions=False)
    with b.install():
        chunk = adapter.decode_one_token(
            held, state.state, capture_queries=False, capture_attentions=False
        )
    return dict(logits=difference(full.logits.cpu(), chunk.logits.cpu()), **compare_captures(a, b))
