"""Reuse an already-owned exact epilogue graph for the parent counterfactual."""

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def replay(transport, layer, residual, heads, active_hidden):
    """Return independent active hidden and the borrowed parent-output view.

    The next ordinary layer call overwrites the borrowed view. Callers must
    finish their decision before advancing. No K/V lease or source is touched,
    and no graph, weights or persistent workspace are allocated here.
    """
    w = transport.workspace
    if (
        transport.closed
        or transport.calls < 32
        or transport.calls % 32
        or len(transport.layers) != 32
        or layer is not transport.layers[31]
        or layer.self_attn.layer_idx != 31
        or 31 not in transport.entries
        or any(t.shape != (1, 1, 4096) for t in (residual, heads, active_hidden))
        or any(
            t.dtype != torch.bfloat16 or t.device != w.output.device
            for t in (residual, heads, active_hidden)
        )
        or residual.untyped_storage().data_ptr() == w.output.untyped_storage().data_ptr()
    ):
        raise ValueError("completed actual final-layer graph and independent residual required")
    order = transport.calls, transport.replays, transport.captures
    saved = active_hidden.clone()
    attention = layer.self_attn.o_proj(heads)
    w.fill(residual, attention)
    transport.entries[31].replay()
    if order != (transport.calls, transport.replays, transport.captures):
        raise RuntimeError("counterfactual changed ordinary layer ordering")
    transport.guard_counterfactual_replays = (
        getattr(transport, "guard_counterfactual_replays", 0) + 1
    )
    return saved, w.output.view(1, 1, -1), backing_bytes((saved,))
