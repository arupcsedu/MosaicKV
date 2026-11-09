"""Bounded CUDA controls for schedule-only history-source reuse; no model weights."""

import weakref
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mosaickv.evaluation import mixed_history_projection as mixed
from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation import sequential_source_history as sequential
from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.history_source_precision import pack_source
from mosaickv.evaluation.native_history_feasibility import Engine, duplicated_phases, make_packet


def build(kind, rows):
    state = native.State()
    state.engine = Engine("cuda", frozen=True)
    theta = torch.randn(1, rows, 64, device="cuda")
    cos, sin = duplicated_phases(theta)
    if kind == "native":
        body = make_packet(torch.randn(1, 1, rows, 4096, device="cuda").bfloat16(), cos, sin)
        packet = native.NativeHistoryFormat(
            **{f.name: getattr(body, f.name) for f in fields(body)},
            phase_versions=(native.phase_token(cos), native.phase_token(sin)),
        )
    else:
        old_rows = max(0, rows - 64)
        older = (
            SimpleNamespace(
                shape=(1, 32, old_rows, 128),
                body=pack_source(
                    torch.randn(1, 1, old_rows, 4096, device="cuda").bfloat16(), group=32
                ),
            )
            if old_rows
            else None
        )
        recent = tuple(
            SimpleNamespace(
                shape=(1, 32, 16, 128),
                archive=SimpleNamespace(
                    source=encode(torch.randn(1, 1, 16, 4096, device="cuda").bfloat16())
                ),
            )
            for _ in range((rows - old_rows) // 16)
        )
        packet = mixed.MixedHistory(
            older,
            recent,
            cos,
            sin,
            (),
            weakref.ref(state),
            (1, 32, rows, 128),
            (native.phase_token(cos), native.phase_token(sin)),
        )

    class Layer(sequential.SequentialSourceHistoryLayer):
        def get_seq_length(self):
            return self.prompt_length + rows + self.tail_used

    layer = Layer.__new__(Layer)
    layer.prompt_length, layer.tail_used = 3, 1
    layer.representation = SimpleNamespace(shape=(1, 32, 3, 128))
    layer.source_sealed = [packet]
    layer.attention = SimpleNamespace(
        k_proj=torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    layer.history = tuple(
        torch.full((1, 32, 16, 128), 11 + c, device="cuda", dtype=torch.bfloat16) for c in (0, 1)
    )
    lease = torch.empty((1, 32, rows + 4, 128), device="cuda", dtype=torch.bfloat16)
    layer.leased, layer.current_component = (lease, lease), None
    fill = sequential.fill_component

    def fill_prompt(packet, attention, component, target, **kwargs):
        if packet is layer.representation:
            # Explicitly simulate the real aliasing hazard; V must precede this overwrite.
            state.engine.source.fill_(91 + component)
            target[..., :3, :].fill_(7 + component)
        else:
            fill(packet, attention, component, target, **kwargs)

    def scope(stack):
        stack.enter_context(patch.object(native, "current", lambda: state))
        stack.enter_context(patch.object(sequential, "fill_component", fill_prompt))

    return layer, state, scope
