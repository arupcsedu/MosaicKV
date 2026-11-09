import weakref
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import mixed_history_projection as mixed
from mosaickv.evaluation import native_history_feasibility as engine
from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation import sequential_source_history as sequential
from mosaickv.evaluation.history_value_first import use
from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_into


def fixture(monkeypatch, rows=32, kind="mixed"):
    class State:
        closed, graphs, projections = False, None, 0

    state = State()

    def execute(source, weight, output, n):
        output[:n].copy_(source[:n] * weight)

    state.engine = SimpleNamespace(
        source=torch.empty(128, 4096, dtype=torch.bfloat16),
        output=torch.empty(128, 4096, dtype=torch.bfloat16),
        plans=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(native, "current", lambda: state)
    monkeypatch.setattr("mosaickv.evaluation.shared_source_kernel.rotate_native", lambda *a: None)
    chunks = tuple(
        SimpleNamespace(
            shape=(1, 32, 16, 128),
            archive=SimpleNamespace(source=pack_cpu(torch.randn(1, 1, 16, 4096).bfloat16())),
        )
        for _ in range(rows // 16)
    )
    cos = torch.ones(1, rows, 128).bfloat16()
    sin = torch.zeros_like(cos)
    packet = mixed.MixedHistory(
        None,
        chunks,
        cos,
        sin,
        (),
        weakref.ref(state),
        (1, 32, rows, 128),
        (native.phase_token(cos), native.phase_token(sin)),
    )
    if kind == "native":
        packet = native.NativeHistoryFormat(
            "encoded_projection",
            (1, 32, rows, 128),
            archive=SimpleNamespace(
                source=pack_cpu(torch.randn(1, 1, rows, 4096).bfloat16()), cos=cos, sin=sin
            ),
            phase_versions=(native.phase_token(cos), native.phase_token(sin)),
        )

        def unpack(packet, target):
            unpack_into(packet.archive.source, target[:rows].view(1, 1, rows, 4096))

        def project(packet, attention, c, target, repairs=None):
            engine.unpack_source(packet, state.engine.source)
            state.engine.plans.execute(
                state.engine.source,
                (attention.k_proj, attention.v_proj)[c].weight,
                state.engine.output,
                rows,
            )
            target[0].copy_(state.engine.output[:rows].view(rows, 32, 128).transpose(0, 1))

        monkeypatch.setattr(engine, "unpack_source", unpack)
        state.engine.project = project

    class Layer(sequential.SequentialSourceHistoryLayer):
        def get_seq_length(self):
            return self.prompt_length + rows + self.tail_used

    layer = Layer.__new__(Layer)
    layer.prompt_length, layer.tail_used = 3, 1
    layer.source_sealed = [packet]
    layer.representation = SimpleNamespace(shape=(1, 32, 3, 128))
    layer.attention = SimpleNamespace(
        k_proj=SimpleNamespace(weight=1), v_proj=SimpleNamespace(weight=2)
    )
    layer.history = tuple(torch.full((1, 32, 16, 128), 11 + c).bfloat16() for c in (0, 1))
    lease = torch.empty((1, 32, rows + 4, 128), dtype=torch.bfloat16)
    layer.leased = lease, lease
    layer.current_component = None
    fill = sequential.fill_component
    events = []

    def fill_prompt(packet, attention, component, target, **kwargs):
        if packet is layer.representation:
            events.append("prompt" + str(component))
            state.engine.source.fill_(91 + component)  # Real overwrite that prevented reuse.
            target[..., :3, :].fill_(7 + component)
        else:
            events.append("history" + str(component))
            fill(packet, attention, component, target, **kwargs)

    monkeypatch.setattr(sequential, "fill_component", fill_prompt)
    return layer, state, events


@torch.inference_mode()
@pytest.mark.parametrize("rows", [16, 32, 64])
@pytest.mark.parametrize("kind", ["mixed", "native"])
def test_reorders_only_history_value_write_and_preserves_all_slots(monkeypatch, rows, kind):
    layer, state, events = fixture(monkeypatch, rows, kind)
    expected = []
    for c in (0, 1):
        layer.materialize_component(c)
        expected.append(layer.leased[0].clone())
    events.clear()
    with use() as report:
        for c in (0, 1):
            layer.materialize_component(c)
            assert torch.equal(layer.leased[0].view(torch.int16), expected[c].view(torch.int16))
    assert events == ["prompt0", "history0", "history1", "prompt1"]
    assert report["skipped_source_unpacks"] == report["early_values"] == 1
    assert report["skipped_duplicate_history_fills"] == 1
    assert report["retained_tensor_owners"] == report["additional_GPU_storage_bytes"] == 0
    assert state.projections == 4  # Both original K and V projections still execute.


@torch.inference_mode()
@pytest.mark.parametrize(
    "case", ["foreign_layer", "foreign_request", "source_change", "missing_value", "alias"]
)
def test_foreign_incomplete_or_mutated_transactions_rejected(monkeypatch, case):
    layer, state, _ = fixture(monkeypatch)
    with pytest.raises(RuntimeError), use():
        if case == "alias":
            layer.leased = (state.engine.source.view(1, 32, 128, 128),) * 2
        layer.materialize_component(0)
        if case == "source_change":
            state.engine.source = torch.empty_like(state.engine.source)
        elif case == "foreign_request":
            monkeypatch.setattr(native, "current", lambda: None)
        elif case == "foreign_layer":
            layer.source_sealed = []
        elif case == "missing_value":
            return
        layer.materialize_component(1)


@torch.inference_mode()
def test_disabled_and_scope_restoration_preserve_original_order(monkeypatch):
    layer, _, events = fixture(monkeypatch)
    original_fill = sequential.fill_component
    original_method = sequential.SequentialSourceHistoryLayer.materialize_component
    with use(False) as report:
        assert sequential.fill_component is original_fill
        assert sequential.SequentialSourceHistoryLayer.materialize_component is original_method
        for c in (0, 1):
            layer.materialize_component(c)
    assert events == ["prompt0", "history0", "prompt1", "history1"]
    assert report["early_values"] == 0
    events.clear()
    for _ in range(2):
        with use() as record:
            with pytest.raises(ValueError, match="nonnested"), use():
                pass
            for c in (0, 1):
                layer.materialize_component(c)
        assert record["early_values"] == 1


@torch.no_grad()
def test_checked_tensor_version_rejects_an_in_place_source_overwrite(monkeypatch):
    layer, state, _ = fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="preceding private key"), use():
        layer.materialize_component(0)
        state.engine.source.fill_(123)
        layer.materialize_component(1)
