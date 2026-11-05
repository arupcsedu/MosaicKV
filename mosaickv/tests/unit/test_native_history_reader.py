from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import native_history_reader as mod
from mosaickv.evaluation.native_history_feasibility import canonical_device, workspace_buffers


def test_phase_tokens_support_inference_but_reject_replacements():
    with torch.inference_mode():
        tensor = torch.ones((1, 16, 128), dtype=torch.bfloat16)
    token = mod.phase_token(tensor)
    assert token[-1] == "inference_tensor_no_version_counter"
    assert token != mod.phase_token(tensor.clone())
    tracked = torch.ones((1, 16, 128), dtype=torch.bfloat16)
    before = mod.phase_token(tracked)
    tracked.add_(1)
    assert before != mod.phase_token(tracked)
    with pytest.raises(ValueError):
        mod.phase_guard(torch.randn((1, 16, 128)).bfloat16(), tracked, 16)


def test_borrowed_workspace_validation_and_implicit_cuda_device(monkeypatch):
    buffers = tuple(torch.empty((128, 4096), dtype=torch.bfloat16) for _ in range(2))
    assert workspace_buffers("cpu", buffers) is buffers
    for bad in ((buffers[0], buffers[0]), buffers[:1], (buffers[0].float(), buffers[1])):
        with pytest.raises(ValueError):
            workspace_buffers("cpu", bad)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert canonical_device("cuda") == torch.device("cuda:2")
    assert canonical_device("cuda:0") == torch.device("cuda:0")


def test_format_is_request_local_and_checks_tracked_phase_mutation():
    cos, sin = (torch.ones((1, 16, 128), dtype=torch.bfloat16) for _ in range(2))
    archive = SimpleNamespace(cos=cos, sin=sin)
    packet = mod.NativeHistoryFormat(
        "native_lossless_history_execution",
        (1, 32, 16, 128),
        archive=archive,
        phase_versions=(mod.phase_token(cos), mod.phase_token(sin)),
    )
    target = torch.empty(packet.shape, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="escape"):
        packet.fill_component(None, 0, target)
    calls = []
    state = SimpleNamespace(
        closed=False,
        projections=0,
        engine=SimpleNamespace(project=lambda *args: calls.append(args)),
    )
    token = mod._scope.set(state)
    try:
        packet.fill_component(None, 0, target)
        assert state.projections == 1 and len(calls) == 1
        cos.add_(1)
        with pytest.raises(RuntimeError, match="changed"):
            packet.fill_component(None, 0, target)
        state.closed = True
        with pytest.raises(RuntimeError, match="escape"):
            packet.fill_component(None, 0, target)
    finally:
        mod._scope.reset(token)


@pytest.mark.parametrize("borrowed", [False, True])
def test_common_workspace_is_charged_once_and_fully_retired(monkeypatch, borrowed):
    device = torch.device("cpu")
    original = (torch.empty(3),)
    buffers = tuple(torch.empty((128, 4096), dtype=torch.bfloat16) for _ in range(2))
    observed, closed = [], []

    class Engine:
        def __init__(self, actual_device, *, frozen, buffers=None):
            assert actual_device == device and frozen
            self.source, self.output = globals_buffers if buffers is None else buffers

        def close(self):
            closed.append(True)

    globals_buffers = buffers
    cache = SimpleNamespace(
        shared_tensors=original,
        shared_bytes=12,
        observe=lambda extra=(): observed.append(tuple(extra)),
    )
    pool = SimpleNamespace(
        cache=cache,
        closed=False,
        lt_engine=(SimpleNamespace(source=buffers[0], output=buffers[1]) if borrowed else None),
    )
    layer = SimpleNamespace(prompt_graph_pool=pool)
    monkeypatch.setattr(mod, "Engine", Engine)
    state = mod.State()
    state.bind(layer, device)
    assert state.extra_workspace == (0 if borrowed else 2**21)
    assert state.borrowed_workspace == (2**21 if borrowed else 0)
    assert len(cache.shared_tensors) == (1 if borrowed else 3)
    state.bind(layer)  # Same request reuses, never allocates a second workspace.
    with pytest.raises(RuntimeError, match="cross-request"):
        state.bind(SimpleNamespace(prompt_graph_pool=SimpleNamespace(closed=False)), device)
    state.retire(pool)
    state.retire()
    assert len(closed) == 1 and state.closed and state.engine is state.pool is None
    assert cache.shared_tensors == original
    assert cache.shared_bytes == 12


def test_default_scope_unchanged_and_enabled_must_retire():
    @mod.managed
    def call(*args, **kwargs):
        return None, None, {}

    assert call()[2]["common_native_history"] == dict(enabled=False, report=None)
    with pytest.raises(ValueError):
        call(native_history_reader=1)
    with pytest.raises(NotImplementedError):
        call(native_history_reader=True)
    with pytest.raises(RuntimeError, match="retire"):
        call(native_history_reader=True, prompt_graph_reader=True)
    assert mod.current() is None
