from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import joint_incoming_projection as joint
from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool


def fixture(monkeypatch, *, native_rows=0):
    pool = PromptGraphPool(SimpleNamespace(layers=[]))
    pool.prepared_replay = True
    pool.buffer = torch.zeros(1, 2, 9, 128, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        leased=(pool.buffer, pool.buffer),
        representation=object(),
        attention=object(),
        history_length=1,
    )
    signature = (
        id(layer.representation),
        id(layer.attention),
        pool.buffer.data_ptr(),
        pool.buffer.stride(),
    )
    seen = []
    pool.entries[id(layer), 0] = signature, SimpleNamespace(replay=lambda: seen.append(1))
    pool.entry_metadata[id(layer), 0] = native_rows, False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return pool, layer, seen


@pytest.mark.parametrize("native_rows,dispatches", [(0, 0), (600, 5), (604, 5)])
def test_prepared_plan_keeps_original_graph_and_counters_without_rewalking_formats(
    monkeypatch, native_rows, dispatches
):
    from mosaickv.evaluation import lt_source_format

    pool, layer, seen = fixture(monkeypatch, native_rows=native_rows)
    monkeypatch.setattr(
        lt_source_format,
        "source_format",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("warm replay should not re-walk format wrappers")
        ),
    )
    pool.reconstruct(layer, 0, pool.buffer)
    assert seen == [1] and pool.replays == pool.prepared_replay_hits == 1
    assert pool.lt_matmul_dispatches == dispatches


@pytest.mark.parametrize("corruption", ["packet", "metadata", "capture", "closed", "lease"])
def test_prepared_plan_preserves_all_original_binding_and_lease_failures(monkeypatch, corruption):
    pool, layer, seen = fixture(monkeypatch)
    if corruption == "packet":
        layer.representation = object()
    elif corruption == "metadata":
        pool.entry_metadata.clear()
    elif corruption == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif corruption == "closed":
        pool.closed = True
    else:
        layer.leased = None
    with pytest.raises(RuntimeError):
        pool.reconstruct(layer, 0, pool.buffer)
    assert not seen


def test_captured_code_epoch_still_advances_exactly_once_before_fast_replay(monkeypatch):
    from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat

    pool, layer, seen = fixture(monkeypatch)
    leaf, advances = layer.representation, []
    replay = SimpleNamespace(
        prepared_views=(leaf,),
        graph_bindings=((object(),),),
        advance_for_graph=lambda index: advances.append(index) or leaf,
    )
    layer.representation = BoundDeltaFormat(None, replay, 0)
    from mosaickv.evaluation.prompt_graph_policy import PromptGraphPolicy

    pool.policy = PromptGraphPolicy(
        history_arena=True, captured_code_replay=True, row_major_codes=False
    )
    pool.reconstruct(layer, 0, pool.buffer)
    assert seen == [1] and advances == [0]
    assert pool.captured_code_replays == pool.captured_code_kernel_dispatches == 1
    with pytest.raises(RuntimeError, match="only once"):
        pool.reconstruct(layer, 0, pool.buffer)
    assert seen == [1] and advances == [0]


@torch.inference_mode()
def test_consuming_preproduced_prompt_still_fills_actual_history_without_second_reconstruct():
    from test_joint_value_transaction import layer as make_layer

    state = make_layer()
    target = torch.full((1, 2, 6, 128), 7.0, dtype=torch.bfloat16)
    state.leased = target, target
    state.tail_used = state.history_length = 1
    state.history = tuple(torch.full((1, 2, 16, 128), -3.0, dtype=torch.bfloat16) for _ in range(2))
    owner = SimpleNamespace(mode="keys", take=lambda *args: True)
    pool = SimpleNamespace(
        joint_projection=owner,
        prepared_replay=True,
        prepared_consume_bypasses=0,
        reconstruct=lambda *args: (_ for _ in ()).throw(AssertionError("duplicate prompt replay")),
    )
    state.prompt_graph_pool = pool
    state.materialize_component(0)
    assert torch.equal(target[..., :5, :], torch.full_like(target[..., :5, :], 7.0))
    assert torch.equal(target[..., 5:, :], torch.full_like(target[..., 5:, :], -3.0))
    assert pool.prepared_consume_bypasses == 1 and state.current_component == 0
    state.release()


def test_the_common_policy_is_request_private_and_boolean_only():
    observed = []

    @joint.managed
    def request(**kwargs):
        observed.append(joint.reuse_graph_plan())
        return None, None, {}

    request(reuse_graph_replay_plan=True)
    request()
    assert observed == [True, False] and not joint.reuse_graph_plan()
    with pytest.raises(ValueError, match="boolean"):
        request(reuse_graph_replay_plan=1)
