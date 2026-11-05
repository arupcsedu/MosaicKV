from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import joint_incoming_projection as joint


@pytest.mark.parametrize("invalid", [True, None, "values", "all", 0])
def test_unregistered_or_implicit_execution_policies_are_rejected(invalid):
    with pytest.raises(ValueError):
        joint.validate(invalid)


def test_exact_once_joint_transaction_over_multiple_layers_and_updates():
    ledger = joint.Ledger(3)
    for update in range(5):
        for layer in range(3):
            ledger.enter(layer, update, ready=update > 0)
            if update:
                assert ledger.issuing == (layer, update + 1)
                ledger.produced()
                assert ledger.consume(layer, update + 1)
                with pytest.raises(RuntimeError, match="already consumed"):
                    ledger.consume(layer, update + 1)
            else:
                assert not ledger.consume(layer, update + 1)
    assert ledger.finish(5) == dict(
        key_calls=15,
        folded_keys=12,
        original_cold_keys=3,
        pending=False,
        all_value_projections_original=True,
    )


@pytest.mark.parametrize(
    "index,history,ready", [(1, 0, False), (0, 1, True), (0, 0, True), (False, 0, False), (0, 0, 0)]
)
def test_cold_warm_order_and_explicit_policy_fail_closed(index, history, ready):
    with pytest.raises(RuntimeError):
        joint.Ledger(2).enter(index, history, ready=ready)


def warm_ledger():
    ledger = joint.Ledger(1)
    ledger.enter(0, 0, ready=False)
    ledger.enter(0, 1, ready=True)
    return ledger


def test_unconsumed_wrong_consumer_and_incomplete_capture_are_not_silently_replayed():
    ledger = warm_ledger()
    with pytest.raises(RuntimeError, match="consumer"):
        ledger.consume(0, 2)
    ledger.produced()
    with pytest.raises(RuntimeError, match="consumer"):
        ledger.consume(1, 2)
    with pytest.raises(RuntimeError, match="previous"):
        ledger.enter(0, 2, ready=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        ledger.finish(2)
    assert ledger.consume(0, 2)
    assert ledger.finish(2)["folded_keys"] == 1
    with pytest.raises(RuntimeError, match="complete"):
        ledger.produced()


def test_exception_scope_isolation_and_no_state_survives_between_requests():
    seen = []

    @joint.managed
    def call(*, joint_incoming_projection="none", fail=False):
        seen.append(joint._scope.get())
        if fail:
            raise RuntimeError("stop")
        return None, None, {}

    with pytest.raises(RuntimeError, match="stop"):
        call(joint_incoming_projection="keys", fail=True)
    assert joint._scope.get() is None
    result = call(joint_incoming_projection="keys")[-1]["joint_incoming_projection"]
    assert result["policy"] == "keys" and result["value_projection_unchanged"]
    assert seen[0] is not seen[1] and joint._scope.get() is None


def test_nested_model_requests_are_not_allowed():
    @joint.managed
    def inner():
        return None, None, {}

    @joint.managed
    def outer():
        return inner()

    with pytest.raises(RuntimeError, match="nest"):
        outer()
    assert joint._scope.get() is None


def test_original_engine_factory_remains_patchable_and_the_default_is_unchanged(monkeypatch):
    from mosaickv.evaluation import lt_source_chunks

    expected = object()
    monkeypatch.setattr(lt_source_chunks, "Engine", lambda packet: expected)
    assert joint.engine_factory(None) is expected


def test_all_actual_extra_workspace_owners_are_charged_and_retired():
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    tensors = tuple(torch.zeros(n) for n in (8, 16, 4, 7))
    cache = SimpleNamespace(
        layers=[],
        shared_tensors=tensors,
        shared_bytes=backing_bytes(tensors),
        observe=lambda *args: None,
    )
    pool = PromptGraphPool(cache)
    engine = SimpleNamespace(
        source=tensors[0],
        output=tensors[1],
        extra_tensors=(tensors[2],),
        record=lambda: {},
        close=lambda: None,
    )
    pool.lt_engine = engine
    assert joint.owned(engine) == tensors[:3]
    assert pool.record()["active_workspace_bytes"] == backing_bytes(tensors[:3])
    pool.close()
    assert cache.shared_tensors == tensors[3:]
    assert cache.shared_bytes == backing_bytes(tensors[3:])
    assert pool.record()["active_workspace_bytes"] == 0


def test_same_buffer_key_producer_is_consumed_without_a_second_graph_replay(monkeypatch):
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    buffer = torch.zeros(1, 1, 5, 128, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        history_length=1, leased=None, representation=object(), attention=object()
    )
    cache = SimpleNamespace(layers=[layer])
    pool = PromptGraphPool(cache)
    pool.buffer = buffer
    ledger, replays = warm_ledger(), []
    owner = object.__new__(joint.Transport)
    owner.pool, owner.indices, owner.ledger, owner.closed = pool, {id(layer): 0}, ledger, False
    pool.joint_projection = owner
    signature = id(layer.representation), id(layer.attention), buffer.data_ptr(), buffer.stride()
    pool.entries[id(layer), 0] = signature, SimpleNamespace(replay=lambda: replays.append(1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pool.reconstruct(layer, 0, buffer, _projection_epoch=2)
    ledger.produced()
    layer.history_length, layer.leased = 2, (buffer, buffer)
    pool.reconstruct(layer, 0, buffer)
    assert replays == [1] and pool.replays == 1 and ledger.pending is None
    with pytest.raises(RuntimeError, match="already consumed"):
        pool.reconstruct(layer, 0, buffer)


def test_unauthorized_preprojection_is_rejected_before_any_graph_is_executed():
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    pool = PromptGraphPool(SimpleNamespace(layers=[]))
    with pytest.raises(RuntimeError, match="producer"):
        pool.reconstruct(SimpleNamespace(), 0, None, _projection_epoch=1)
