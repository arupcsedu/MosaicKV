from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.device_guard_records import DeviceRecords, retire_before_pool
from mosaickv.evaluation.guard_tail_graph import GuardTail
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def record():
    return torch.tensor([0.001, 0.125, 1.0, 1.0, 1.0], dtype=torch.float64)


def test_records_have_bounded_real_backing_and_do_not_alias_replayed_outputs():
    state = DeviceRecords(127, "cpu")
    source = record()
    state.append(source)
    source.zero_()
    assert torch.equal(state.snapshot()[0], record())
    assert state.bytes == 127 * 5 * 8
    saved = state.snapshot()
    for _ in range(126):
        state.append(record())
    assert state.snapshot().shape == (127, 5)
    assert saved.shape == (1, 5) and torch.equal(saved[0], record())
    with pytest.raises(ValueError, match="bounded"):
        state.append(record())
    state.release()
    with pytest.raises(RuntimeError, match="retired"):
        state.snapshot()


@pytest.mark.parametrize("location", (0, 2, 4))
def test_every_invalid_step_fails_before_results_can_be_published(location):
    state = DeviceRecords(5, "cpu")
    for i in range(5):
        item = record()
        if i == location:
            item[0], item[4] = float("nan"), 0.0
        state.append(item)
    with pytest.raises(ValueError, match="nonfinite logits invalidate request"):
        state.snapshot()
    assert state.invalid_steps == [location] and state.failed
    with pytest.raises(ValueError, match="bounded"):
        state.append(record())
    state.release()


def test_forgotten_validation_still_rejects_request_and_releases_graph_owners():
    state = DeviceRecords(1, "cpu")
    item = record()
    item[4] = 0
    state.append(item)
    events = []
    tail = GuardTail(None, None, device_records=True)
    tail.records = state
    tail.graph = SimpleNamespace(reset=lambda: events.append("graph retired"))
    with pytest.raises(ValueError, match="invalidate request"):
        tail.close()
    assert tail.closed and state.data is None and events == ["graph retired"]
    assert tail.report()["record_validation_failed"]


def test_pool_ownership_check_runs_even_when_validation_fails():
    events = []
    tail = SimpleNamespace(closed=False)

    def close():
        tail.closed = True
        events.append("guard graph retired")
        raise ValueError("invalid request")

    tail.close = close
    with pytest.raises(ValueError, match="invalid request"):
        retire_before_pool("pool", tail, lambda p: events.append("strict pool check"))
    assert events == ["guard graph retired", "strict pool check"]


def test_device_policy_cannot_silently_disable_export_or_repair():
    with pytest.raises(ValueError, match="device records require"):
        OverlayPolicy(device_guard_records=True)
    p = OverlayPolicy(
        inline_repair=True,
        live_anchor_guard=True,
        anchor_tail="manual_shared_stream",
        reuse_anchor_readout=True,
        export_parent_logits=False,
        device_guard_records=True,
    )
    assert p.device_guard_records
    for maximum in (0, 128, False):
        with pytest.raises(ValueError, match="bounded guard horizon"):
            DeviceRecords(maximum, "cpu")


def test_transferred_records_preserve_every_native_acceptance_and_veto_bitwise():
    from mosaickv.evaluation.live_anchor_guard import decision

    state = DeviceRecords(4, "cpu")
    parent = torch.tensor([4.0, 1.0, 0.0])
    expected = []
    for candidate in (parent, parent + 0.01, parent + 1.01, torch.tensor([0.0, 4.0, 0.0])):
        _, record = decision(parent, candidate)
        value = torch.cat((record, torch.ones(1, dtype=record.dtype)))
        state.append(value)
        expected.append(value)
    assert torch.equal(state.snapshot(), torch.stack(expected))


def test_pool_patch_and_readout_hooks_restore_on_a_failed_request():
    from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard, _current
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    norm, head = torch.nn.Identity(), torch.nn.Identity()
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(norm=norm)),
        get_output_embeddings=lambda: head,
    )
    policy = OverlayPolicy(
        inline_repair=True,
        live_anchor_guard=True,
        anchor_tail="manual_shared_stream",
        reuse_anchor_readout=True,
        export_parent_logits=False,
        device_guard_records=True,
    )
    original = PromptGraphPool.close
    g = LiveAnchorGuard(SimpleNamespace(model=model), SimpleNamespace(head=0), {}, policy=policy)
    with pytest.raises(RuntimeError, match="controlled request"), g:
        assert PromptGraphPool.close is not original
        raise RuntimeError("controlled request failure")
    assert PromptGraphPool.close is original and _current.get() is None
    assert g.tail.closed


def test_graph_reset_failure_still_releases_buffer_and_weight_owners():
    tail = GuardTail(object(), object(), device_records=True)
    tail.records = DeviceRecords(1, "cpu")
    tail.records.append(record())

    def fail():
        raise RuntimeError("controlled graph reset failure")

    tail.graph = SimpleNamespace(reset=fail)
    with pytest.raises(RuntimeError, match="controlled graph reset"):
        tail.close()
    assert tail.closed and tail.norm is tail.head is tail.graph is None
    assert tail.records.data is None
