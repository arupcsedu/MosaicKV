import pytest
import torch

from mosaickv.evaluation import joint_incoming_projection as joint
from mosaickv.evaluation.joint_input_copy import Stamp, identity


def test_every_route_consumes_its_same_input_exactly_once():
    stamp = Stamp(32)
    for step in range(127):
        for route in range(32):
            token = (step, route, "CPU metadata only")
            stamp.write(route, step, token)
            stamp.take(route, step, token)
    assert stamp.finish(127) == dict(prepared=4064, consumed=4064, pending=False)


@pytest.mark.parametrize("bad", ["route", "history", "identity", "duplicate", "unconsumed"])
def test_missing_stale_and_wrong_consumer_inputs_fail_closed(bad):
    stamp = Stamp(2)
    stamp.write(0, 0, (123,))
    with pytest.raises(RuntimeError):
        if bad == "route":
            stamp.take(1, 0, (123,))
        elif bad == "history":
            stamp.take(0, 1, (123,))
        elif bad == "identity":
            stamp.take(0, 0, (124,))
        elif bad == "duplicate":
            stamp.take(0, 0, (123,))
            stamp.take(0, 0, (123,))
        else:
            stamp.write(1, 0, (124,))


def test_wrong_prepare_order_and_incomplete_retirement_are_rejected():
    stamp = Stamp(32)
    with pytest.raises(RuntimeError, match="ordered"):
        stamp.write(1, 0, ())
    with pytest.raises(RuntimeError, match="complete"):
        stamp.finish(127)


def test_identical_storage_alias_is_not_the_identical_hidden_object():
    hidden = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    assert identity(hidden) == identity(hidden)
    assert identity(hidden) != identity(hidden.view_as(hidden))


def test_default_request_copy_scope_is_private_and_boolean_only():
    @joint.managed
    def call(**kw):
        return None, None, {}

    assert not call()[2]["joint_incoming_projection"]["source_input_copy"]["enabled"]
    assert call(reuse_source_input_copy=True)[2]["joint_incoming_projection"][
        "source_input_copy"
    ] == dict(enabled=True, reports=[])
    with pytest.raises(ValueError, match="boolean"):
        call(reuse_source_input_copy=1)
    assert joint._scope.get() is None
