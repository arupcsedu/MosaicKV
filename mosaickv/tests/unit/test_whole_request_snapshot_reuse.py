"""Constructor opt-in plumbing only; no CUDA or peak-certification fixtures."""

from unittest.mock import Mock

import pytest

from mosaickv.evaluation.whole_request_kv_ledger import WholeRequestKVLedger


def build(factory, **kwargs):
    return WholeRequestKVLedger(
        device="synthetic",
        layer_count=2,
        source=True,
        ledger_factory=factory,
        **kwargs,
    )


@pytest.mark.parametrize("kwargs", [{}, {"constructor_snapshot_reuse": False}])
def test_default_and_explicit_false_preserve_legacy_device_only_factory(kwargs):
    calls = []
    marker = object()

    def legacy(*, device):
        calls.append(device)
        return marker

    wrapper = build(legacy, **kwargs)
    assert calls == ["synthetic"]
    assert wrapper.ledger is marker
    assert wrapper.phases == []
    assert wrapper.closed is False


def test_true_forwards_only_the_explicit_opt_in():
    factory = Mock()
    wrapper = build(factory, constructor_snapshot_reuse=True)
    factory.assert_called_once_with(device="synthetic", constructor_snapshot_reuse=True)
    assert wrapper.ledger is factory.return_value


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_nonboolean_reuse_is_rejected_before_any_factory_side_effect(value):
    factory = Mock()
    with pytest.raises(ValueError, match="explicit boolean"):
        build(factory, constructor_snapshot_reuse=value)
    factory.assert_not_called()


def test_incompatible_opt_in_factory_fails_without_silent_default_retry():
    calls = []

    def legacy(*, device):
        calls.append(device)

    with pytest.raises(TypeError, match="constructor_snapshot_reuse"):
        build(legacy, constructor_snapshot_reuse=True)
    assert not calls
