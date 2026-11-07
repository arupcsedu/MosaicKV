from types import SimpleNamespace

import pytest

from mosaickv.evaluation.bounded_probe_horizon import expected_updates, two_prediction_probe


def test_default_complete_contract_and_exception_isolation():
    assert expected_updates() == 127
    with pytest.raises(ValueError), two_prediction_probe():
        assert expected_updates() == 1
        with pytest.raises(RuntimeError, match="nest"), two_prediction_probe():
            pass
        raise ValueError("deliberate")
    assert expected_updates() == 127


def test_rotary_still_requires_exactly_all_layers_at_the_registered_horizon():
    from mosaickv.evaluation import exact_rotary_dispatch as rotary

    @rotary.managed
    def calls(*, count, **kwargs):
        rotary._scope.get()["calls"] = count
        return [], [], {}

    assert calls(count=4064, exact_rotary_dispatch=True)
    with pytest.raises(RuntimeError), two_prediction_probe():
        calls(count=31, exact_rotary_dispatch=True)
    with two_prediction_probe():
        assert calls(count=32, exact_rotary_dispatch=True)
    with pytest.raises(RuntimeError):
        calls(count=32, exact_rotary_dispatch=True)


@pytest.mark.parametrize("updates", [1, 127])
def test_norm_counts_keep_every_cold_capture_replay_and_copy(updates):
    from contextlib import nullcontext

    from mosaickv.evaluation.input_norm_segment import Transport

    state = SimpleNamespace(
        close=lambda: None,
        calls=32 * updates,
        cold=32,
        captures=0 if updates == 1 else 32,
        replays=32 * (updates - 1),
        skipped_copies=31 * (updates - 1),
        first_layer_copies=updates - 1,
        state={"reuse": True},
        bytes=0,
        extra_graph_bytes=0,
        closed=True,
    )
    with two_prediction_probe() if updates == 1 else nullcontext():
        Transport.__exit__(state, None, None, None)
        state.replays += 1
        with pytest.raises(RuntimeError):
            Transport.__exit__(state, None, None, None)


def test_one_update_epilogue_cannot_silently_skip_a_layer():
    from mosaickv.evaluation.post_attention_segment import Transport

    state = SimpleNamespace(
        close=lambda: None,
        calls=32,
        captures=32,
        replays=32,
        state={},
        bytes=0,
        extra_graph_bytes=0,
        closed=True,
        backend_warm_allocation_bytes=0,
    )
    with two_prediction_probe():
        Transport.__exit__(state, None, None, None)
        state.calls -= 1
        with pytest.raises(RuntimeError):
            Transport.__exit__(state, None, None, None)
