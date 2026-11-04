import pytest
import torch

from mosaickv.evaluation.deferred_transport_checks import use
from mosaickv.evaluation.shared_source_format import construct, validate_source_tensors


def fixture():
    shape = (1, 2, 6, 128)
    source = torch.arange(6 * 256).reshape(1, 6, 256).bfloat16()
    pair = tuple(source.reshape(shape).clone() for _ in range(2))
    phases = (torch.ones(1, 6, 128).bfloat16(), torch.zeros(1, 6, 128).bfloat16())
    identity = (
        torch.arange(6, dtype=torch.int32),
        torch.arange(6, dtype=torch.int16)[None].expand(2, 6),
    )
    return source, pair, phases, identity


def test_complete_predicates_queue_only_scalar_and_same_packet_words():
    source, pair, phases, identity = fixture()
    args = (source, pair, ((0, 2, 5), (1, 3, 5)), (5,), phases, identity)
    before = construct(*args)
    with use(True, batch_source_validation=True) as state:
        after = construct(*args)
        assert state.counts == {"source": 1}
        assert all(t.dtype == torch.bool and t.numel() == 1 for t in state.flags)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(before.private(), after.private(), strict=True)
    )
    assert state.report()["all_checks_before_decode"] and not state.flags


@pytest.mark.parametrize(
    "kind", ["union", "slots", "source", "key", "value", "cos", "sin", "cos_half", "sin_half"]
)
@pytest.mark.parametrize("batched", [False, True])
def test_every_original_tensor_predicate_rejects(kind, batched):
    source, pair, phases, identity = fixture()
    all_tensors = dict(
        source=source,
        key=pair[0],
        value=pair[1],
        cos=phases[0],
        sin=phases[1],
        union=identity[0],
        slots=identity[1].clone(),
    )
    identity = (all_tensors["union"], all_tensors["slots"])
    if kind in ("union", "slots"):
        all_tensors[kind].flatten()[0] = 2
    elif kind.endswith("_half"):
        all_tensors[kind[:-5]].flatten()[0] = 0.25
    else:
        all_tensors[kind].flatten()[0] = float("nan")
    with (
        pytest.raises(ValueError, match="shared original phase"),
        use(True, batch_source_validation=batched),
    ):
        validate_source_tensors(source, pair, phases, identity, h=2, n=6)


def test_no_source_flag_without_scope_or_when_disabled():
    source, pair, phases, identity = fixture()
    with use(True) as state:
        validate_source_tensors(source, pair, phases, identity, h=2, n=6)
    assert not state.counts
    with pytest.raises(ValueError), use(False, batch_source_validation=True):
        pass


def test_metadata_fails_immediately_even_when_tensor_predicates_deferred():
    source, pair, phases, identity = fixture()
    with use(True, batch_source_validation=True) as state, pytest.raises(ValueError):
        construct(source, pair, ((0, 5), (1, 5)), (5,), phases, (identity[0].long(), identity[1]))
    assert not state.counts


def test_complete_32_layer_transaction_stays_inside_original_flag_bound():
    from mosaickv.evaluation.deferred_transport_checks import enqueue

    source, pair, phases, identity = fixture()
    with use(True, batch_source_validation=True) as state:
        for _ in range(32):
            validate_source_tensors(source, pair, phases, identity, h=2, n=6)
            for _ in range(2):
                enqueue(torch.tensor(True), "finite")
                enqueue(torch.tensor(False), "protected", expected=False)
                enqueue(torch.tensor(True), "close")
    assert state.counts == dict(source=32, finite=64, protected=64, close=64)
    assert state.flushes == 1 and state.peak_bytes == 897
    assert not state.flags


@pytest.mark.parametrize(
    "policy",
    [
        dict(batch_source_validation="yes"),
        dict(coalesce_transport_nonzero=1),
        dict(batch_source_validation=True),
        dict(coalesce_transport_nonzero=True),
    ],
)
def test_invalid_policies_fail_before_model_or_cuda_work(policy):
    from mosaickv.evaluation.common_streaming_timing import request

    with pytest.raises(ValueError, match="validation"):
        request(None, None, None, "mosaickv_exact", **policy)
