import pytest
import torch

from mosaickv.evaluation.deferred_transport_checks import predicate_span, use
from mosaickv.evaluation.predicate_span_operator import summarize
from mosaickv.evaluation.shared_source_format import assert_transport_close, transport_corrections


@pytest.mark.parametrize("chunk", [1, 17, 128, 65536])
def test_whole_span_preserves_patch_words_and_every_verification_element(chunk):
    reference = torch.arange(256).reshape(1, 2, 1, 128).bfloat16() / 32
    actual = reference.clone()
    actual.flatten()[::13] += 0.5
    protected = torch.empty(0, dtype=torch.int32)
    with use(True, coalesce_nonzero=True):
        old = transport_corrections((reference,), (actual,), protected, chunk_scalars=chunk)[0]
    with use(True, coalesce_nonzero=True, full_predicate_span=True) as state:
        new = transport_corrections((reference,), (actual,), protected, chunk_scalars=chunk)[0]
        checked = actual.clone()
        new.apply(checked)
        assert_transport_close(checked, reference, chunk_scalars=chunk)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(old.tensors, new.tensors, strict=True)
    )
    assert state.span_calls == dict(corrections=1, verification=1)
    assert state.span_peak_elements == reference.numel()


@pytest.mark.parametrize(
    "kind,error", [("finite", ValueError), ("protected", RuntimeError), ("close", AssertionError)]
)
def test_larger_span_keeps_negative_guards(kind, error):
    x = torch.ones(1, 1, 17, 128, dtype=torch.bfloat16)
    y = x.clone()
    y.flatten()[-1] = float("nan") if kind == "finite" else 2.0
    with pytest.raises(error), use(True, coalesce_nonzero=True, full_predicate_span=True):
        if kind == "close":
            assert_transport_close(y, x, chunk_scalars=128)
        else:
            transport_corrections(
                (x,), (y,), torch.tensor([16], dtype=torch.int32), chunk_scalars=128
            )


def test_scalar_bound_and_default_do_not_silently_expand():
    assert predicate_span(500, 17) == 17
    with use(True, full_predicate_span=True) as state:
        assert predicate_span(500, 17, kind="verification") == 500
        assert predicate_span(4194305, 17, kind="verification") == 17
        assert predicate_span(4194305, None, kind="corrections") == 524288
    assert state.span_peak_elements == 500 and state.span_fallbacks == 2
    with pytest.raises(ValueError), use(False, full_predicate_span=True):
        pass


def fixture(ratio):
    return [
        dict(
            trial=t, full_span=full, wall_seconds=ratio if full else 1.0, patch_words_identical=True
        )
        for t in range(-5, 20)
        for full in (False, True)
    ]


def test_registered_operator_gate_is_not_a_request_gate():
    assert summarize(fixture(0.7))["operator_improvement_gate"]
    assert not summarize(fixture(0.9))["operator_improvement_gate"]
    assert not summarize(fixture(0.7))["request_latency_evidence"]
    bad = fixture(0.7)
    bad[-1]["patch_words_identical"] = False
    with pytest.raises(ValueError):
        summarize(bad)
    with pytest.raises(ValueError):
        summarize(fixture(0.7)[:-1])


@pytest.mark.parametrize("mutation", ["nan", "zero", "bool", "duplicate"])
def test_operator_gate_rejects_invalid_timing_rows(mutation):
    rows = fixture(0.7)
    if mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1]["wall_seconds"] = {"nan": float("nan"), "zero": 0, "bool": True}[mutation]
    with pytest.raises(ValueError):
        summarize(rows)


def test_invalid_span_flags_fail_before_model_or_cuda_work():
    from mosaickv.evaluation.common_streaming_timing import request

    for flags in (dict(full_predicate_span=True), dict(full_predicate_span="yes")):
        with pytest.raises(ValueError, match="validation"):
            request(None, None, None, "mosaickv_exact", **flags)
