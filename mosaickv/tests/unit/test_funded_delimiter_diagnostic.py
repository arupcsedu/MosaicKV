import inspect

import evicastkv_funded_delimiter_diagnostic as diagnostic
import pytest
import torch


def test_all_methods_receive_the_same_registered_options():
    assert len(diagnostic.VARIANTS) == 15
    for m in diagnostic.METHODS:
        for p in ("anchor", "source", "completion"):
            assert diagnostic.VARIANTS[m + "--" + p] == (m, p)
    worker = inspect.getsource(diagnostic.worker)
    assert "zero_validity_control.json" in worker
    assert "cycle_controls.json" in worker and "mask_controls.json" in worker
    assert "generation_delimiter(prepared, adapter.processor.tokenizer)" in worker
    assert "transaction(" in worker
    assert 'decode_wall_seconds=storage["decode_wall_seconds"]' in worker
    assert 'decode_cuda_ms=storage["decode_cuda_ms"]' in worker
    assert "prototype_or_repair=False" in worker
    assert 'elif representation == "source"' in worker


def test_no_fresh_or_paper_promotion():
    assert "independent_quality_N=0" in inspect.getsource(diagnostic.analyze)
    assert '"--array=0-1%2"' in inspect.getsource(diagnostic.submit)


@pytest.mark.parametrize(
    "count,parallel,expected",
    [(2, 2, "--array=0-1%2"), (10, 6, "--array=0-9%6"), (2, 6, "--array=0-1%2")],
)
def test_finite_common_method_array_preserves_legacy_scope(count, parallel, expected):
    assert diagnostic.bounded_worker_array(count, parallel) == expected


@pytest.mark.parametrize("count,parallel", [(0, 2), (13, 6), (10, 7), (True, 2), (2, True)])
def test_array_cannot_expand_beyond_registered_finite_resources(count, parallel):
    with pytest.raises(ValueError):
        diagnostic.bounded_worker_array(count, parallel)


def test_identity_gate_is_source_only_completion_changes_are_expected():
    anchor = torch.zeros(2, 3)
    changed = torch.ones(2, 3)
    diagnostic.validate_source_identity("completion", changed, changed, anchor, anchor, 0)
    diagnostic.validate_source_identity("source", anchor, anchor, anchor, anchor, 0)
    with pytest.raises(RuntimeError):
        diagnostic.validate_source_identity("source", changed, anchor, anchor, anchor, 0)
    with pytest.raises(RuntimeError):
        diagnostic.validate_source_identity("source", anchor, anchor, anchor, anchor, 1)
    with pytest.raises(ValueError):
        diagnostic.validate_source_identity(
            "arbitrary_fallback", changed, changed, anchor, anchor, 0
        )


def test_computed_tensors_survive_postcondition_failure_and_compatible_rows_reused():
    worker = inspect.getsource(diagnostic.worker)
    assert worker.index('"-computed.pt"') < worker.index("validate_source_identity(")
    assert "if name in reused:" in worker
    assert "checkpoint_variant(out, reused[name], fixed, greedy)" in worker
