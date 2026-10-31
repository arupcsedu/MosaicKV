import inspect

import evicastkv_funded_delimiter_diagnostic as diagnostic


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
