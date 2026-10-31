import inspect

import evicastkv_text_source_diagnostic as diagnostic


def test_each_baseline_receives_identical_storage_option():
    assert len(diagnostic.VARIANTS) == 10
    for method in diagnostic.METHODS:
        assert diagnostic.VARIANTS[method + "--anchor"] == (method, "anchor")
        assert diagnostic.VARIANTS[method + "--source"] == (method, "source")
    text = inspect.getsource(diagnostic.worker)
    assert "for name, (method, representation) in VARIANTS.items():" in text
    assert "TextSourceCapture(adapter, prepared) as text_capture" in text
    assert 'policy = "head3_boundary8"' in text
    assert "torch.equal(greedy, greedy_anchors[method])" in text
    assert "changed_scalars" in text
    assert "prototype_or_repair=False" in text
    assert "mass_policy=" not in text and "value_absorption=" not in text


def test_finite_analysis_and_no_fresh_promotion():
    text = inspect.getsource(diagnostic.submit)
    assert '"--array=0-1%2"' in text and "--dependency=afterany:" in text
    assert '"--time=00:30:00"' in text
    assert "independent_quality_N=0" in inspect.getsource(diagnostic.analyze)
