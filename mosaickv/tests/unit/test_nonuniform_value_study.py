import inspect

import evicastkv_nonuniform_value_study as study


def test_one_common_format_and_preserved_stronger_baseline_rows():
    assert study.SETTINGS == {"nonuniform2_boundary8": ("head3k2v_nonuniform_boundary8", 0)}
    source = inspect.getsource(study.prepare)
    for condition in (
        "len(rows) != 768",
        "new_expected_rows=80",
        "total_expected_rows=848",
        "bootstrap_comparisons=52",
        "independent_quality_N=0",
        '"lookm_reimpl", "vl_cache_reimpl", "prefixkv_reimpl"',
    ):
        assert condition in source
    assert '"--array=1-19%6"' in inspect.getsource(study.submit)


def test_nonuniform_candidate_does_not_enable_prototype_repair_or_forecast():
    source = inspect.getsource(study.construct)
    assert "future_steps=steps" in source
    assert "balanced" in source
    assert "mass_policy" not in source and "value_absorption" not in source
