import evicastkv_mixed_boundary_study as driver
import pytest


def test_task_constructor_keeps_union_control_and_explicit_forecast_settings(monkeypatch):
    from mosaickv.evaluation import head_local_precision

    calls = []

    def bundle(*args, **kwargs):
        calls.append(kwargs)
        return {kwargs["policies"][0]: "packet"}, {}

    monkeypatch.setattr(head_local_precision, "bundle", bundle)
    args = [*range(8), "query3"]
    args[5] = "mosaickv_exact"
    for policy in ("union3", *driver.SETTINGS):
        _, proof = driver.construct(*args, policy=policy)
        assert proof["precision"] == policy
    assert calls == [
        dict(policies=("union3",), allocation_mode="profile", future_steps=0),
        dict(policies=("head3k2v_refit",), allocation_mode="balanced", future_steps=0),
        dict(policies=("head3k2v_refit_boundary8",), allocation_mode="balanced", future_steps=0),
    ]
    for method in ("lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl"):
        args[5] = method
        driver.construct(*args, policy="mixed_refit2_boundary8")
        assert calls[-1] == dict(
            policies=("head3k2v_refit_boundary8",), allocation_mode="profile", future_steps=0
        )
    with pytest.raises(ValueError, match="registered"):
        driver.construct(*range(8), "query3", policy="unregistered")
