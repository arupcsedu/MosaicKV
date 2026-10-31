from types import SimpleNamespace

import evicastkv_precision_rate_distortion as driver
import pytest

from mosaickv.evaluation import head_local_precision, request_precision


@pytest.mark.parametrize("fail", [False, True])
def test_request_delegation_restores_constructor_and_function_on_failure(monkeypatch, fail):
    old_constructor = request_precision.construct_request_formats
    calls = []

    def complete(*args, **kwargs):
        calls.append(kwargs["precision"])
        assert request_precision.construct_request_formats is not old_constructor
        if fail:
            raise RuntimeError("fixture")
        return "logits", {}

    def worker(*args):
        logits, record = request_precision.complete_request(precision="head3k2v_refit")
        assert logits == "logits"
        assert record["development_precision_policy"] == "head3k2v_refit"

    monkeypatch.setattr(request_precision, "complete_request", complete)
    monkeypatch.setattr(driver.previous, "worker", worker)
    if fail:
        with pytest.raises(RuntimeError, match="fixture"):
            driver.worker(None, None, 0)
    else:
        driver.worker(None, None, 0)
    assert request_precision.complete_request is complete
    assert request_precision.construct_request_formats is old_constructor
    assert calls == ["query3"]


def test_retention_one_remains_an_ordinary_untransformed_bypass(monkeypatch):
    seen = []

    def complete(*args, **kwargs):
        if kwargs["method"] == "mosaickv_exact" and kwargs["ratio"] == 1:
            return request_precision.complete_request(
                method="full_kv", precision="ordinary", ratio=1
            )
        seen.append(kwargs)
        return "ordinary_logits", {}

    def worker(*args):
        logits, _ = request_precision.complete_request(
            method="mosaickv_exact", precision="head3k2v_refit", ratio=1
        )
        assert logits == "ordinary_logits"

    monkeypatch.setattr(request_precision, "complete_request", complete)
    monkeypatch.setattr(driver.previous, "worker", worker)
    driver.worker(None, None, 0)
    assert seen == [dict(method="full_kv", precision="ordinary", ratio=1)]


def test_full_prompt_prior_survives_summary_capture_in_complete_request(monkeypatch):
    seen = []
    prior = {0: "full_prompt_prior_fixture"}

    def online(*args, **kwargs):
        seen.append(kwargs["priors"])
        return {}, {}

    monkeypatch.setattr(request_precision, "online_frozen", online)
    monkeypatch.setattr(head_local_precision, "baseline_payload", lambda *a, **kw: ((), None, {}))
    monkeypatch.setattr(head_local_precision.Path, "read_text", lambda *a, **kw: "[]")
    adapter = SimpleNamespace(_legacy_layers=lambda _: ((), "tuple"))
    head_local_precision.bundle(
        adapter,
        SimpleNamespace(modality_map=()),
        SimpleNamespace(state=SimpleNamespace(past_key_values=None)),
        SimpleNamespace(summaries=True, priors=prior),
        None,
        "mosaickv_exact",
        "fixture",
        0.5,
        policies=("union3",),
    )
    assert seen == [prior]


def test_canary_rejects_missing_rows_and_changed_union_hash():
    rows = [
        dict(
            sample_id=str(i),
            method="lookm_reimpl",
            precision=policy,
            logits_sha256="same",
            status="completed",
            warmup=False,
        )
        for i in range(4)
        for policy in driver.POLICIES
    ]
    old = [
        dict(
            sample_id=str(i),
            method="lookm_reimpl",
            precision="query3",
            logits_sha256="same",
            status="completed",
            warmup=False,
        )
        for i in range(4)
    ]
    result = dict(status="completed", failures=[])
    assert driver.canary_valid(result, rows, old)
    assert not driver.canary_valid(result, rows[:-1], old)
    rows[0]["logits_sha256"] = "changed"
    assert not driver.canary_valid(result, rows, old)
