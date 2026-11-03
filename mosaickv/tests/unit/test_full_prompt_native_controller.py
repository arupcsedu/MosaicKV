import json

import evicastkv_full_prompt_native as driver
import pytest


def summary(improve=0.98, versus=1.0):
    return dict(
        within_method={"mosaickv_exact": dict(registered_two_policy_ratio_bounds=[0.8, improve])},
        versus_prefix={"native": dict(registered_two_policy_ratio_bounds=[0.8, versus])},
    )


def test_promotion_requires_both_inputs_and_both_latency_controls():
    assert driver.paired_passes([summary(), summary()])
    assert not driver.paired_passes([summary(), summary(improve=0.991)])
    assert not driver.paired_passes([summary(), summary(versus=1.051)])
    assert not driver.paired_passes([summary()])


def test_memory_gate_uses_strongest_byte_competitor_on_every_boundary():
    fields = (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    )
    refs = {
        m: dict(storage=dict(zip(fields, sizes, strict=True)))
        for m, sizes in {
            "mosaickv_exact": (9, 19, 29),
            "prefixkv_reimpl": (20, 30, 40),
            "prompt_attention_topk": (10, 25, 35),
        }.items()
    }
    assert driver.memory_gate(dict(zip(driver.STORAGE, (9, 24, 34), strict=True)), refs)
    assert not driver.memory_gate(dict(zip(driver.STORAGE, (10, 24, 34), strict=True)), refs)
    assert not driver.memory_gate(dict(zip(driver.STORAGE, (9, 24, 36), strict=True)), refs)


def test_timed_peak_is_never_relabeled_audited_peak():
    audit = dict(initial_encoded_bytes=10, final_encoded_bytes=20, decode_owned_peak_bytes=40)
    fast = dict(
        initial_encoded_bytes=10,
        final_encoded_bytes=20,
        reported_counter_peak_bytes=30,
        decode_owned_peak_bytes=None,
        peak_measurement_admissible=False,
        tensor_result_auditing=False,
    )
    driver.verify_fast(fast, audit, 30)
    for changed in (
        dict(decode_owned_peak_bytes=30),
        dict(reported_counter_peak_bytes=29),
        dict(final_encoded_bytes=21),
    ):
        with pytest.raises(RuntimeError):
            driver.verify_fast(dict(fast, **changed), audit, 30)


def test_second_policy_is_finite_and_never_runs_after_unsafe_failure(monkeypatch, tmp_path):
    plan = dict(
        source_sha="s",
        chain_root=str(tmp_path),
        selected_policy="kv",
        policy_attempt=0,
        eligible_policies=["kv", "v_only"],
    )
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(driver, "checked", lambda *a: plan)
    monkeypatch.setattr(driver, "load_producers", lambda *a: ([summary(versus=1.1)] * 2, []))
    calls = []
    monkeypatch.setattr(driver, "child", lambda *a, **kw: calls.append(kw) or tmp_path / "next")
    monkeypatch.setattr(driver, "submit_array", lambda *a: None)
    driver.paired_analysis(tmp_path, "h")
    assert (
        len(calls) == 1
        and calls[0]["policy_attempt"] == 1
        and calls[0]["selected_policy"] == "v_only"
    )
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "plan.json").write_text(json.dumps(dict(plan, chain_root=str(unsafe))))
    monkeypatch.setattr(driver, "load_producers", lambda *a: ([], ["bad logits"]))
    driver.paired_analysis(unsafe, "h")
    assert len(calls) == 1
    assert not driver.read(unsafe / "engineering_closeout.json")["paper_superiority_established"]


def test_missing_primitive_has_terminal_failure_instead_of_orphaned_chain(monkeypatch, tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(dict(source_sha="s", chain_root=str(tmp_path))))
    monkeypatch.setattr(driver, "checked", lambda *a: None)
    monkeypatch.setattr(driver, "submit_array", lambda *a: pytest.fail("must not submit GPU"))
    driver.begin(tmp_path, "h")
    record = driver.read(tmp_path / "engineering_closeout.json")
    assert not record["complete"] and record["failures"]


def test_registered_bounds_are_reproducible_and_reject_missing_trials():
    rows = [
        dict(
            method=m,
            reader=mode,
            trial=t,
            warmup=False,
            total_wall_seconds=(2 if m == "mosaickv_exact" else 1)
            * (1 if mode == "original" else 0.9),
        )
        for m in ("mosaickv_exact", "prefixkv_reimpl")
        for mode in driver.READERS
        for t in range(20)
    ]
    bounds = driver.registered_bounds(rows, 11)
    assert bounds == driver.registered_bounds(rows, 11)
    assert bounds["within_method"]["mosaickv_exact"] == pytest.approx([0.9, 0.9])
    assert bounds["versus_prefix"]["native"] == pytest.approx([2, 2])
    with pytest.raises(ValueError, match="twenty-trial"):
        driver.registered_bounds(rows[:-1], 11)
