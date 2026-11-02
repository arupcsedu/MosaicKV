import pytest
from evicastkv_construction_phase import constructor_memory_gate
from evicastkv_constructor_followup import followup_needed


def rows():
    return [
        dict(
            input_index=i,
            method=m,
            warmup=True,
            packet_bitwise=True,
            prefill_logits_bitwise=True,
            constructor_memory=dict(
                tensor_results_observed=True,
                original_KV_tensor_objects_alive=0,
                original_native_layers_remaining=0,
                observed_prefill_cuda_peak_bytes=96 if m == "mosaickv_exact" else 100,
                final_prefill_cache_bytes=80,
            ),
        )
        for i in (0, 1)
        for m in (
            "mosaickv_exact",
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
        )
    ]


def test_advance_only_after_complete_negative_latency():
    assert followup_needed(dict(complete=True, failures=[], all_method_similar_time_gate=False))
    assert not followup_needed(dict(complete=True, failures=[], all_method_similar_time_gate=True))
    for value in (
        {},
        dict(complete=False, failures=[]),
        dict(complete=True, failures=["error"]),
        dict(complete=True, failures=[], all_method_similar_time_gate="false"),
    ):
        with pytest.raises(ValueError):
            followup_needed(value)


def test_complete_unique_audited_lower_constructor_peak_required():
    data = rows()
    assert constructor_memory_gate(data)
    assert not constructor_memory_gate(data[:-1])
    assert not constructor_memory_gate([*data, data[0]])
    data[0]["constructor_memory"]["observed_prefill_cuda_peak_bytes"] = 101
    assert not constructor_memory_gate(data)


@pytest.mark.parametrize(
    "key,value", [("packet_bitwise", False), ("prefill_logits_bitwise", False), ("warmup", False)]
)
def test_identity_and_excluded_warmup_required(key, value):
    data = rows()
    data[0][key] = value
    assert not constructor_memory_gate(data)


def test_conditional_start_keeps_one_gpu_worker_resource_wrapper(monkeypatch, tmp_path):
    import evicastkv_constructor_followup as driver

    calls = []
    monkeypatch.setattr(driver.phase, "PARENT", driver.phase.PARENT)
    monkeypatch.setattr(driver, "clean_source", lambda _: "new-sha")
    monkeypatch.setattr(driver, "sha", lambda _: driver.PARENT_PLAN_SHA)
    monkeypatch.setattr(
        driver.common.previous,
        "read",
        lambda _: dict(
            complete=True,
            failures=[],
            all_method_similar_time_gate=False,
            plan_sha256=driver.PARENT_PLAN_SHA,
        ),
    )
    monkeypatch.setattr(
        driver.phase, "prepare", lambda root, **kwargs: calls.append(("prepare", kwargs))
    )
    monkeypatch.setattr(driver.common.previous, "cpu", lambda *args: calls.append(("cpu", args)))
    monkeypatch.setattr(
        driver.common.control.parent.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(("command", command)),
    )
    monkeypatch.setattr(
        driver.common.control.parent,
        "submit",
        lambda *args: driver.common.control.parent.subprocess.check_output(
            ["sbatch", "/x/evicastkv_funded_delimiter_diagnostic.sbatch"]
        ),
    )
    driver.start(tmp_path / "new-root", "new-sha")
    assert calls[0] == ("prepare", {"verify_constructor_memory_once": True})
    assert calls[1][0] == "cpu"
    assert calls[2][1][-1].endswith("/evicastkv_constructor_followup.sbatch")


def test_failed_memory_analysis_cannot_submit_timing(monkeypatch, tmp_path):
    import evicastkv_constructor_followup as driver

    writes = []
    monkeypatch.setattr(driver.phase, "analyze", lambda *args: None)
    monkeypatch.setattr(
        driver.common.previous.common, "inputs", lambda *args: {"source_sha": "new-sha"}
    )
    monkeypatch.setattr(
        driver.common.previous,
        "read",
        lambda _: dict(complete=True, failures=[], constructor_memory_gate=False),
    )
    monkeypatch.setattr(driver, "sha", lambda _: "digest")
    monkeypatch.setattr(driver, "write_once", lambda path, value: writes.append((path, value)))
    monkeypatch.setattr(
        driver.common, "submit", lambda *args: pytest.fail("failed gate must not submit timing")
    )
    driver.analyze(tmp_path, "plan")
    assert writes[0][0].name == "timing_blocked.json"
    assert writes[0][1]["GPU_timing_submitted"] is False
