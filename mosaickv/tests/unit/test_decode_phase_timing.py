import json

import pytest

from mosaickv.evaluation.construction_phase_timing import PhaseTimer
from mosaickv.evaluation.decode_phase_timing import instrument, next_dependency


def test_decode_instrumentation_preserves_return_and_restores_on_exception(monkeypatch):
    from mosaickv.evaluation import lossless_code_delta_gpu

    value = object()
    monkeypatch.setattr(lossless_code_delta_gpu, "decode", lambda *a, **kw: value)
    original = lossless_code_delta_gpu.decode
    timer = PhaseTimer()
    with pytest.raises(RuntimeError, match="stop"), instrument(timer):
        assert lossless_code_delta_gpu.decode(None, None) is value
        raise RuntimeError("stop")
    assert lossless_code_delta_gpu.decode is original
    assert not timer.stack
    assert any(r["phase"].endswith("lossless_code_delta_gpu.decode") for r in timer.resolve())


def test_dependency_prefers_latest_published_stage_and_rejects_missing(tmp_path):
    with pytest.raises(ValueError, match="dependency"):
        next_dependency(tmp_path / "missing")
    root = tmp_path / "run"
    root.mkdir()
    (root / "submission.json").write_text(json.dumps(dict(cpu_analysis="123")))
    assert next_dependency(root) == "123"
    (root / "timing").mkdir()
    (root / "timing/submission.json").write_text(json.dumps(dict(cpu_analysis="124")))
    assert next_dependency(root) == "124"


@pytest.mark.parametrize("passed", [True, False])
def test_followup_stops_on_timing_success_or_incomplete_parent(monkeypatch, tmp_path, passed):
    import evicastkv_decode_phase_followup as driver

    stage = tmp_path / "stage"
    (stage / "timing/analysis").mkdir(parents=True)
    (stage / "engineering_closeout.json").write_text("{}")
    (stage / "timing/plan.json").write_text("{}")
    decision = dict(
        complete=passed,
        failures=[],
        all_method_similar_time_gate=passed,
        plan_sha256=driver.sha(stage / "timing/plan.json"),
    )
    (stage / "timing/analysis/decision.json").write_text(json.dumps(decision))
    monkeypatch.setattr(driver, "STAGED", stage)
    monkeypatch.setattr(driver, "clean_source", lambda _: "revision")
    monkeypatch.setattr(driver.common.previous, "cpu", lambda *a: pytest.fail("no CPU/GPU launch"))
    root = tmp_path / "out"
    if passed:
        driver.watch(root, "revision", 0)
        result = json.loads((root / "not_needed.json").read_text())
        assert result["paper_superiority_established"] is False
    else:
        with pytest.raises(ValueError, match="safe complete parent"):
            driver.watch(root, "revision", 0)
        assert not root.exists()


def test_followup_rejects_unbounded_callbacks(monkeypatch, tmp_path):
    import evicastkv_decode_phase_followup as driver

    monkeypatch.setattr(driver, "clean_source", lambda _: "revision")
    with pytest.raises(ValueError, match="finite"):
        driver.watch(tmp_path / "out", "revision", 9)
