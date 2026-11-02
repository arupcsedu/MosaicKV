from types import SimpleNamespace

import pytest

from mosaickv.evaluation import reconstruction_policy
from mosaickv.evaluation.request_reader_diagnostic import (
    METHODS,
    WindowTimer,
    instrument,
    order,
    phase_window,
    summarize,
    trace_summary,
    window,
)
from mosaickv.evaluation.scheduler_identity import scheduler_identity


def test_current_scheduler_identity_replaces_parent_and_preserves_zero_task():
    parent = dict(slurm_job_id="old", slurm_array_task_id="99")
    result = scheduler_identity(
        parent, dict(SLURM_JOB_ID="123", SLURM_ARRAY_JOB_ID="122", SLURM_ARRAY_TASK_ID="0")
    )
    assert result["slurm_job_id"] == "123"
    assert result["slurm_array_task_id"] == "0"
    assert result["slurm_array_element"] == "122_0"
    assert result["parent_scheduler_identity"] == parent
    assert parent["slurm_job_id"] == "old"
    assert scheduler_identity(parent, {})["slurm_job_id"] == "not_slurm"
    for env in (
        dict(SLURM_JOB_ID="bad"),
        dict(SLURM_ARRAY_TASK_ID="0"),
        dict(SLURM_ARRAY_JOB_ID="2", SLURM_ARRAY_TASK_ID="0"),
    ):
        with pytest.raises(ValueError):
            scheduler_identity(parent, env)


def test_fixed_windows_and_reproducible_complete_pair_grid():
    assert [window(n) for n in (0, 15, 16, 95, 96, 126)] == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]
    for value in (-1, 127, True):
        with pytest.raises(ValueError):
            window(value)
    rows = order(17)
    assert rows == order(17) and len(rows) == 100
    assert sum(r["warmup"] for r in rows) == 20
    assert len({(r["trial"], r["method"], r["reader"]) for r in rows}) == 100
    assert phase_window() == "prefill"


def example_rows():
    return [
        dict(
            **r,
            logits_bitwise=True,
            accounting_identical=True,
            instrumented=False,
            total_wall_seconds=1.0 if r["reader"] == "carry_literals" else 2.0,
            decode_wall_seconds=0.8,
            prefill_construct_retire_wall_seconds=0.2,
        )
        for r in order(17)
    ]


def test_paired_statistics_preserve_all_controls_and_separate_quality_N():
    data = summarize(example_rows(), 17)
    assert data["independent_quality_N"] == 0 and not data["paper_superiority_established"]
    for method in METHODS:
        assert data["within_method"][method]["mean_ratio"] == 0.5
        assert data["within_method"][method]["familywise_ratio_bounds"] == [0.5, 0.5]
    for fault in ("duplicate", "missing", "logit", "storage", "instrumented", "nan"):
        rows = example_rows()
        if fault == "duplicate":
            rows[-1] = rows[0]
        elif fault == "missing":
            rows.pop()
        elif fault == "logit":
            rows[-1]["logits_bitwise"] = False
        elif fault == "storage":
            rows[-1]["accounting_identical"] = False
        elif fault == "instrumented":
            rows[-1]["instrumented"] = True
        else:
            rows[-1]["decode_wall_seconds"] = float("nan")
        with pytest.raises(ValueError):
            summarize(rows, 17)


def test_window_instrumentation_counts_actual_launch_branches_and_restores(monkeypatch):
    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation import storage_runs

    def project(*args, **kwargs):
        return dict(
            compiled_reader_branch="carried"
            if reconstruction_policy.carry_literals()
            else "original"
        )

    def decode():
        encoded.project(SimpleNamespace(raw=False))
        return "unchanged"

    adapter = SimpleNamespace(decode_one_token=decode)

    def run(*args, **kwargs):
        for _ in range(127):
            assert adapter.decode_one_token() == "unchanged"
        return "result"

    monkeypatch.setattr(encoded, "project", project)
    monkeypatch.setattr(storage_runs, "run", run)
    timer = WindowTimer()
    with reconstruction_policy.use("carry_literals"), instrument(adapter, timer) as evidence:
        assert storage_runs.run() == "result"
    assert adapter.decode_one_token is decode and storage_runs.run is run
    assert encoded.project is project and phase_window() == "prefill"
    trace = trace_summary(timer, evidence)
    assert trace["steps"] == 127
    assert {x["window"]: x["calls"] for x in trace["branch_counts"]} == dict(
        early=16, middle=80, late=31
    )
    assert {x["branch"] for x in trace["branch_counts"]} == {"carried"}


def test_instrumentation_rejects_wrong_compiled_branch_without_leaking_context(monkeypatch):
    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation import storage_runs

    monkeypatch.setattr(encoded, "project", lambda *a, **k: dict(compiled_reader_branch="original"))
    adapter = SimpleNamespace(decode_one_token=lambda: encoded.project(SimpleNamespace(raw=False)))
    monkeypatch.setattr(storage_runs, "run", lambda *a, **kw: adapter.decode_one_token())
    with (
        pytest.raises(RuntimeError, match="branch mismatch"),
        reconstruction_policy.use("carry_literals"),
        instrument(adapter, WindowTimer()),
    ):
        storage_runs.run()
    assert phase_window() == "prefill" and reconstruction_policy.current() == "original"


def test_execution_modes_cannot_widen_scope():
    from evicastkv_paired_request_readers import mode_config

    plan = dict(paired_reader_stage="reader", reader_modes=["original", "carry_literals"])
    assert mode_config(plan, "original") == ("original", False)
    with pytest.raises(ValueError):
        mode_config(plan, "arbitrary")
    plan = dict(
        paired_reader_stage="dispatch",
        selected_reader="original",
        reader_modes=["original_dispatch", "reused_dispatch"],
    )
    assert mode_config(plan, "reused_dispatch") == ("original", True)


@pytest.mark.parametrize("stage,failed", [("reader", True), ("dispatch", False)])
def test_negative_or_failed_analysis_stops_without_another_campaign(
    monkeypatch, tmp_path, stage, failed
):
    import evicastkv_paired_request_readers as driver

    plan = dict(
        paired_reader_stage=stage,
        reader_modes=["a", "b"],
        selected_reader="original",
        source_sha="sha",
        study_root=str(tmp_path),
    )
    monkeypatch.setattr(driver.common.previous.common, "inputs", lambda *a: plan)
    monkeypatch.setattr(
        driver, "read_producers", lambda *a: ([], [], ["producer failed"] if failed else [])
    )
    monkeypatch.setattr(driver, "submit", lambda *a: pytest.fail("no new GPU submission"))
    monkeypatch.setattr(driver.common.previous, "read", lambda *a: plan)
    writes = []
    monkeypatch.setattr(driver, "write_once", lambda p, d: writes.append((p, d)))
    driver.analyze(tmp_path, "hash")
    assert len(writes) == 2 and writes[-1][0].name == "engineering_closeout.json"
    assert not writes[-1][1]["paper_superiority_established"]


def test_conditional_followup_uses_only_unchanged_consumed_inputs(monkeypatch, tmp_path):
    import json

    import evicastkv_paired_request_readers as driver

    plan = dict(
        paired_reader_stage="reader",
        reader_modes=["original", "carry_literals"],
        source_sha="sha",
        study_root=str(tmp_path),
        seed=7,
        bound_files={},
        jobs=[dict(input_index=0), dict(input_index=1)],
        maximum_dispatch_attempts=1,
        fresh_or_large_campaign_authorized=False,
    )
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(driver.common.previous.common, "inputs", lambda *a: plan)
    summary = dict(within_method=dict(mosaickv_exact=dict(familywise_ratio_bounds=[0.96, 0.98])))
    profiles = [
        dict(
            method="mosaickv_exact",
            phases={"early/text_overlay.projection": dict(exclusive_wall_seconds=0.2)},
        )
        for _ in range(4)
    ]
    monkeypatch.setattr(driver, "read_producers", lambda *a: ([summary, summary], profiles, []))
    monkeypatch.setattr(driver, "reuse_cpu", lambda *a: None)
    calls = []
    monkeypatch.setattr(driver, "submit", lambda p: calls.append(p))
    driver.analyze(tmp_path, "hash")
    assert calls == [tmp_path / "dispatch"]
    child = json.loads((tmp_path / "dispatch/plan.json").read_text())
    assert child["jobs"] == plan["jobs"] and child["selected_reader"] == "carry_literals"
    assert child["paired_reader_stage"] == "dispatch" and child["maximum_dispatch_attempts"] == 1
    assert not child["fresh_or_large_campaign_authorized"]


def test_missing_or_failed_precheck_cannot_submit_model_workers(monkeypatch, tmp_path):
    import evicastkv_paired_request_readers as driver

    monkeypatch.setattr(driver.common.previous.common, "inputs", lambda *a: dict(source_sha="sha"))
    monkeypatch.setattr(driver, "submit", lambda *a: pytest.fail("model GPU work blocked"))
    driver.begin(tmp_path, "hash")
    assert (tmp_path / "precheck/blocked.json").exists()
