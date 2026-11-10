"""Window sensitivity retains frozen controls and an exhaustive registered grid."""

import json
from dataclasses import replace

import evicastkv_window_study as study
import pytest

from mosaickv.config import RunConfig, canonical_config
from mosaickv.evaluation.fox_campaign import source_config


def fixture_config():
    cfg = RunConfig.from_mapping(source_config("coco"))
    return replace(cfg, cache=replace(cfg.cache, budget_value=study.CAP))


@pytest.mark.parametrize("window", [8, 16, 32, 64])
@pytest.mark.parametrize("method", study.METHODS)
def test_window_overlay_changes_only_registered_fields(window, method):
    cfg = fixture_config()
    reference = study.window_config(cfg, method, 16)
    changed = study.window_config(cfg, method, window)
    assert changed.forecasting.prompt_window == window
    assert changed.cache.budget_value == reference.cache.budget_value == 40 * 2**20
    assert canonical_config(
        replace(changed, forecasting=reference.forecasting)
    ) == canonical_config(reference)
    assert cfg.forecasting.prompt_window == 16
    if method == "mosaickv_exact":
        assert changed.graph.uncertain_head_fraction == 1 / 16
        assert changed.prototypes.uncertain_head_fraction == 1 / 16


@pytest.mark.parametrize("bad", [0, -1, 2, 128, True, 8.0, "16"])
def test_unregistered_windows_fail(bad):
    with pytest.raises(ValueError, match="registered integer"):
        study.validate_window(bad)


def test_wrong_cap_or_method_fails():
    cfg = fixture_config()
    with pytest.raises(ValueError, match="exact 40 MiB"):
        study.window_config(
            replace(cfg, cache=replace(cfg.cache, budget_value=20 * 2**20)), "mosaickv_exact", 16
        )
    with pytest.raises(ValueError, match="compressed method"):
        study.window_config(cfg, "full_kv", 16)


def test_grid_is_complete_deterministic_and_fullkv_is_not_duplicated():
    jobs = study.jobs()
    assert jobs == study.jobs()
    assert len(jobs) == 16
    assert [i for job in jobs for i in job["indices"]] == list(range(64))
    assert all(len(job["indices"]) == 4 and len(job["arms"]) == 9 for job in jobs)
    assert sum(method == "full_kv" for method, _, _ in study.arms()) == 1
    assert len(set(tuple(arm) for arm in study.arms())) == 9
    assert len(study.comparisons()) == len(set(study.comparisons())) == 7
    assert sum(len(job["indices"]) * len(job["arms"]) for job in jobs) == 576


def test_worker_overlay_is_isolated_and_fullkv_unchanged(monkeypatch):
    import evicastkv_worker as worker

    cfg = fixture_config()
    calls = []

    def original(lock, workload, method, cap, *, variant="complete"):
        calls.append((method, variant))
        return cfg

    monkeypatch.setattr(worker, "resolved_config", original)
    with study.config_overlay():
        result = worker.resolved_config({}, "coco", "mosaickv_exact", 40, variant="window-32")
        assert result.forecasting.prompt_window == 32
        assert result.graph.uncertain_head_fraction == 1 / 16
        assert worker.resolved_config({}, "coco", "full_kv", None) is cfg
        with pytest.raises(ValueError, match="unregistered"):
            worker.resolved_config({}, "coco", "mosaickv_exact", 20, variant="window-16")
    assert worker.resolved_config is original
    assert calls == [("mosaickv_exact", "complete"), ("full_kv", "complete")]


def test_partial_grid_cannot_be_analyzed():
    with pytest.raises(ValueError, match="incomplete"):
        study.validate_rows([], ["consumed"])


@pytest.fixture
def valid_rows(tmp_path):
    rows = []
    cfg = fixture_config()
    for index, (method, cap, variant) in enumerate(study.arms()):
        current = (
            cfg
            if method == "full_kv"
            else study.window_config(cfg, method, int(variant.removeprefix("window-")))
        )
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(canonical_config(current)))
        trace_path = tmp_path / f"{index}-trace.json"
        trace_path.write_text(
            json.dumps(
                dict(
                    source_kv_layout=[
                        dict(
                            key_shape=[1, 8, 24, 128],
                            value_shape=[1, 8, 24, 128],
                        )
                    ],
                    forecast_statistics=dict(
                        provenance=dict(
                            requested_prompt_window=current.forecasting.prompt_window,
                            actual_prompt_window=min(24, current.forecasting.prompt_window),
                            original_logical_sequence_length=24,
                        )
                    )
                    if method == "mosaickv_exact"
                    else {},
                )
            )
        )
        rows.append(
            dict(
                sample_id="consumed",
                method=method,
                effective_method=method,
                emitted_tokens=64,
                forward_steps=63,
                variant=variant,
                status="completed",
                input_sha="fixed-input",
                processor_sha="fixed-processor",
                scorer_sha="fixed-scorer",
                references=["synthetic fixed reference one", "synthetic reference two"],
                budget_bytes=study.CAP if cap else None,
                initial_kv_bytes=study.CAP,
                final_kv_bytes=study.CAP + 100,
                peak_kv_bytes=study.CAP + 100,
                resolved_config_path=str(path),
                resolved_config_file_sha256=study.sha(path),
                trace_path=str(trace_path),
                trace_sha256=study.sha(trace_path),
            )
        )
    return rows


def test_actual_windows_and_measured_storage_are_validated(valid_rows):
    windows = study.validate_rows(valid_rows, ["consumed"])
    assert len(windows) == 8
    assert {row["resolved_prompt_window"] for row in windows} == set(study.WINDOWS)
    assert {row["actual_prompt_window"] for row in windows if row["actual_window_measured"]} == {
        8,
        16,
        24,
    }
    assert {row["source_sequence_length"] for row in windows} == {24}
    assert {row["effective_prompt_window"] for row in windows} == {8, 16, 24}
    assert all(row["effective_prompt_window"] <= row["requested_prompt_window"] for row in windows)
    topk = [row for row in windows if row["method"] == "prompt_attention_topk"]
    assert all(
        row["actual_prompt_window"] is None and not row["actual_window_measured"] for row in topk
    )
    assert all("not measured" in row["effective_window_provenance"] for row in topk)


def test_duplicate_fullkv_reference_blocks_analysis(valid_rows):
    with pytest.raises(ValueError, match="duplicate"):
        study.validate_rows([*valid_rows, valid_rows[0]], ["consumed"])


def test_measured_final_byte_mismatch_blocks_analysis(valid_rows):
    valid_rows[-1]["final_kv_bytes"] += 1
    with pytest.raises(ValueError, match="realized initial/final/peak"):
        study.validate_rows(valid_rows, ["consumed"])


def test_fallback_is_explicitly_rejected(valid_rows):
    valid_rows[-1]["effective_method"] = "full_kv"
    with pytest.raises(ValueError, match="fallback"):
        study.validate_rows(valid_rows, ["consumed"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size", 4),
        ("physical_budget_block_size", 4),
        ("physical_budget_mandatory_policy", "first_and_last_token"),
        ("accounting_spec_sha", "changed-accounting-boundary"),
    ],
)
def test_rehashed_cache_policy_change_is_rejected(valid_rows, field, value):
    from pathlib import Path

    row = valid_rows[-1]
    path = Path(row["resolved_config_path"])
    config = json.loads(path.read_text())
    config["cache"][field] = value
    path.write_text(json.dumps(config))
    row["resolved_config_file_sha256"] = study.sha(path)
    with pytest.raises(ValueError, match="cache/block/mandatory policy"):
        study.validate_rows(valid_rows, ["consumed"])


def test_reference_annotation_mismatch_is_rejected(valid_rows):
    valid_rows[-1]["references"] = ["different reference"]
    with pytest.raises(ValueError, match="reference annotations differ"):
        study.validate_rows(valid_rows, ["consumed"])


def test_rehashed_source_length_mismatch_is_rejected(valid_rows):
    from pathlib import Path

    row = valid_rows[-1]
    path = Path(row["trace_path"])
    trace = json.loads(path.read_text())
    trace["source_kv_layout"][0]["key_shape"][2] = 23
    trace["source_kv_layout"][0]["value_shape"][2] = 23
    path.write_text(json.dumps(trace))
    row["trace_sha256"] = study.sha(path)
    with pytest.raises(ValueError, match="source length differs"):
        study.validate_rows(valid_rows, ["consumed"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_prompt_window", 32),
        ("requested_prompt_window", 32),
        ("original_logical_sequence_length", 32),
    ],
)
def test_rehashed_forecast_provenance_mismatch_is_rejected(valid_rows, field, value):
    from pathlib import Path

    row = valid_rows[-1]
    assert row["variant"] == "window-64" and row["method"] == "mosaickv_exact"
    path = Path(row["trace_path"])
    trace = json.loads(path.read_text())
    trace["forecast_statistics"]["provenance"][field] = value
    path.write_text(json.dumps(trace))
    row["trace_sha256"] = study.sha(path)
    with pytest.raises(ValueError, match="trace forecast window/request/source length mismatch"):
        study.validate_rows(valid_rows, ["consumed"])
