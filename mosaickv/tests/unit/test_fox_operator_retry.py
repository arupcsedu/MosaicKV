from pathlib import Path

import pytest


def test_legacy_arithmetic_default_matches_original_worker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from evicastkv_operator_localization import native_reduction_policy

    assert native_reduction_policy({}) is True
    assert (
        native_reduction_policy(dict(arithmetic=dict(allow_fp16_reduced_precision_reduction=False)))
        is False
    )
    with pytest.raises(ValueError):
        native_reduction_policy(
            dict(arithmetic=dict(allow_fp16_reduced_precision_reduction="false"))
        )


def test_retry_schedules_only_missing_milebench_cases(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_operator_localization as module

    from mosaickv.evaluation.fox_campaign import sha, write_once

    parent, numerical, retry = tmp_path / "incomplete", tmp_path / "numerical", tmp_path / "retry"
    bindings = []
    for index in range(2):
        path = parent / f"case-{index}.json"
        write_once(path, dict(workload="coco", sample_id=str(index)))
        bindings.append(dict(path=str(path), sha256=sha(path)))
    write_once(
        parent / "results.json", dict(cases=bindings, errors=[dict(error="KeyError: 'arithmetic'")])
    )
    write_once(numerical / "results.json", dict(cases=[]))
    monkeypatch.setattr(module, "clean_source", lambda *a: "synthetic")
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        return str(100 + len(commands))

    monkeypatch.setattr(module.subprocess, "check_output", submit)
    module.submit(retry, numerical, incomplete_parent=parent)
    protocol = json.loads((retry / "protocol.json").read_text())
    assert protocol["enabled_workloads"] == ["milebench"]
    assert protocol["inherited_completed_diagnostic"]["completed_coco_cases_preserved"] == bindings
    assert len(commands) == 2 and "--dependency=afterany:101" in commands[1]
