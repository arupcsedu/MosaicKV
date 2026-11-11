import json
import subprocess
from types import SimpleNamespace

import evicastkv_allocator_transport_probe as probe
import evicastkv_review_closure_v2 as runner
import pytest


@pytest.mark.parametrize("seconds", [True, 539, 1801, "1200", 1200.0])
def test_invalid_resource_timeout_fails_before_execution(tmp_path, seconds):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match=r"540\.\.1800"):
        runner.cpu_suite(root, timeout_seconds=seconds)
    assert not root.exists()


@pytest.mark.parametrize("seconds", [540, 1200, 1800])
def test_resource_budget_does_not_filter_tests_or_change_numerics(monkeypatch, tmp_path, seconds):
    monkeypatch.setattr(runner, "clean_source", lambda source: "synthetic")
    monkeypatch.setenv("MOSAICKV_CACHE_ROOT", str(tmp_path / "cache"))
    calls = []

    def successful(command, **kwargs):
        calls.append((command, kwargs))
        path = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        from pathlib import Path

        Path(path).write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", successful)
    root = tmp_path / "complete"
    runner.cpu_suite(root, timeout_seconds=seconds)
    command, options = calls[0]
    assert options["timeout"] == seconds
    assert command[1:5] == ["-m", "pytest", "-q", "mosaickv/tests/unit"]
    assert "-k" not in command and "--deselect" not in command
    assert command.count("-m") == 1  # Python module invocation, not a pytest marker filter.
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["HF_DATASETS_OFFLINE"] == "1"
    record = json.loads((root / "cpu-tests.json").read_text())
    assert record["status"] == "passed" and record["timeout_seconds"] == seconds


def test_timeout_is_durable_failure_not_a_missing_or_passing_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "clean_source", lambda source: "synthetic")
    monkeypatch.setenv("MOSAICKV_CACHE_ROOT", str(tmp_path / "cache"))

    def expired(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", expired)
    root = tmp_path / "timeout"
    with pytest.raises(ValueError, match="timed out"):
        runner.cpu_suite(root, timeout_seconds=1200)
    record = json.loads((root / "cpu-tests.json").read_text())
    assert record["status"] == "failed" and record["returncode"] == "timeout"
    assert record["all_tests_completed"] is False
    assert record["junit_sha256"] is None


def test_frozen_probe_reserves_margin_over_its_python_test_timeout(tmp_path):
    assert probe.POLICY["cpu_suite_timeout_seconds"] == 1200
    assert "--time=00:25:00" in probe.command(tmp_path, "cpu", "tests")
    assert "--time=00:10:00" in probe.command(tmp_path, "probe", "worker")
