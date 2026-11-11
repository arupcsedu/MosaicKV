"""Global capture can change without changing CPU tests or numerical controls."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import evicastkv_review_closure_v2 as study
import pytest


def configure(monkeypatch, tmp_path, *, returncode=0, timeout=False):
    calls = []
    monkeypatch.setenv("MOSAICKV_CACHE_ROOT", str(tmp_path / "scratch-cache"))
    monkeypatch.setattr(study, "clean_source", lambda _source: "a" * 40)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("native stdout retained in the outer log\n")
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        junit = next(
            part.removeprefix("--junitxml=")
            for part in command
            if part.startswith("--junitxml=")
        )
        Path(junit).write_text('<testsuites><testsuite tests="1" failures="0"/></testsuites>')
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(study.subprocess, "run", run)
    return calls


def test_default_capture_preserves_historical_command_and_timeout(monkeypatch, tmp_path):
    calls = configure(monkeypatch, tmp_path)
    root = tmp_path / "default"
    study.cpu_suite(root)
    command, options = calls[0]
    assert command == [
        sys.executable, "-m", "pytest", "-q", "mosaickv/tests/unit", "-o",
        f"cache_dir={tmp_path / 'scratch-cache'}/pytest-fox-closure-full",
        f"--junitxml={root}/cpu-tests.xml",
    ]
    assert options["timeout"] == 540
    record = json.loads((root / "cpu-tests.json").read_text())
    assert record["status"] == "passed"
    assert record["capture_mode"] == "fd"


def test_sys_capture_changes_only_the_explicit_capture_argument(monkeypatch, tmp_path):
    calls = configure(monkeypatch, tmp_path)
    root = tmp_path / "sys"
    study.cpu_suite(root, timeout_seconds=1200, capture_mode="sys")
    command, options = calls[0]
    assert command[-1] == "--capture=sys"
    assert command.count("--capture=sys") == 1
    assert "mosaickv/tests/unit" in command
    assert options["timeout"] == 1200
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    assert options["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert options["stderr"] == subprocess.STDOUT
    record = json.loads((root / "cpu-tests.json").read_text())
    assert record["capture_mode"] == "sys"
    assert record["command"] == command
    assert "native stdout retained" in (root / "cpu-tests.log").read_text()


@pytest.mark.parametrize("mode", ("no", "tee-sys", "FD", None, True))
def test_invalid_capture_fails_before_running_or_creating_artifacts(monkeypatch, tmp_path, mode):
    calls = configure(monkeypatch, tmp_path)
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="capture mode"):
        study.cpu_suite(root, capture_mode=mode)
    assert not calls
    assert not root.exists()


@pytest.mark.parametrize("timeout", (False, True))
def test_sys_capture_preserves_failure_and_timeout_receipts(monkeypatch, tmp_path, timeout):
    configure(monkeypatch, tmp_path, returncode=1, timeout=timeout)
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match=r"timed out|suite failed"):
        study.cpu_suite(root, timeout_seconds=1200, capture_mode="sys")
    record = json.loads((root / "cpu-tests.json").read_text())
    assert record["capture_mode"] == "sys"
    assert record["status"] == "failed"
    assert record["returncode"] == ("timeout" if timeout else 1)
    assert record["timeout_seconds"] == 1200
    assert record["models_or_data_downloaded"] is False
