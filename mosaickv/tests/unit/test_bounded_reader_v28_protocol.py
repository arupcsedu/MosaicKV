"""CPU-only V28 registration checks; no scheduler, models or data access."""

import ast
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_bounded_reader_v28.py"
    spec = importlib.util.spec_from_file_location("bounded_v28_protocol_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reuse_is_explicit_and_no_observations_or_scientific_gates_removed(protocol):
    assert protocol.POLICY["constructor_snapshot_reuse"] is True
    assert protocol.POLICY["all_ownership_observations_and_deltas_retained"] is True
    assert protocol.POLICY["fresh_snapshot_at_owner_and_lifecycle_transitions"] is True
    assert protocol.POLICY["predictions"] == 128
    assert protocol.POLICY["maximum_A100s"] == 6
    assert protocol.POLICY["fresh_cohort_authorized"] is False
    assert protocol.POLICY["paper_campaign_authorized"] is False


def test_both_model_arms_use_identical_optional_resolver_policy(protocol):
    tree = ast.parse(Path(protocol.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    whole = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "WholeRequestKVLedger"
    ]
    assert len(whole) == 1
    flags = {kw.arg: kw.value for kw in whole[0].keywords}
    assert isinstance(flags["constructor_snapshot_reuse"], ast.Constant)
    assert flags["constructor_snapshot_reuse"].value is True


@pytest.mark.parametrize(
    "mode,cpus,memory,minutes",
    [
        ("tests", 4, "8G", 15),
        ("analyze", 12, "24G", 20),
    ],
)
def test_cpu_only_resources_avoid_gpu_and_long_standard_queue(
    protocol, monkeypatch, tmp_path, mode, cpus, memory, minutes
):
    request = Mock(return_value="1234\n")
    monkeypatch.setattr(protocol.subprocess, "check_output", request)
    protocol.submit(tmp_path, "a" * 40, mode, mode)
    command = request.call_args.args[0]
    assert "--partition=interactive" in command
    assert f"--cpus-per-task={cpus}" in command
    assert f"--mem={memory}" in command
    assert f"--time=00:{minutes}:00" in command
    assert not any(arg.startswith("--gres") for arg in command)


def test_gpu_bounds_reservation_and_afterany_preserved(protocol, monkeypatch, tmp_path):
    request = Mock(return_value="1234\n")
    monkeypatch.setattr(protocol.subprocess, "check_output", request)
    protocol.submit(
        tmp_path, "a" * 40, "gpu", "worker", gpu=True, array=True, dependencies=("123",)
    )
    command = request.call_args.args[0]
    for arg in (
        "--partition=bii-gpu",
        "--reservation=bi_fox_dgx",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--array=0-3%4",
        "--time=00:15:00",
        "--dependency=afterany:123",
    ):
        assert arg in command
