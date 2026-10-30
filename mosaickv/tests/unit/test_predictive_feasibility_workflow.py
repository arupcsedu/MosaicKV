import json

import evicastkv_predictive_feasibility as module
import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.types import BudgetUnit


def fixture_root(tmp_path, monkeypatch):
    plan = dict(
        source_sha="fixture",
        jobs=[dict(index=i, task=f"synthetic-{i}", sample_id=str(i)) for i in range(2)],
        predictor_kinds=["none", "xor16", "ordered_delta16"],
        codec_kinds=["bf16_exponents", "sparse_high_byte"],
        predictor_header_bytes=4,
        minimum_evicast_prospective_reduction=0.03,
    )
    monkeypatch.setattr(module.base.common, "inputs", lambda root, expected: plan)
    for index in range(2):
        parent = tmp_path / f"workers/{index}"
        parent.mkdir(parents=True)
        outputs = {}
        for method in module.paired.METHODS:
            tensor = torch.randn(1, 2, 16, 32).bfloat16()
            packet = pack_cpu(tensor)
            size = sum(t.untyped_storage().nbytes() for t in packet.buffers)
            artifact = dict(
                method=method,
                selection=dict(budget_unit=BudgetUnit.BYTES),
                bodies=[
                    dict(
                        layer=0,
                        body_index=0,
                        role="synthetic",
                        shape=packet.shape,
                        raw=packet.raw,
                        buffers=packet.buffers,
                        original_GPU_payload_bytes=size,
                    )
                ],
                original_GPU_initial_component_bytes=size + 20,
                original_GPU_payload_bytes=size,
                non_payload_bytes=20,
            )
            path = parent / f"{method}-codec-input.pt"
            torch.save(artifact, path)
            outputs[path.name] = module.sha(path)
        (parent / "result.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    failures=[],
                    source_sha="fixture",
                    plan_sha256="fixture-plan",
                    outputs=outputs,
                )
            )
        )
    return plan


def test_cpu_worker_and_afterany_aggregation_keep_all_methods_and_exact_bits(tmp_path, monkeypatch):
    allowed_before = set(torch.serialization.get_safe_globals())
    fixture_root(tmp_path, monkeypatch)
    for index in range(2):
        module.codec_worker(tmp_path, "fixture-plan", index)
    module.aggregate(tmp_path, "fixture-plan")
    gate = json.loads((tmp_path / "codec_analysis/GATE.json").read_text())
    assert gate["status"] == "complete_CPU_feasibility"
    assert len(gate["rows"]) == 12
    assert not gate["automatic_GPU_submission"]
    assert all(row["all_tensor_bits_preserved"] for row in gate["rows"])
    assert all(
        row["prospective_installed_bytes_NOT_GPU_verified"]
        <= row["original_measured_GPU_initial_bytes"]
        for row in gate["rows"]
    )
    assert set(torch.serialization.get_safe_globals()) == allowed_before


def test_changed_capture_is_rejected_with_structured_worker_failure(tmp_path, monkeypatch):
    fixture_root(tmp_path, monkeypatch)
    path = tmp_path / "workers/0/full_kv-codec-input.pt"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.codec_worker(tmp_path, "fixture-plan", 0)
    result = json.loads((tmp_path / "codec_workers/0/result.json").read_text())
    assert result["status"] == "failed"
    assert result["failures"]


def test_cpu_launcher_never_requests_a_GPU(tmp_path, monkeypatch):
    fixture_root(tmp_path, monkeypatch)
    calls = []

    def submit(args, **kwargs):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(module.subprocess, "check_output", submit)
    module.launch_cpu(tmp_path, "fixture-plan")
    assert len(calls) == 2
    assert all(not any("--gres" in arg or "bii-gpu" in arg for arg in args) for args in calls)
    assert "--dependency=afterany:101" in calls[1]
