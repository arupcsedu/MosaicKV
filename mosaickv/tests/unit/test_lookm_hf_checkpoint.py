"""Prepared real-checkpoint controls using synthetic CPU-only fixture metadata."""

import copy
import json
import sys
from types import SimpleNamespace

import evicastkv_lookm_hf_checkpoint as study
import pytest
import torch


def payload():
    return {
        "task": "consumed",
        "sample_id": "synthetic",
        "prompt": "describe",
        "media": [{"kind": "image", "path": "not_opened", "sha256": "bound"}],
    }


def test_payload_rejects_reference_access(tmp_path):
    path = tmp_path / "input.json"
    value = payload()
    value["references"] = ["forbidden"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Only consumed"):
        study.blind_payload(path)


def test_single_consumed_image_route_only(tmp_path):
    path = tmp_path / "input.json"
    value = payload()
    path.write_text(json.dumps(value))
    assert study.blind_payload(path) == value
    value["media"][0]["kind"] = "video"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="one hash-bound"):
        study.blind_payload(path)


def test_resource_and_claim_bounds_are_explicit():
    assert (study.POLICY["a100s"], study.POLICY["cpus"], study.POLICY["wall_seconds"]) == (
        1,
        12,
        1200,
    )
    assert study.TOKENS >= 16 and study.ATOL == 1e-4
    assert study.POLICY["no_downloads"] and study.POLICY["no_benchmark_labels"]
    assert study.POLICY["same_checkpoint_as_LOOKM_paper_not_verified"]
    assert study.GATE == "GATE_lookm_hf_checkpoint_compat.json"


def test_unique_storage_bytes_include_backing_not_only_views():
    backing = torch.zeros(100)
    a, b = backing[:8], backing[8:16]
    assert study.live_cache_bytes(((a, b),)) == 400
    assert study.live_cache_bytes(((a, b.clone()),)) == 432


def snapshot(*, error=None, other=False):
    return SimpleNamespace(
        query_error=error,
        devices=[SimpleNamespace(uuid="GPU-A")],
        processes=[SimpleNamespace(gpu_uuid="GPU-A", pid=10)]
        + ([SimpleNamespace(gpu_uuid="GPU-A", pid=11)] if other else []),
    )


def test_assigned_gpu_isolation_rejects_unknown_or_background_processes():
    assert study.assigned_isolation(snapshot(), "GPU-A", 10)["assigned_gpu_isolation_verified"]
    for evidence in (snapshot(error="unknown"), snapshot(other=True)):
        assert not study.assigned_isolation(evidence, "GPU-A", 10)[
            "assigned_gpu_isolation_verified"
        ]
    assert not study.assigned_isolation(snapshot(), "GPU-B", 10)["assigned_gpu_isolation_verified"]


def tensors():
    names = (
        "fullkv_0",
        "fullkv_1",
        "fullkv_explicit_0",
        "fullkv_explicit_1",
        "compressed_A",
        "compressed_B",
        "compressed_A_repeat",
        "fullkv_A_restored",
    )
    return {
        name: {"tokens": torch.zeros((1, 16), dtype=torch.long), "logits": torch.zeros((16, 4))}
        for name in names
    }


def test_saved_checks_recompute_fullkv_and_isolation_without_decoding_answers():
    values = tensors()
    study.verify_saved_tensors(values, expected_vocab=4)
    values["fullkv_explicit_1"]["logits"][0, 0] = 0.01
    with pytest.raises(ValueError, match="parity failed"):
        study.verify_saved_tensors(values, expected_vocab=4)


def test_missing_rows_are_not_intersected():
    values = tensors()
    del values["compressed_B"]
    with pytest.raises(ValueError, match="no intersection"):
        study.verify_saved_tensors(values, expected_vocab=4)


def test_saved_aba_guard_is_independent():
    values = tensors()
    values["compressed_A_repeat"]["tokens"][0, 0] = 1
    with pytest.raises(ValueError, match="isolation"):
        study.verify_saved_tensors(values, expected_vocab=4)


def rows():
    result = []
    for arm in (
        "fullkv_0",
        "fullkv_1",
        "compressed_A",
        "compressed_B",
        "compressed_A_repeat",
        "fullkv_A_restored",
    ):
        length = 8 if arm.startswith("compressed") else 16
        calls = [
            {
                "position_last": 15 + step,
                "physical_lengths": [length + step] * 32,
                "active_bytes": 1024 + step,
                "tensor_payload_bytes": 500,
            }
            for step in range(16)
        ]
        result.append(
            {
                "arm": arm,
                "complete": True,
                "logical_prompt_length": 16,
                "calls": calls,
                "initial_kv_bytes": 1024,
                "final_kv_bytes": 1039,
                "observed_decode_output_peak_bytes": 1039,
                "attention_class": (
                    "TextPivotLlamaAttention_drop"
                    if arm.startswith("compressed")
                    else "LlamaAttention"
                ),
                "kv_mode": study.POLICY["compressed_mode"]
                if arm.startswith("compressed")
                else "origin",
                "source_native_rotary_guard": {
                    "calls_per_layer": [16] * 32,
                    "maximum_length_per_layer": [31] * 32,
                    "registered_required_length": 32,
                    "prebuilt_length": 2048,
                },
            }
        )
    return result


def test_trace_reducer_checks_layout_positions_and_accounting():
    values = rows()
    study.verify_rows(values)
    changed = copy.deepcopy(values)
    changed[2]["calls"][1]["position_last"] = 8
    with pytest.raises(ValueError, match="renumbered"):
        study.verify_rows(changed)
    changed = copy.deepcopy(values)
    changed[2]["final_kv_bytes"] += 8
    with pytest.raises(ValueError, match="accounting differs"):
        study.verify_rows(changed)


def test_saved_rotary_guard_cannot_be_omitted_or_rebounded():
    values = rows()
    study.verify_rows(values, expected_required_length=32)
    values[0]["source_native_rotary_guard"]["calls_per_layer"][0] = 1
    with pytest.raises(ValueError, match="rotary guard"):
        study.verify_rows(values, expected_required_length=32)


def test_prepare_requires_current_cpu_gate_before_any_dataset_or_weight_access(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(study, "clean_source", lambda _: "immutable")
    with pytest.raises(ValueError, match="CPU gate root"):
        study.prepare(tmp_path / "new")


def test_failed_synthetic_compatibility_blocks_before_checkpoint_access(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "registration.json").write_text(
        json.dumps({"source_sha": "immutable", "execute": True})
    )
    monkeypatch.setattr(study, "clean_source", lambda _: "immutable")
    monkeypatch.setattr(study, "cpu_gate", lambda *_: {"status": "passed"})
    monkeypatch.setattr(
        study.transport, "run", lambda *_: {"status": "blocked_at_recorded_interface"}
    )
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="synthetic compatibility"):
        study.prepare(root, cpu_gate_root=parent)
    assert (root / "cpu_compatibility.json").is_file()
    assert not (root / "protocol.json").exists()


def test_failed_interface_is_saved_without_gpu_execution_or_dropped_denominator(
    tmp_path, monkeypatch
):
    root = tmp_path / "diag"
    root.mkdir()
    (root / "protocol.json").write_text("{}")
    monkeypatch.setattr(
        study,
        "validate",
        lambda _: {
            "source_sha": "synthetic",
            "gpu_execution_authorized": True,
        },
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic_not_submitted")
    monkeypatch.delitem(sys.modules, "llava", raising=False)

    def rejected(*_args, **_kwargs):
        raise AttributeError("recorded_forward_api_boundary")

    monkeypatch.setattr(study, "_checkpoint_run", rejected)
    result = study.worker(root)
    assert result["complete"] is False
    assert result["errors"][0]["message"] == "recorded_forward_api_boundary"
    assert (root / "worker/result.json").is_file()
    gate = study.analyze(root)
    assert gate["status"] == "blocked"
    assert gate["superiority_established"] is False
