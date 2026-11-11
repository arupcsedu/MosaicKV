"""CPU-only synthetic protocol/scope fixtures; no pretrained numerical claims."""

import json
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import evicastkv_coco_boundary_v28 as protocol
import pytest
import torch

from mosaickv.adapters.huggingface.types import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    PreparedInputs,
)
from mosaickv.cache_state import Modality, ModalitySpan


def whole_gate():
    return dict(
        source_sha="a" * 40,
        failures=[],
        complete=True,
        source_fidelity_128_pass=True,
        whole_request_KV_byte_match_pass=True,
    )


def counts(layers=32):
    return dict(
        observed_prefill_layers=layers,
        attention_padded_calls=layers,
        linear_padded_calls=7 * layers,
        normalization_expected_calls=2 * layers + 1,
        normalization_padded_calls=2 * layers + 1,
        normalization_input_calls=layers,
        normalization_post_attention_calls=layers,
        normalization_final_calls=1,
        normalization_phase_entries=1,
        normalization_phase_completions=1,
    )


def trajectory(predictions=128, layers=32):
    logits = torch.zeros(predictions, 11, dtype=torch.float32)
    logits[:, 0] = 1.0
    components = [
        dict(
            component=kind,
            shape=[1, 8, 7, 128],
            dtype="torch.float16",
            finite=True,
            reference_sha256="a" * 64,
            candidate_sha256="a" * 64,
            bitwise_equal=True,
            maximum_absolute_difference=0.0,
        )
        for kind in ("K", "V")
    ]
    initial = dict(
        layers=[dict(layer=i, components=deepcopy(components)) for i in range(layers)],
        bitwise_equal=True,
        maximum_absolute_difference=0.0,
    )
    final = deepcopy(initial)
    for layer in final["layers"]:
        for component in layer["components"]:
            component["shape"][2] = 7 + predictions - 1
    return dict(
        reference_logits=logits,
        candidate_logits=logits.clone(),
        initial_cache=initial,
        final_cache=final,
        operation_counts=counts(layers),
        ordinary_methods_restored_before_continuation=True,
        held_token_forward_count=1,
        ordinary_continuation_steps=predictions - 1,
        original_prompt_length=7,
        final_logical_position=7 + predictions - 1,
        reference_prefix_forced_on_both_paths=True,
    )


def test_whole_request_prerequisite_is_required():
    protocol.require_whole_gate(whole_gate(), "a" * 40)


@pytest.mark.parametrize(
    "field", ["complete", "source_fidelity_128_pass", "whole_request_KV_byte_match_pass"]
)
@pytest.mark.parametrize("value", [False, None, 1])
def test_no_scope_advance_from_failed_or_coerced_gate(field, value):
    gate = whole_gate()
    gate[field] = value
    with pytest.raises(ValueError, match="prerequisite"):
        protocol.require_whole_gate(gate, "a" * 40)


def test_whole_gate_must_share_clean_source():
    with pytest.raises(ValueError, match="immutable"):
        protocol.require_whole_gate(whole_gate(), "b" * 40)


def test_whole_gate_errors_are_not_ignored():
    gate = whole_gate()
    gate["failures"] = ["incomplete"]
    with pytest.raises(ValueError):
        protocol.require_whole_gate(gate, "a" * 40)


def test_git_proves_every_original_producer_file_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        protocol.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "A\0mosaickv/scripts/new_CPU_tool.py\0",
    )
    monkeypatch.setattr(protocol, "sha", lambda _path: "f" * 64)
    monkeypatch.setattr(protocol, "read", lambda _path: dict(implementation_sha256="f" * 64))
    proof = protocol.execution_base_proof("a" * 40)
    assert proof["actual_GPU_execution_base_sha"] == protocol.BRIDGE_PRODUCER
    assert proof["new_control_source_sha"] == "a" * 40
    assert proof["all_original_producer_files_byte_identical"] is True
    assert proof["added_paths"] == ["mosaickv/scripts/new_CPU_tool.py"]
    assert calls == [
        ["git", "diff", "--name-status", "-z", "--no-renames", protocol.BRIDGE_PRODUCER, "a" * 40]
    ]


@pytest.mark.parametrize(
    "changes",
    [
        "M\0mosaickv/src/mosaickv/adapters/huggingface/base.py\0",
        "D\0mosaickv/src/mosaickv/evaluation/fox_shape_transport.py\0",
        "R100\0old.py\0new.py\0",
        "A\0unterminated.py",
        "A\0",
    ],
)
def test_changed_or_incomplete_base_source_proof_rejected(monkeypatch, changes):
    monkeypatch.setattr(protocol.subprocess, "check_output", lambda *_args, **_kwargs: changes)
    with pytest.raises(ValueError):
        protocol.execution_base_proof("a" * 40)


def test_old_v9_transport_hash_remains_required(monkeypatch):
    monkeypatch.setattr(protocol.subprocess, "check_output", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(protocol, "sha", lambda _path: "a" * 64)
    monkeypatch.setattr(protocol, "read", lambda _path: dict(implementation_sha256="b" * 64))
    with pytest.raises(ValueError, match="V9 transport"):
        protocol.execution_base_proof("a" * 40)


def bridge_fixture(monkeypatch, tmp_path):
    """Synthetic files exercise provenance controls, not measured GPU evidence."""
    root = tmp_path / "actual-producer"
    (root / "analysis").mkdir(parents=True)
    producer = protocol.BRIDGE_PRODUCER
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(dict(source_sha=producer)))
    plan_sha = protocol.sha(plan_path)
    original_gate = whole_gate() | dict(source_sha=producer, plan_sha256=plan_sha)
    gate_path = root / "analysis" / protocol.BRIDGE_GATE_NAME
    gate_path.write_text(json.dumps(original_gate))
    monkeypatch.setattr(protocol, "BRIDGE_ROOT", root)
    monkeypatch.setattr(protocol, "BRIDGE_PLAN_SHA", plan_sha)
    return whole_gate() | dict(
        source_sha=producer,
        actual_execution_source_sha=producer,
        validation_source_sha="a" * 40,
        gate_origin="derived_validation_not_new_GPU_execution",
        independent_replay_from_actual_GPU_artifacts=True,
        actual_execution_root=str(root),
        actual_execution_root_sha256=protocol.digest(str(root.resolve())),
        actual_plan_path=str(plan_path),
        actual_plan_sha256=plan_sha,
        actual_gate_path=str(gate_path),
        actual_gate_sha256=protocol.sha(gate_path),
    )


def test_pinned_original_execution_can_use_stronger_derived_validation(monkeypatch, tmp_path):
    gate = bridge_fixture(monkeypatch, tmp_path)
    bindings = protocol.require_whole_gate(gate, "a" * 40)
    assert len(bindings) == 2
    assert gate["source_sha"] == protocol.BRIDGE_PRODUCER
    assert gate["source_sha"] != gate["validation_source_sha"]


@pytest.mark.parametrize(
    "field",
    [
        "actual_execution_source_sha",
        "validation_source_sha",
        "gate_origin",
        "independent_replay_from_actual_GPU_artifacts",
        "actual_execution_root",
        "actual_execution_root_sha256",
        "actual_plan_path",
        "actual_plan_sha256",
        "actual_gate_path",
        "actual_gate_sha256",
    ],
)
def test_derived_gate_requires_every_explicit_provenance_field(monkeypatch, tmp_path, field):
    gate = bridge_fixture(monkeypatch, tmp_path)
    del gate[field]
    with pytest.raises(ValueError):
        protocol.require_whole_gate(gate, "a" * 40)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "b" * 40),
        ("actual_execution_source_sha", "b" * 40),
        ("validation_source_sha", "b" * 40),
        ("gate_origin", "new_GPU_execution"),
        ("independent_replay_from_actual_GPU_artifacts", 1),
        ("actual_execution_root_sha256", "b" * 64),
        ("actual_plan_sha256", "b" * 64),
        ("actual_gate_sha256", "b" * 64),
    ],
)
def test_derived_gate_cannot_fabricate_sources_origin_or_bindings(
    monkeypatch, tmp_path, field, value
):
    gate = bridge_fixture(monkeypatch, tmp_path)
    gate[field] = value
    with pytest.raises(ValueError):
        protocol.require_whole_gate(gate, "a" * 40)


def test_derived_gate_cannot_be_relabelled_as_current_source(monkeypatch, tmp_path):
    gate = bridge_fixture(monkeypatch, tmp_path)
    gate["source_sha"] = "a" * 40
    with pytest.raises(ValueError, match="source provenance"):
        protocol.require_whole_gate(gate, "a" * 40)


def test_actual_producer_output_mutation_invalidates_bridge(monkeypatch, tmp_path):
    gate = bridge_fixture(monkeypatch, tmp_path)
    path = protocol.Path(gate["actual_gate_path"])
    original = json.loads(path.read_text())
    original["source_fidelity_128_pass"] = False
    path.write_text(json.dumps(original))
    # Even recomputing the declared content hash cannot turn a failed producer
    # into successful actual execution.
    gate["actual_gate_sha256"] = protocol.sha(path)
    with pytest.raises(ValueError, match="did not pass"):
        protocol.require_whole_gate(gate, "a" * 40)


def test_128_predictions_are_recomputed_not_inherited_32_counts():
    result = protocol.summarize(trajectory())
    assert result["pass_128"] is True
    assert result["predictions"] == result["prediction_agreement"] == 128
    assert result["ordinary_continuation_steps"] == 127
    assert result["maximum_logit_difference"] == 0.0
    assert result["unchanged_absolute_tolerance"] == 1e-4


@pytest.mark.parametrize("change", ["token", "logit", "initial_cache", "final_cache"])
def test_numerical_loss_is_negative_not_hidden(change):
    record = trajectory()
    if change == "token":
        record["candidate_logits"][100, 1] = 2.0
    elif change == "logit":
        record["candidate_logits"][110, 0] += 0.001
    else:
        record[change]["layers"][5]["components"][0].update(
            candidate_sha256="b" * 64, bitwise_equal=False, maximum_absolute_difference=0.25
        )
        record[change].update(bitwise_equal=False, maximum_absolute_difference=0.25)
    assert protocol.summarize(record)["pass_128"] is False


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "short",
        "scope",
        "norm_calls",
        "attention_calls",
        "cache_hash",
        "cache_coverage",
        "position",
    ],
)
def test_invalid_scope_or_incomplete_evidence_rejected(change):
    record = trajectory()
    if change == "nan":
        record["candidate_logits"][2, 2] = float("nan")
    elif change == "short":
        record["candidate_logits"] = record["candidate_logits"][:16]
    elif change == "scope":
        record["ordinary_methods_restored_before_continuation"] = False
    elif change == "norm_calls":
        record["operation_counts"]["normalization_padded_calls"] += 1
    elif change == "attention_calls":
        record["operation_counts"]["attention_padded_calls"] += 127 * 32
    elif change == "cache_hash":
        record["initial_cache"]["layers"][0]["components"][0]["candidate_sha256"] = "b" * 64
    elif change == "cache_coverage":
        record["final_cache"]["layers"].pop()
    else:
        record["final_logical_position"] += 1
    with pytest.raises(ValueError):
        protocol.summarize(record)


def test_registered_no_claim_and_resource_scopes():
    assert protocol.POLICY["tolerance"] == 1e-4
    assert protocol.POLICY["predictions"] == 128
    assert protocol.POLICY["ordinary_continuation_steps"] == 127
    assert protocol.POLICY["workers"] == 2
    assert protocol.POLICY["gpus_per_worker"] == 1
    assert protocol.POLICY["cpus_per_worker"] == 12
    assert protocol.POLICY["worker_minutes"] == 15
    for field in (
        "fresh_confirmation_authorized",
        "model_source_reconstruction_enabled",
        "scoring",
        "headline_timing",
        "whole_request_peak_claim",
        "paper_campaign_authorized",
    ):
        assert protocol.POLICY[field] is False


@pytest.mark.parametrize("change", ["shape", "dtype", "boolean", "hash", "length"])
def test_cache_metadata_cannot_forge_pinned_geometry(change):
    record = trajectory()
    item = record["final_cache"]["layers"][0]["components"][0]
    if change == "shape":
        item["shape"][1] = 32
    elif change == "dtype":
        item["dtype"] = "torch.bfloat16"
    elif change == "boolean":
        item["bitwise_equal"] = 1
    elif change == "hash":
        item["reference_sha256"] = item["candidate_sha256"] = "not-a-hash"
    else:
        item["shape"][2] -= 1
    with pytest.raises(ValueError, match="geometry/hash"):
        protocol.summarize(record)


def test_submission_is_one_gpu_two_element_array_and_afterany_cpu(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        protocol.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "123456\n",
    )
    root = tmp_path / "run"
    (root / "jobs").mkdir(parents=True)
    job = protocol.submit(root, "a" * 40, "gpu", "worker", gpu=True)
    protocol.submit(root, "a" * 40, "analysis", "analyze", dependency=job)
    assert "--array=0-1%2" in calls[0]
    assert "--gres=gpu:a100:1" in calls[0]
    assert "--reservation=bi_fox_dgx" in calls[0]
    assert "--cpus-per-task=12" in calls[0]
    assert "--time=00:15:00" in calls[0]
    assert "--dependency=afterany:123456" in calls[1]
    assert not any(arg.startswith("--gres=") for arg in calls[1])


def test_uncertain_submission_does_not_automatically_retry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        protocol.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "unknown",
    )
    root = tmp_path / "run"
    (root / "jobs").mkdir(parents=True)
    with pytest.raises(ValueError, match="uncertain"):
        protocol.submit(root, "a" * 40, "gpu", "worker", gpu=True)
    assert len(calls) == 1


def test_single_boundary_then_127_native_continuations_and_restored_methods(monkeypatch):
    """Synthetic adapter tests scope/loop order, not pretrained numerical parity."""
    from mosaickv.evaluation import fox_shape_transport as transport

    active = [False]
    calls = []

    class Operations(dict):
        @contextmanager
        def boundary_normalization(self):
            yield

    @contextmanager
    def shape(adapter, rows, *, native_norm_boundary=False):
        assert rows == 7 and native_norm_boundary
        active[0] = True
        try:
            yield Operations(counts(layers=1))
        finally:
            active[0] = False

    class Adapter:
        def _legacy_layers(self, cache):
            return cache, "tuple"

        def borrow_past_key_values(self, cache):
            return CacheSnapshot(
                tuple(CacheLayerSnapshot(k, v, 2) for k, v in cache),
                tuple,
                "tuple",
                cache[0][0].shape[2],
                CachedKeyState.POST_ROPE,
            )

        def inject_past_key_values(self, snapshot):
            return tuple((layer.key, layer.value) for layer in snapshot.layers)

        def _output(self, cache, state):
            logits = torch.zeros((1, 11), dtype=torch.float32)
            logits[0, int(cache[0][0].sum()) % 11] = 1.0
            return SimpleNamespace(
                logits=logits, next_token_id=logits.argmax(-1, keepdim=True), state=state
            )

        def prefill(self, prepared, **kwargs):
            ids = prepared.model_inputs["input_ids"].to(torch.float16)
            key = ids[:, None, :, None].repeat(1, 1, 1, 2)
            cache = ((key, key.clone()),)
            n = prepared.logical_sequence_length
            return self._output(cache, DecodeState(cache, None, n, n, n, prepared.modality_map))

        def decode_one_token(self, token, state, **kwargs):
            calls.append(active[0])
            value = token.to(torch.float16)[:, None, :, None].repeat(1, 1, 1, 2)
            cache = tuple(
                (torch.cat((k, value), 2), torch.cat((v, value), 2))
                for k, v in state.past_key_values
            )
            n = state.next_decode_position + 1
            state = DecodeState(cache, None, n, n, n, state.modality_map)
            return self._output(cache, state)

    monkeypatch.setattr(transport, "shape_matched_transport", shape)
    monkeypatch.setattr(protocol, "_forward_identities", lambda adapter: (active[0],))
    prepared = PreparedInputs(
        dict(input_ids=torch.arange(7)[None], attention_mask=torch.ones(1, 7, dtype=torch.long)),
        (ModalitySpan(0, 7, Modality.TEXT),),
        7,
    )
    result = protocol.run_paths(Adapter(), prepared, predictions=128)
    assert calls == [True] + [False] * 254
    assert result["ordinary_continuation_steps"] == 127
    assert result["reference_logits"].shape == (128, 11)
    assert torch.equal(result["reference_logits"], result["candidate_logits"])
    assert result["initial_cache"]["bitwise_equal"]
    assert result["final_cache"]["bitwise_equal"]
    assert result["final_logical_position"] == 134


def test_sbatch_exports_cache_and_determinism_before_python():
    path = protocol.SOURCE / "mosaickv/slurm/fox_coco_boundary_v28.sbatch"
    script = path.read_text()
    assert script.index("source mosaickv/scripts/cache_env.sh") < script.index("exec /scratch")
    assert script.index("CUBLAS_WORKSPACE_CONFIG=:4096:8") < script.index("exec /scratch")
    assert "HF_HUB_OFFLINE=1" in script and "assert_clean_worktree.sh" in script
