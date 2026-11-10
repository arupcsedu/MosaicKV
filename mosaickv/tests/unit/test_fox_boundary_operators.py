"""CPU-only final-token hooks, independent state boundaries and negative gates."""

import copy
import json
from types import SimpleNamespace

import evicastkv_boundary_operators as diagnostic
import pytest
import torch

from mosaickv.adapters.huggingface import PreparedInputs
from mosaickv.cache_state import Modality, ModalitySpan


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Identity()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()
        self.o_proj = torch.nn.Identity()

    def forward(self, hidden_states):
        value = (
            self.q_proj(hidden_states) + self.k_proj(hidden_states) + self.v_proj(hidden_states)
        ) / 3
        return self.o_proj(value), None


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Identity()
        self.up_proj = torch.nn.Identity()
        self.down_proj = torch.nn.Identity()

    def forward(self, value):
        return self.down_proj(self.gate_proj(value) + self.up_proj(value))


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = Attention()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = Mlp()

    def forward(self, hidden_states):
        value = hidden_states + self.self_attn(self.input_layernorm(hidden_states))[0]
        return value + self.mlp(self.post_attention_layernorm(value))


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.norm = torch.nn.Identity()

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.readout = torch.nn.Linear(4, 8, bias=False)
        torch.nn.init.constant_(self.readout.weight, 1)

    def get_output_embeddings(self):
        return self.readout

    def forward(self, ids):
        hidden = ids.float().unsqueeze(-1).expand(-1, -1, 4).contiguous()
        return self.readout(self.decoder(hidden))


class Adapter:
    def __init__(self, wrong_position=False):
        self.model = Model()
        self.prefill_lengths = []
        self.decode_positions = []
        self.wrong_position = wrong_position

    def _language_layers(self):
        return self.model.decoder.layers

    def state(self, length):
        return SimpleNamespace(
            logical_sequence_length=length,
            active_cache_length=length,
            next_decode_position=length + int(self.wrong_position),
        )

    def prefill(self, prepared, **kwargs):
        assert kwargs == dict(capture_queries=False, capture_attentions=False)
        self.prefill_lengths.append(prepared.logical_sequence_length)
        return SimpleNamespace(
            logits=self.model(prepared.model_inputs["input_ids"])[:, -1],
            state=self.state(prepared.logical_sequence_length),
        )

    def decode_one_token(self, token, state, **kwargs):
        assert kwargs == dict(capture_queries=False, capture_attentions=False)
        self.decode_positions.append(state.next_decode_position)
        # Return a different authoritative state, not a mutation of the input.
        return SimpleNamespace(
            logits=self.model(token)[:, -1], state=self.state(state.logical_sequence_length + 1)
        )


def prepared():
    return PreparedInputs(
        dict(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), attention_mask=torch.ones(1, 5)),
        (ModalitySpan(0, 5, Modality.TEXT),),
        5,
    )


def test_two_paths_capture_last_position_only_without_prefix_capture_or_mutation():
    adapter, inputs = Adapter(), prepared()
    saved = inputs.model_inputs["input_ids"].clone()
    capture = diagnostic.FinalPositionCapture(diagnostic.operator_targets(adapter))
    result = diagnostic.run_paths(adapter, inputs, capture)
    report = diagnostic.compare_capture(result)
    assert adapter.prefill_lengths == [5, 4] and adapter.decode_positions == [4]
    assert result["held_token_forward_count"] == 1
    assert report["fullkv_chunked_logit_control_passed"]
    assert report["first_nonbitwise_operator"] is None
    assert set(result["captures"]) == {"A", "B"}
    for phase, tensors in result["captures"].items():
        for entry in tensors.values():
            assert entry["vector"].device.type == "cpu"
            assert entry["vector"].shape in ((1, 1, 4), (1, 1, 8))
            assert entry["source_shape"][1] == (5 if phase == "A" else 1)
    assert torch.equal(saved, inputs.model_inputs["input_ids"])
    assert not capture.handles
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in adapter.model.modules()
    )


def test_position_violation_fails_before_held_token_and_removes_hooks():
    adapter = Adapter(wrong_position=True)
    capture = diagnostic.FinalPositionCapture(diagnostic.operator_targets(adapter))
    with pytest.raises(AssertionError, match="positions/capacity"):
        diagnostic.run_paths(adapter, prepared(), capture)
    assert adapter.decode_positions == [] and not capture.handles


def test_capture_byte_limit_rejects_instead_of_retaining_full_prompt():
    adapter = Adapter()
    capture = diagnostic.FinalPositionCapture(diagnostic.operator_targets(adapter), maximum_bytes=1)
    with pytest.raises(ValueError, match="byte ceiling"):
        diagnostic.run_paths(adapter, prepared(), capture)
    assert capture.total_bytes == 0 and not capture.handles


def test_first_divergent_operator_and_negative_global_control_are_retained():
    adapter = Adapter()
    result = diagnostic.run_paths(
        adapter, prepared(), diagnostic.FinalPositionCapture(diagnostic.operator_targets(adapter))
    )
    changed = copy.deepcopy(result)
    label = changed["orders"]["B"][2]
    changed["captures"]["B"][label]["vector"].reshape(-1)[1] += 0.125
    changed["logits_B"][0, 2] += 0.5
    report = diagnostic.compare_capture(changed)
    assert report["first_nonbitwise_operator"] == label
    assert report["first_operator_exceeding_absolute_1e_minus4"] == label
    assert report["first_logit_comparison"]["maximum_absolute_difference"] == 0.5
    assert not report["fullkv_chunked_logit_control_passed"]
    assert report["fullkv_chunked_logit_absolute_tolerance"] == 1e-4


def test_shape_dtype_nonfinite_and_missing_operator_checks_are_strict():
    with pytest.raises(ValueError, match="shape/dtype"):
        diagnostic.compare_vectors(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="shape/dtype"):
        diagnostic.compare_vectors(torch.zeros(1, 2), torch.zeros(1, 2).half())
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.compare_vectors(torch.tensor([float("nan")]), torch.zeros(1))
    with pytest.raises(ValueError, match="order or coverage"):
        diagnostic.compare_capture(dict(orders=dict(A=["a"], B=[])))


def test_signed_zero_is_bitwise_difference_not_numerical_threshold_repair():
    result = diagnostic.compare_vectors(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not result["bitwise_equal"] and result["maximum_absolute_difference"] == 0
    assert result["first_different_storage_byte"] is not None


def test_forbidden_ledger_values_are_redacted_before_object_decoding(monkeypatch):
    raw = json.dumps(
        dict(
            dataset_id="dataset",
            dataset_revision="revision",
            split="test",
            task="coco_caption",
            samples=[
                dict(
                    sample_id="fixture",
                    prompt="input only",
                    media=[],
                    references=["FORBIDDEN_REFERENCE"],
                    answers=dict(secret="FORBIDDEN_ANSWER"),
                    teacher_forcing_target="FORBIDDEN_TARGET",
                )
            ],
        )
    )
    actual_loads = diagnostic.json.loads

    def checked_loads(value, **kwargs):
        assert "FORBIDDEN_" not in value
        return actual_loads(value, **kwargs)

    monkeypatch.setattr(diagnostic.json, "loads", checked_loads)
    result = diagnostic.input_only_projection(raw)
    assert result["samples"] == [dict(sample_id="fixture", prompt="input only", media=[])]


@pytest.mark.parametrize("text", ['"unterminated', '["bad}', '{"x":[1,2]'])
def test_forbidden_value_lexical_skip_rejects_truncation(text):
    with pytest.raises(ValueError):
        diagnostic._skip_json_value(text, 0)


def test_cpu_tensor_capture_is_write_once_and_safe_loadable(tmp_path):
    target = tmp_path / "captures" / "00.pt"
    diagnostic._save_tensor_once(target, {"value": torch.ones(2)})
    assert torch.equal(torch.load(target, weights_only=True)["value"], torch.ones(2))
    with pytest.raises(FileExistsError):
        diagnostic._save_tensor_once(target, {"value": torch.zeros(2)})
    assert not list(target.parent.glob(".capture-*"))


@pytest.fixture
def parent_plan(tmp_path, monkeypatch):
    source, prior, parent = (tmp_path / name for name in ("source", "prior", "parent"))
    for base in (source, prior):
        for relative in (
            "mosaickv/scripts/evicastkv_campaign.py",
            "mosaickv/src/mosaickv/adapters/huggingface/base.py",
            "mosaickv/src/mosaickv/adapters/huggingface/llava_next.py",
            "mosaickv/configs/experiments/evicastkv_fox/coco/full_kv.json",
        ):
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture unchanged control\n")
        path = base / "mosaickv/src/mosaickv/evaluation/fox_boundary.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def hold_back_last_text_token(value):\n    return value\n")
    parent.mkdir()
    ledger = parent / "profiler.json"
    ledger.write_text(
        json.dumps(
            dict(
                dataset_id="fixture",
                dataset_revision="f" * 40,
                split="val",
                task="coco_caption",
                samples=[
                    dict(
                        sample_id=f"TEST_ONLY-{i}",
                        prompt="input",
                        media=[],
                        references=["MUST_NOT_BE_DECODED"],
                    )
                    for i in range(8)
                ],
            )
        )
    )
    lock = dict(
        source_root=str(prior),
        source_sha="1" * 40,
        arithmetic=dict(allow_fp16_reduced_precision_reduction=True),
        cohorts=dict(
            coco=dict(
                profiler=dict(
                    path=str(ledger),
                    sha256=diagnostic.sha(ledger),
                    sample_ids=[f"TEST_ONLY-{i}" for i in range(8)],
                )
            )
        ),
    )
    lock_path = parent / "protocol_lock.json"
    lock_path.write_text(json.dumps(lock))
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                dict(
                    status="passed",
                    blockers=[],
                    source_sha="1" * 40,
                    protocol_sha256=diagnostic.sha(lock_path),
                )
            )
        )
    monkeypatch.setattr(diagnostic, "clean_source", lambda *_args: "1" * 40)
    monkeypatch.setattr(diagnostic, "runtime_files", lambda _source: {"fixture": "same"})
    return source, prior, parent


def test_registered_plan_uses_first_two_consumed_inputs_and_all_four_gates(parent_plan):
    source, _prior, parent = parent_plan
    _lock, inputs, bindings = diagnostic.registered_plan(parent, source=source)
    assert [sample["sample_id"] for sample in inputs["samples"]] == ["TEST_ONLY-0", "TEST_ONLY-1"]
    assert all(set(sample) == {"sample_id", "prompt", "media"} for sample in inputs["samples"])
    assert sum("GATE_" in key for key in bindings) == 4


@pytest.mark.parametrize("stage", ["correctness", "scoring", "profiling", "measurement"])
def test_failed_parent_gate_blocks_diagnostic_without_gpu(parent_plan, stage):
    source, _prior, parent = parent_plan
    path = parent / "analysis" / stage / f"GATE_{stage}.json"
    value = json.loads(path.read_text())
    value["status"] = "blocked"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="parent gate"):
        diagnostic.registered_plan(parent, source=source)


def test_changed_shared_source_control_is_not_accepted(parent_plan):
    source, _prior, parent = parent_plan
    (source / "mosaickv/src/mosaickv/adapters/huggingface/base.py").write_text("changed")
    with pytest.raises(ValueError, match="controls changed"):
        diagnostic.registered_plan(parent, source=source)
