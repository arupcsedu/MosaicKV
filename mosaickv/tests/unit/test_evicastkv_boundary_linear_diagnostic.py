"""Synthetic CPU regressions for bounded, operator-only numerical localization."""

import copy
import json
from types import SimpleNamespace

import evicastkv_boundary_linear_diagnostic as diagnostic
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.fixture
def flags():
    with diagnostic.isolated_flags(torch, diagnostic.ORIGINAL_FLAGS):
        yield


@pytest.fixture
def operands():
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(1, 7, 8, generator=generator).half()
    layer = torch.nn.Linear(8, 5, bias=True).half()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(5, 8, generator=generator).half())
        layer.bias.copy_(torch.randn(5, generator=generator).half())
    return inputs, layer


def test_settings_restore_even_when_kernel_raises():
    initial = diagnostic.arithmetic_flags(torch)
    alternate = {**diagnostic.ORIGINAL_FLAGS, "allow_fp16_reduced_precision_reduction": False}
    with (
        pytest.raises(RuntimeError, match="synthetic kernel failure"),
        diagnostic.isolated_flags(torch, alternate),
    ):
        assert diagnostic.arithmetic_flags(torch) == alternate
        raise RuntimeError("synthetic kernel failure")
    assert diagnostic.arithmetic_flags(torch) == initial


def test_tf32_and_incomplete_settings_are_rejected():
    with (
        pytest.raises(ValueError, match="no-TF32"),
        diagnostic.isolated_flags(torch, {**diagnostic.ORIGINAL_FLAGS, "allow_tf32": True}),
    ):
        pass
    with (
        pytest.raises(ValueError, match="complete"),
        diagnostic.isolated_flags(torch, {"allow_tf32": False}),
    ):
        pass


def test_capture_saves_actual_full_inputs_without_modification(operands):
    inputs, layer = operands
    original = inputs.clone()
    capture = diagnostic.LinearInputCapture(("layer.0.attention.q_proj", "model.layers.0.q", layer))
    with capture:
        with capture.phase("A"):
            expected = layer(inputs)
        with capture.phase("B"):
            actual = layer(inputs[:, -1:, :])
    assert capture.complete_inputs["A"].shape == inputs.shape
    assert capture.complete_inputs["B"].shape == (1, 1, 8)
    assert torch.equal(capture.complete_inputs["A"], inputs)
    assert torch.equal(inputs, original)
    assert torch.equal(
        capture.records["A"]["layer.0.attention.q_proj:output"]["vector"], expected[:, -1:, :]
    )
    assert torch.equal(capture.records["B"]["layer.0.attention.q_proj:output"]["vector"], actual)
    assert not capture.handles


def test_capture_limit_rejects_before_copying(operands):
    inputs, layer = operands
    capture = diagnostic.LinearInputCapture(("q", "q", layer), maximum_bytes=1)
    with pytest.raises(ValueError, match="capture limit"), capture, capture.phase("A"):
        layer(inputs)
    assert not capture.handles
    assert capture.complete_inputs == {}


def packet(operands):
    inputs, layer = operands
    capture = diagnostic.LinearInputCapture(("layer.0.attention.q_proj", "q", layer))
    with capture:
        with capture.phase("A"):
            layer(inputs)
        with capture.phase("B"):
            layer(inputs[:, -1:, :])
    logits_a = torch.zeros(1, 5)
    logits_b = logits_a.clone()
    logits_b[0, 0] = 0.01  # Illustrative fixture, never an experimental result.
    return {
        "model_paths": {
            "captures": capture.records,
            "orders": capture.orders,
            "logits_A": logits_a,
            "logits_B": logits_b,
        },
        "complete_inputs": capture.complete_inputs,
        "complete_input_layouts": capture.complete_input_layouts,
        "operator_replays": diagnostic.operator_replays(
            inputs, layer.weight, layer.bias, torch=torch
        ),
    }, {"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()}


def test_operator_shapes_and_reference_use_identical_last_operands(operands, flags):
    inputs, layer = operands
    preserved = inputs.clone(), layer.weight.clone(), layer.bias.clone()
    result = diagnostic.operator_replays(inputs, layer.weight, layer.bias, torch=torch)
    assert [v["name"] for v in result["variants"]] == [name for name, _ in diagnostic.VARIANTS]
    for variant in result["variants"]:
        if variant["status"] == "completed":
            assert set(variant["outputs"]) == {
                "full_sequence",
                "last_row",
                "shape_matched_zero_prefix",
            }
            assert all(tensor.shape == (1, 1, 5) for tensor in variant["outputs"].values())
    expected = torch.nn.functional.linear(
        inputs[:, -1:, :].double(), layer.weight.double(), layer.bias.double()
    )
    assert torch.equal(expected, result["CPU_FP64_last_row"])
    assert result["arithmetic_flags_restored"]
    assert all(
        torch.equal(before, after)
        for before, after in zip(preserved, (inputs, layer.weight, layer.bias), strict=True)
    )


@pytest.mark.parametrize("mutation", ["rank", "batch", "length", "dtype", "width"])
def test_operator_rejects_unregistered_operand_geometry(operands, flags, mutation):
    inputs, layer = operands
    if mutation == "rank":
        inputs = inputs[0]
    elif mutation == "batch":
        inputs = inputs.expand(2, -1, -1)
    elif mutation == "length":
        inputs = inputs[:, :1]
    elif mutation == "dtype":
        inputs = inputs.float()
    else:
        inputs = inputs[..., :7]
    with pytest.raises(ValueError):
        diagnostic.operator_replays(inputs, layer.weight, layer.bias, torch=torch)


def test_perfect_operator_replay_does_not_turn_negative_model_control_positive(operands, flags):
    value, weight = packet(operands)
    report = diagnostic.compare_packet(value, weight)
    assert report["normalized_last_inputs"]["bitwise_equal"]
    assert all(report["native_operator_replay_checks"].values())
    assert not report["original_fullkv_chunked_logit_control_passed"]
    assert report["original_absolute_logit_tolerance"] == 1e-4
    assert report["original_control_unchanged"]
    assert report["operator_checks_do_not_pass_model_numerical_control"]


def test_cpu_reanalysis_is_deterministic(operands, flags):
    value, weight = packet(operands)
    assert diagnostic.compare_packet(value, weight) == diagnostic.compare_packet(value, weight)


def test_fp64_reference_preserves_caller_thread_budget(operands):
    inputs, layer = operands
    prior = torch.get_num_threads()
    diagnostic.cpu_fp64_reference(inputs[:, -1:, :], layer.weight, layer.bias, torch=torch)
    assert torch.get_num_threads() == prior


def test_worker_and_analysis_cpu_thread_counts_do_not_change_report(operands, flags):
    value, weight = packet(operands)
    report = diagnostic.compare_packet(value, weight)
    torch.set_num_threads(2)
    try:
        assert diagnostic.compare_packet(value, weight) == report
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(1)


@pytest.mark.parametrize("corruption", ["reference", "settings", "shapes", "extra_variant"])
def test_cpu_reanalysis_rejects_corruption(operands, flags, corruption):
    value, weight = packet(operands)
    if corruption == "reference":
        value["operator_replays"]["CPU_FP64_last_row"] += 0.5
    elif corruption == "settings":
        value["operator_replays"]["variants"][0]["flags"]["allow_tf32"] = True
    elif corruption == "shapes":
        del value["operator_replays"]["variants"][0]["outputs"]["last_row"]
    else:
        value["operator_replays"]["variants"].append(
            copy.deepcopy(value["operator_replays"]["variants"][0])
        )
    with pytest.raises(ValueError):
        diagnostic.compare_packet(value, weight)


def test_optional_unsupported_kernel_is_explicitly_recorded(operands, flags, monkeypatch):
    import torch.nn.functional

    inputs, layer = operands
    original = torch.nn.functional.linear

    def controlled(value, weight, bias):
        if torch.backends.cuda.matmul.allow_fp16_accumulation:
            raise RuntimeError("synthetic optional unsupported kernel")
        return original(value, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", controlled)
    result = diagnostic.operator_replays(inputs, layer.weight, layer.bias, torch=torch)
    optional = result["variants"][-1]
    assert optional["status"] == "unsupported"
    assert "unsupported" in optional["error"]
    assert result["arithmetic_flags_restored"]


def prior_fixture(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    prior.mkdir()
    inputs = {"samples": [{"sample_id": "TEST_ONLY_0"}, {"sample_id": "TEST_ONLY_1"}]}
    (prior / "inputs.json").write_text(json.dumps(inputs))
    (prior / "protocol.json").write_text("{}")
    report = {
        "first_nonbitwise_operator": "layer.0.attention.q_proj:output",
        "fullkv_chunked_logit_control_passed": False,
        "operators": [{"operator": "layer.0.attention.q_proj:input", "bitwise_equal": True}],
    }
    (prior / "operator_comparisons.json").write_text(json.dumps([report, report]))
    gate = {
        "status": "passed_diagnostic_capture",
        "attempted": 2,
        "completed": 2,
        "original_absolute_logit_tolerance": 1e-4,
        "thresholds_changed": False,
        "fullkv_chunked_logit_control_all_passed": False,
        "operator_comparisons_sha256": diagnostic.sha(prior / "operator_comparisons.json"),
    }
    (prior / "GATE_boundary_operators.json").write_text(json.dumps(gate))
    monkeypatch.setattr(diagnostic.boundary, "_load_protocol", lambda _: {})
    return prior, inputs


def test_prior_diagnosis_binds_only_the_same_consumed_inputs(tmp_path, monkeypatch):
    prior, inputs = prior_fixture(tmp_path, monkeypatch)
    bindings, _ = diagnostic._prior_bindings(prior, inputs)
    assert len(bindings) == 4
    with pytest.raises(ValueError, match="same two consumed"):
        diagnostic._prior_bindings(prior, {"samples": list(reversed(inputs["samples"]))})


@pytest.mark.parametrize(
    "field,value",
    [("thresholds_changed", True), ("completed", 1), ("original_absolute_logit_tolerance", 0.1)],
)
def test_prior_control_cannot_be_relaxed_or_incomplete(tmp_path, monkeypatch, field, value):
    prior, inputs = prior_fixture(tmp_path, monkeypatch)
    path = prior / "GATE_boundary_operators.json"
    gate = json.loads(path.read_text())
    gate[field] = value
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="original negative"):
        diagnostic._prior_bindings(prior, inputs)


def test_argument_contract_and_fixed_scope_are_inspectable():
    assert diagnostic.CAPTURE_LIMIT == 128 * 2**20
    assert diagnostic.STAGE == "boundary_linear_diagnostic"
    assert all(flags["allow_tf32"] is False for _, flags in diagnostic.VARIANTS)
    assert len(diagnostic.VARIANTS) == 3
    fake = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(**diagnostic.ORIGINAL_FLAGS))
        )
    )
    assert diagnostic.arithmetic_flags(fake) == diagnostic.ORIGINAL_FLAGS
