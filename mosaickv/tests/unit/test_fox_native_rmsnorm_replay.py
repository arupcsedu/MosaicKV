"""Synthetic CPU-only native-norm replay, cached-weight scope and provenance."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("evicastkv_boundary_rmsnorm_replay")


@pytest.fixture
def config(driver):
    return dict(
        model=driver.MODEL,
        method="full_kv",
        cache=dict(retention_ratio=1.0),
        execution=dict(backend="huggingface", attention_implementation="eager"),
    )


@pytest.fixture
def checkpoint(driver, config, tmp_path):
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / ("models--" + driver.MODEL["id"].replace("/", "--"))
        / "snapshots"
        / driver.MODEL["revision"]
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            dict(model_type="llava_next", text_config=dict(model_type="mistral", rms_norm_eps=1e-5))
        )
    )
    shard = snapshot / "fixture.safetensors"
    weight = torch.linspace(0.5, 1.5, 4096).half()
    save_file({driver.WEIGHT_KEY: weight, "unrequested.fixture.weight": torch.ones(3)}, shard)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                weight_map={driver.WEIGHT_KEY: shard.name, "unrequested.fixture.weight": shard.name}
            )
        )
    )
    return dict(cache=cache, snapshot=snapshot, shard=shard, weight=weight, config=config)


def saved_capture():
    entry = dict(vector=torch.zeros(1, 1, 4096, dtype=torch.float16), source_shape=[1, 2185, 4096])
    return dict(
        captures={
            side: {
                f"layer.29.input_layernorm:{kind}": copy.deepcopy(entry)
                for kind in ("input", "output")
            }
            for side in ("A", "B")
        }
    )


def test_operator_scope_cannot_promote_model_or_quality_gate(driver):
    assert driver.POLICY["predictions_generated"] == 0
    assert driver.POLICY["original_forward_only"]
    assert driver.POLICY["weights_or_epsilon_changed"] is False
    assert driver.POLICY["original_logit_gate_absolute_tolerance"] == 1e-4
    assert driver.POLICY["full_model_transport_automatic_release"] is False


def test_only_exact_selected_checkpoint_norm_is_materialized(driver, checkpoint):
    inventory, weight = driver.checkpoint_inventory(checkpoint["config"], cache=checkpoint["cache"])
    assert torch.equal(weight.view(torch.int16), checkpoint["weight"].view(torch.int16))
    assert inventory["epsilon"] == 1e-5
    assert inventory["tensor_count_read"] == 1
    assert inventory["only_selected_tensor_materialized"]
    assert inventory["full_model_loaded"] is False
    assert inventory["weight_tensor_sha256"] == driver.tensor_sha(weight)
    assert inventory["weight_key"] == "language_model.model.layers.29.input_layernorm.weight"
    assert (
        inventory["checkpoint_conversion_mapping"]["^language_model.model"]
        == "model.language_model"
    )


@pytest.mark.parametrize(
    "field,value", [("id", "Qwen/other"), ("revision", "main"), ("precision", "bf16")]
)
def test_different_model_or_precision_is_rejected(driver, config, field, value):
    value_config = copy.deepcopy(config)
    value_config["model"][field] = value
    with pytest.raises(ValueError, match="pinned FP16"):
        driver.validate_model(value_config)


@pytest.mark.parametrize("mutation", ["sdpa", "compressed", "retention"])
def test_parent_backend_and_fullkv_scope_are_fixed(driver, config, mutation):
    value = copy.deepcopy(config)
    if mutation == "sdpa":
        value["execution"]["attention_implementation"] = "sdpa"
    elif mutation == "compressed":
        value["method"] = "mosaickv_exact"
    else:
        value["cache"]["retention_ratio"] = 0.5
    with pytest.raises(ValueError):
        driver.validate_model(value)


@pytest.mark.parametrize(
    "bad", ["wrong_architecture", "wrong_text", "wrong_width", "negative_epsilon"]
)
def test_cached_native_geometry_is_validated(driver, checkpoint, bad):
    path = checkpoint["snapshot"] / "config.json"
    value = json.loads(path.read_text())
    if bad == "wrong_architecture":
        value["model_type"] = "llava"
    elif bad == "wrong_text":
        value["text_config"]["model_type"] = "llama"
    elif bad == "wrong_width":
        value["text_config"]["hidden_size"] = 2048
    else:
        value["text_config"]["rms_norm_eps"] = -1e-5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        driver.checkpoint_inventory(checkpoint["config"], cache=checkpoint["cache"])


@pytest.mark.parametrize(
    "filename", ["../outside.safetensors", "/absolute.safetensors", "not-weights.json", None]
)
def test_checkpoint_index_cannot_escape_or_omit_exact_weight(driver, checkpoint, filename):
    path = checkpoint["snapshot"] / "model.safetensors.index.json"
    path.write_text(json.dumps(dict(weight_map={driver.WEIGHT_KEY: filename})))
    with pytest.raises(ValueError, match="unsafe shard"):
        driver.checkpoint_inventory(checkpoint["config"], cache=checkpoint["cache"])


@pytest.mark.parametrize("mutation", ["float32", "wrong_shape", "nonfinite", "missing"])
def test_incompatible_cached_norm_weight_is_rejected(driver, checkpoint, mutation):
    weights = {driver.WEIGHT_KEY: checkpoint["weight"]}
    if mutation == "float32":
        weights[driver.WEIGHT_KEY] = checkpoint["weight"].float()
    elif mutation == "wrong_shape":
        weights[driver.WEIGHT_KEY] = torch.ones(2048).half()
    elif mutation == "nonfinite":
        weights[driver.WEIGHT_KEY][0] = float("nan")
    else:
        weights = {"other": torch.ones(4096).half()}
    save_file(weights, checkpoint["shard"])
    with pytest.raises(ValueError):
        driver.checkpoint_inventory(checkpoint["config"], cache=checkpoint["cache"])


def test_selected_vectors_are_bitwise_equal_but_original_gpu_layout_is_not_invented(driver):
    capture = saved_capture()
    for kind in ("input", "output"):
        capture["captures"]["B"][f"layer.29.input_layernorm:{kind}"]["source_shape"] = [1, 1, 4096]
    value = driver.selected_case(capture)
    assert value["original_GPU_layout_known"] is False
    assert value["A_input_source_shape"] == [1, 2185, 4096]
    assert value["B_input_source_shape"] == [1, 1, 4096]
    assert (
        value["A_input"].data_ptr()
        != capture["captures"]["A"]["layer.29.input_layernorm:input"]["vector"].data_ptr()
    )


@pytest.mark.parametrize("bad", ["different_input", "wrong_dtype", "wrong_width", "wrong_B_rows"])
def test_unusable_saved_operator_boundary_is_rejected(driver, bad):
    capture = saved_capture()
    for kind in ("input", "output"):
        capture["captures"]["B"][f"layer.29.input_layernorm:{kind}"]["source_shape"] = [1, 1, 4096]
    entry = capture["captures"]["B"]["layer.29.input_layernorm:input"]
    if bad == "different_input":
        entry["vector"][0, 0, 0] = 1
    elif bad == "wrong_dtype":
        entry["vector"] = entry["vector"].float()
    elif bad == "wrong_width":
        entry["vector"] = entry["vector"][..., :2048]
    else:
        entry["source_shape"] = [1, 2, 4096]
    with pytest.raises(ValueError):
        driver.selected_case(capture)


def test_tiny_real_native_forward_cpu_replay_is_unchanged_and_private(driver):
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    torch.manual_seed(2)
    norm = MistralRMSNorm(8, eps=1e-5).eval().half()
    vector = torch.randn(1, 1, 8).half()
    weight, original = norm.weight.clone(), vector.clone()
    result = driver.replay(norm, vector, 7)
    expected = norm(vector).detach()
    assert set(result["branches"]) == {"single_compact", "single_same_buffer_view", "full_rows"}
    for branch in result["branches"].values():
        assert torch.equal(branch["output"].view(torch.int16), expected.view(torch.int16))
        assert branch["repeat_bitwise"]
        assert branch["variance_last"].dtype == torch.float32
    assert result["branches"]["single_same_buffer_view"]["input_layout"]["storage_offset"] == 6 * 8
    assert result["branches"]["single_compact"]["input_layout"]["storage_offset"] == 0
    assert result["epsilon"] == 1e-5
    assert torch.equal(norm.weight, weight) and torch.equal(vector, original)
    assert result["memory_or_timing_claim"] is False
    assert result["complete_operator_workspace_peak_measured"] is False


def test_native_replay_rejects_different_precision_or_invalid_rows(driver):
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    norm = MistralRMSNorm(8).half().eval()
    for vector, rows in (
        (torch.ones(1, 1, 8), 7),
        (torch.ones(1, 1, 8).half(), 1),
        (torch.ones(1, 1, 8).half(), 5000),
    ):
        with pytest.raises(ValueError):
            driver.replay(norm, vector, rows)


def test_comparison_is_bitwise_strict_and_finite(driver):
    left = torch.zeros(1, 1, 4).half()
    right = left.clone()
    right[..., 1] = 0.00390625
    result = driver.compare(left, right)
    assert not result["bitwise_equal"]
    assert result["unequal_elements"] == 1
    assert result["maximum_absolute_difference"] == 0.00390625
    with pytest.raises(ValueError):
        driver.compare(left, torch.ones(1, 1, 4))
    with pytest.raises(ValueError):
        driver.compare(left, right * float("nan"))


def test_no_full_model_loader_is_used(driver):
    import inspect

    text = inspect.getsource(driver.checkpoint_inventory) + inspect.getsource(driver.worker)
    assert ".from_pretrained(" not in text
    assert ".get_tensor(WEIGHT_KEY)" in text
    assert "MistralRMSNorm(4096" in "".join(text.split())


@pytest.fixture
def analysis_fixture(driver, tmp_path, monkeypatch):
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    root = tmp_path / "operator-analysis"
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps(dict(synthetic_fixture=True)))
    norm = MistralRMSNorm(4096, eps=1e-5).eval().half()
    cases, records = [], []
    for index in range(2):
        capture = saved_capture()
        for kind in ("input", "output"):
            capture["captures"]["A"][f"{driver.OPERATOR}:{kind}"]["source_shape"] = [1, 3, 4096]
            capture["captures"]["B"][f"{driver.OPERATOR}:{kind}"]["source_shape"] = [1, 1, 4096]
        case = driver.selected_case(capture)
        case.update(index=index, sample_id=f"synthetic-{index}")
        cases.append(case)
        value = driver.replay(norm, case["A_input"], 3)
        path = root / f"{index}.pt"
        torch.save(value, path)
        records.append(
            dict(
                index=index,
                sample_id=case["sample_id"],
                status="completed",
                capture_path=str(path),
                capture_sha256=driver.sha(path),
            )
        )
    torch.save(dict(cases=cases), root / "selected_inputs.pt")
    source_sha = "a" * 40
    protocol = dict(
        source_sha=source_sha,
        checkpoint=dict(weight_tensor_sha256=driver.tensor_sha(norm.weight), epsilon=1e-5),
    )
    monkeypatch.setattr(driver, "validate", lambda root: (protocol, norm.weight))
    (root / "environment.json").write_text(
        json.dumps(dict(source_sha=source_sha, model=driver.MODEL, full_VLM_loaded=False))
    )
    result = dict(
        source_sha=source_sha,
        protocol_sha256=driver.sha(root / "protocol.json"),
        environment_sha256=driver.sha(root / "environment.json"),
        records=records,
    )
    (root / "result.json").write_text(json.dumps(result))
    return root, result


def test_saved_operator_result_analyzer_remains_diagnostic_only(driver, analysis_fixture):
    root, _result = analysis_fixture
    gate = driver.analyze(root)
    assert gate["complete"]
    assert gate["both_saved_shape_pairs_reproduced"]
    assert gate["full_model_parity_established"] is False
    assert gate["full_model_auto_release"] is False
    assert gate["superiority_or_performance_claim"] is False
    assert gate["negative_native_and_shape_gates_preserved"]
    with pytest.raises(FileExistsError):
        driver.analyze(root)


def test_failed_operator_case_is_not_silently_dropped(driver, analysis_fixture):
    root, result = analysis_fixture
    result["records"][1] = dict(
        index=1, sample_id="synthetic-1", status="failed", error="synthetic operation failure"
    )
    (root / "result.json").write_text(json.dumps(result))
    gate = driver.analyze(root)
    assert not gate["complete"]
    assert gate["status"] == "blocked_incomplete_operator_replay"
    assert len(gate["reports"]) == 1 and len(gate["failures"]) == 1


def test_operator_result_cannot_rebind_an_unrelated_environment(driver, analysis_fixture):
    root, result = analysis_fixture
    (root / "environment.json").write_text(
        json.dumps(dict(source_sha="b" * 40, model=driver.MODEL, full_VLM_loaded=False))
    )
    result["environment_sha256"] = driver.sha(root / "environment.json")
    (root / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="environment/model"):
        driver.analyze(root)
