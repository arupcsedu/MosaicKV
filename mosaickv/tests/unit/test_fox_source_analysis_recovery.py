"""CPU-only reduction-verification policy; no new predictions or GPU access."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("evicastkv_review_source_prefix")


@pytest.fixture
def diagnostics():
    comparison = dict(
        per_prefix_KL_FP32=[0.0, 0.0001],
        per_prefix_KL_FP64=[0.0, 0.0001],
        per_prefix_maximum_logit_difference=[0.0, 0.3],
        per_prefix_greedy_agreement=[True, True],
        mean_KL_FP64=0.00005,
        maximum_KL_FP32=0.0001,
        maximum_logit_difference=0.3,
        token_agreement=1.0,
        first_divergent_prediction=None,
        registered_guard_pass=True,
    )
    return dict(
        registered_guard_32=copy.deepcopy(comparison),
        cache_dependent_guard_32=copy.deepcopy(comparison),
        prefix_64=copy.deepcopy(comparison),
        prefix_128=copy.deepcopy(comparison),
        first_prediction_is_common_prefill=True,
        registered_guard_unchanged=True,
        same_conditioning_prefix=True,
        actual_cache_dependent_prediction_count=127,
        not_free_greedy_sequence_agreement=True,
    )


def test_exact_recomputation_passes_without_numeric_drift(driver, diagnostics):
    record = copy.deepcopy(diagnostics)
    verified = driver.verify_recomputed_diagnostics(diagnostics, record)
    assert verified["maximum_KL_FP32_recomputation_drift"] == 0.0
    assert verified["maximum_KL_FP64_recomputation_drift"] == 0.0
    assert verified["exact_guard_booleans_checked"] == 4
    assert verified["verified_KL_FP32_values"] == 12
    assert verified["verified_KL_FP64_values"] == 12
    assert record == diagnostics


def test_cpu_kl_roundoff_is_accepted_but_recorded_values_never_replaced(driver, diagnostics):
    saved = copy.deepcopy(diagnostics)
    saved["prefix_128"]["per_prefix_KL_FP32"][1] += 3.725290298461914e-9
    saved["prefix_128"]["per_prefix_KL_FP64"][1] += 5.6e-17
    before = copy.deepcopy(saved)
    result = driver.verify_recomputed_diagnostics(diagnostics, saved)
    assert 0 < result["maximum_KL_FP32_recomputation_drift"] < 4e-9
    assert 0 < result["maximum_KL_FP64_recomputation_drift"] < 6e-17
    assert result["changed_KL_FP32_values"] == result["changed_KL_FP64_values"] == 1
    assert saved == before
    assert result["exact_guard_booleans_checked"] == 4


@pytest.mark.parametrize("key,delta", [("maximum_KL_FP32", 1e-8), ("mean_KL_FP64", 1e-16)])
def test_only_registered_kl_summary_reductions_can_have_roundoff(driver, diagnostics, key, delta):
    saved = copy.deepcopy(diagnostics)
    saved["registered_guard_32"][key] += delta
    driver.verify_recomputed_diagnostics(diagnostics, saved)


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_logit_difference", 0.300000000000001),
        ("token_agreement", 0.5),
        ("first_divergent_prediction", 0),
        ("registered_guard_pass", False),
        ("per_prefix_greedy_agreement", [True, False]),
        ("per_prefix_maximum_logit_difference", [0.0, 0.3000000001]),
    ],
)
def test_logit_token_identity_and_guard_changes_are_never_tolerated(
    driver, diagnostics, key, value
):
    saved = copy.deepcopy(diagnostics)
    saved["registered_guard_32"][key] = value
    with pytest.raises(ValueError, match="non-KL"):
        driver.verify_recomputed_diagnostics(diagnostics, saved)


def test_near_threshold_kl_roundoff_cannot_turn_a_failed_guard_into_pass(driver, diagnostics):
    saved = copy.deepcopy(diagnostics)
    diagnostics["registered_guard_32"]["maximum_KL_FP32"] = 0.01999999
    saved["registered_guard_32"]["maximum_KL_FP32"] = 0.02000001
    saved["registered_guard_32"]["registered_guard_pass"] = False
    with pytest.raises(ValueError, match="registered_guard_pass"):
        driver.verify_recomputed_diagnostics(diagnostics, saved)


@pytest.mark.parametrize("key", ["per_prefix_KL_FP32", "per_prefix_KL_FP64"])
def test_material_kl_drift_is_rejected(driver, diagnostics, key):
    saved = copy.deepcopy(diagnostics)
    saved["registered_guard_32"][key][1] += 0.001
    with pytest.raises(ValueError, match="precision bound"):
        driver.verify_recomputed_diagnostics(diagnostics, saved)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "0.0"])
@pytest.mark.parametrize("key", ["per_prefix_KL_FP32", "per_prefix_KL_FP64"])
def test_kl_values_must_be_finite_numeric(driver, diagnostics, key, bad):
    saved = copy.deepcopy(diagnostics)
    saved["registered_guard_32"][key][1] = bad
    with pytest.raises(ValueError, match="nonfinite or invalid KL"):
        driver.verify_recomputed_diagnostics(diagnostics, saved)


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "short_list", "wrong_type", "bool_as_int"]
)
def test_structure_lengths_and_non_kl_types_are_exact(driver, diagnostics, mutation):
    saved = copy.deepcopy(diagnostics)
    if mutation == "extra":
        saved["invented"] = 1
    elif mutation == "missing":
        del saved["prefix_64"]
    elif mutation == "short_list":
        saved["prefix_128"]["per_prefix_KL_FP32"].pop()
    elif mutation == "wrong_type":
        saved["prefix_128"]["per_prefix_KL_FP32"] = (0.0, 0.0001)
    else:
        saved["registered_guard_unchanged"] = 1
    with pytest.raises(ValueError):
        driver.verify_recomputed_diagnostics(diagnostics, saved)


def test_policy_epsilon_is_precision_derived_and_model_thresholds_unchanged(driver):
    assert driver.ANALYSIS_POLICY["fp32_epsilon"] == torch.finfo(torch.float32).eps
    assert driver.ANALYSIS_POLICY["fp64_epsilon"] == torch.finfo(torch.float64).eps
    assert driver.ANALYSIS_POLICY["fp32_epsilon_multiplier"] == 8
    assert driver.ANALYSIS_POLICY["fp64_epsilon_multiplier"] == 64
    assert driver.POLICY["guard"] == dict(
        tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0
    )


def test_cpu_analysis_requires_prelaunch_avx2(driver, monkeypatch):
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    with pytest.raises(ValueError, match="before Python"):
        driver.analysis_cpu_controls()


def test_cpu_analysis_rejects_actual_non_avx2_dispatch(driver, monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "AVX512")
    with pytest.raises(ValueError, match="dispatcher"):
        driver.analysis_cpu_controls()


def test_cpu_analysis_sets_twelve_threads_locally(driver, monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "AVX2")
    requested = []
    monkeypatch.setattr(torch, "set_num_threads", requested.append)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 12)
    result = driver.analysis_cpu_controls()
    assert requested == [12]
    assert result["intraop_threads"] == 12
    assert result["torch_cpu_capability"] == "AVX2"


def test_derived_analyzer_accepts_explicit_clean_producer_not_implicit_sha_rewrite(
    driver, tmp_path, monkeypatch
):
    original = tmp_path / "original-source"
    original.mkdir()
    bound = original / "producer.txt"
    bound.write_text("synthetic producer fixture")
    root = tmp_path / "derived"
    root.mkdir()
    revision = "a" * 40
    value = dict(
        source_sha=revision,
        policy=driver.POLICY,
        arms=driver.ARMS,
        bound_files={str(bound): driver.sha(bound)},
    )
    (root / "plan.json").write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(
        driver,
        "clean_source",
        lambda source, expected=None: calls.append((source, expected)) or revision,
    )
    assert driver.validate(root, producer_source=original)["source_sha"] == revision
    assert calls == [(original.resolve(), revision)]
    bound.write_text("changed producer evidence")
    with pytest.raises(ValueError, match="binding changed"):
        driver.validate(root, producer_source=original)


def test_producer_source_option_is_cli_analyzer_only(driver, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "script",
            "worker",
            "--root",
            "/scratch/djy8hg/fixture",
            "--producer-source",
            "/scratch/djy8hg/producer",
        ],
    )
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
