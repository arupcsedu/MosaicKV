"""No model downloads: forcing, storage, bounded families and no-label controls."""

import dataclasses
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("evicastkv_review_source_prefix")


def test_policy_keeps_unchanged_guard_and_explicit_diagnostic_scope(driver):
    assert driver.POLICY["guard"] == dict(
        tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0
    )
    assert driver.POLICY["selector_can_read_forced_prefix"] is False
    assert driver.POLICY["no_fresh_cohort"] and driver.POLICY["no_calibration"]
    assert len(driver.ARMS) == 7
    assert not any("replay_on" in name for name in driver.ARMS)


def test_prefix_comes_only_from_complete_finite_fullkv_logits(driver):
    logits = torch.arange(128 * 6).reshape(128, 6).float()
    assert driver.forced_prefix(logits) == (5,) * 128
    for bad in (logits[:32], logits.reshape(1, 128, 6), logits * float("nan")):
        with pytest.raises(ValueError, match="FullKV reference"):
            driver.forced_prefix(bad)


def test_fixed_prefix_comparison_preserves_original_guard_and_later_prefixes(driver):
    reference = torch.randn(128, 20)
    candidate = reference.clone()
    candidate[80, 0] += 3
    result = driver.comparisons(reference, candidate)
    assert result["registered_guard_32"]["registered_guard_pass"]
    assert result["cache_dependent_guard_32"]["registered_guard_pass"]
    assert result["prefix_64"]["registered_guard_pass"]
    assert not result["prefix_128"]["registered_guard_pass"]
    assert result["actual_cache_dependent_prediction_count"] == 127
    assert result["same_conditioning_prefix"]


def test_common_prefill_does_not_mask_first_cache_dependent_failure(driver):
    reference = torch.zeros(128, 4)
    candidate = reference.clone()
    candidate[32, 1] = 2
    result = driver.comparisons(reference, candidate)
    assert result["registered_guard_32"]["registered_guard_pass"]
    assert not result["cache_dependent_guard_32"]["registered_guard_pass"]


@pytest.mark.parametrize(
    "field", ["answer", "reference", "references", "draft_tokens", "score", "annotation"]
)
def test_payload_rejects_any_benchmark_annotation(driver, tmp_path, field):
    path = tmp_path / "input.json"
    value = dict(task="consumed", sample_id="x", prompt="hello", media=[])
    path.write_text(json.dumps(value))
    assert driver.label_blind_payload(path) == value
    value[field] = "not permitted"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="no labels"):
        driver.label_blind_payload(path)


@dataclasses.dataclass
class Packet:
    codes: object
    scale: object
    exact: object
    corrections: object
    row_positions: object
    sin: object

    def tensors(self):
        return self.codes, self.scale, self.exact, self.corrections, self.row_positions, self.sin


def packet():
    return Packet(
        torch.zeros(20, dtype=torch.uint8),
        torch.ones(2, dtype=torch.float32),
        torch.ones(3, dtype=torch.bfloat16),
        torch.ones(2, dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.ones(3, dtype=torch.bfloat16),
    )


def test_costs_include_headers_protection_corrections_rotary_and_masks(driver):
    result = driver.storage_components((packet(),))
    assert result["representation_storage_bytes"] == 20 + 8 + 6 + 8 + 16 + 6
    assert set(result["category_bytes"]) == {
        "quantization_codes",
        "quantization_scales_and_minima",
        "exact_protection_or_original_payload",
        "numerical_corrections",
        "logical_positions_masks_and_indices",
        "rotary_phases",
    }
    assert not result["whole_request_peak_certified"]
    assert result["CPU_residual_bytes"] == 0


def test_storage_aliases_count_backing_once_not_view_numel(driver):
    p = packet()
    p.scale = p.codes[3:5]
    first = driver.storage_components((p,))
    second = driver.storage_components((p, p))
    assert first["representation_storage_bytes"] == second["representation_storage_bytes"]
    assert first["representation_storage_bytes"] == 20 + 6 + 8 + 16 + 6
    code = next(row for row in second["components"] if "layer[0].codes" in row["paths"])
    assert code["bytes"] == 20
    assert code["declared_alias_count"] == 4


def test_actual_measured_family_selects_largest_feasible_not_nominal_fraction(driver):
    candidates = [
        dict(candidate_id="high-fraction", actual_bytes=101),
        dict(candidate_id="low-fraction", actual_bytes=97),
        dict(candidate_id="nonmonotonic", actual_bytes=99),
    ]
    result = driver.choose_attainable(candidates, 100)
    assert result["chosen_candidate_id"] == "nonmonotonic"
    assert result["shortfall_bytes"] == 1
    assert result["utilization"] == 0.99
    assert not result["exact_ceiling_fill"]
    assert not result["globally_largest_cache_certified"]
    assert not result["equal_realized_byte_comparison"]


def test_feasibility_uses_real_bytes_and_reports_unsupported_family(driver):
    result = driver.choose_attainable([dict(candidate_id="only", actual_bytes=101)], 100)
    assert not result["feasible"]
    assert result["actual_bytes"] is None
    assert result["shortfall_bytes"] is None
    assert not result["largest_feasible_in_registered_family"]


def test_deterministic_candidate_ties_and_no_label_api(driver):
    import inspect

    candidates = [dict(candidate_id="b", actual_bytes=99), dict(candidate_id="a", actual_bytes=99)]
    assert driver.choose_attainable(candidates, 100)["chosen_candidate_id"] == "a"
    assert list(inspect.signature(driver.choose_attainable).parameters) == ["candidates", "cap"]
    assert "prefix" not in inspect.signature(driver._formats_with_ceiling).parameters


@pytest.mark.parametrize("cap", [None, 0, -1, True, 1.5])
def test_bad_ceiling_fails(driver, cap):
    with pytest.raises(ValueError):
        driver.choose_attainable([dict(candidate_id="a", actual_bytes=1)], cap)


def test_raw_topk_largest_uniform_capacity_at_or_below_cap(driver):
    capacity = driver.raw_topk_capacity(
        2 * 3 * 128 * 4 * 15 + 61, layers=2, heads=3, length=64, width=128, mandatory_count=12
    )
    assert capacity["count"] == 15
    assert capacity["shortfall_bytes"] == 61
    assert capacity["physical_padding_bytes"] == 0
    assert capacity["actual_bytes"] <= capacity["cap_bytes"]
    assert capacity["actual_bytes"] + capacity["physical_slot_bytes"] > capacity["cap_bytes"]


def test_mandatory_cache_cannot_fit_no_silent_eviction(driver):
    capacity = driver.raw_topk_capacity(
        1000, layers=2, heads=3, length=64, width=128, mandatory_count=12
    )
    assert not capacity["feasible"]
    assert capacity["count"] is None
    assert capacity["minimum_mandatory_bytes"] > capacity["cap_bytes"]


def test_source_ladder_uses_actual_cost_without_exposing_draft_tokens(driver, monkeypatch):
    import evicastkv_review_source_controls as previous

    observed = []

    def constructor(adapter, prepared, prefill, captured, normalized, phases, arm):
        masks, info = previous.protection_masks(torch.ones(2, 100), (), seed=13)
        count = info["optional_count_per_head"]
        observed.append(count)
        # Deliberately non-monotonic encoded costs: actual storage controls,
        # rather than assuming that protection fraction determines bytes.
        cost = {0: 51, 3: 55, 5: 59}[count]
        p = packet()
        p.codes = torch.zeros(cost, dtype=torch.uint8)
        return (p,), [dict(optional=count, mandatory=masks["mandatory"].sum().item())]

    monkeypatch.setattr(previous, "_formats", constructor)
    record = {}
    # Non-code costs are 44 bytes, so the middle family is the largest feasible.
    formats, proof = driver._formats_with_ceiling(
        None, None, None, None, None, None, "source3_attention_optional", 100, record
    )
    assert observed == [0, 3, 5]
    assert formats[0].codes.numel() == 55
    assert proof[0]["optional"] == 3
    assert record["allocation"]["actual_bytes"] == 99
    assert record["allocation"]["shortfall_bytes"] == 1
    assert not record["uncapped_numerical_diagnostic"]


def test_infeasible_source_is_rejected_before_decode(driver, monkeypatch):
    import evicastkv_review_source_controls as previous

    monkeypatch.setattr(previous, "_formats", lambda *args: ((packet(),), []))
    with pytest.raises(driver.InfeasibleCeiling) as error:
        driver._formats_with_ceiling(
            None, None, None, None, None, None, "source3_mandatory_only", 1, {}
        )
    assert not error.value.evidence["allocation"]["feasible"]


def test_unfit_direct_quantizer_is_reported_uncapped_not_falsely_matched(driver, monkeypatch):
    import evicastkv_review_source_controls as previous

    monkeypatch.setattr(previous, "_formats", lambda *args: ((packet(),), []))
    record = {}
    driver._formats_with_ceiling(
        None,
        None,
        None,
        None,
        None,
        None,
        "direct_projected_kv3_attention_optional_local",
        1,
        record,
    )
    assert not record["allocation"]["feasible"]
    assert record["physical_ceiling_unsupported"]
    assert record["uncapped_numerical_diagnostic"]
    assert not record["allocation"]["equal_realized_byte_comparison"]


def test_topk_constructor_retains_original_values_mandatory_and_positions(driver):
    torch.manual_seed(2)
    source = [
        (torch.randn(1, 2, 32, 128).bfloat16(), torch.randn(1, 2, 32, 128).bfloat16())
        for _ in range(2)
    ]
    original = [(k.clone(), v.clone()) for k, v in source]
    adapter = SimpleNamespace(_legacy_layers=lambda _: (source, None))
    prepared = SimpleNamespace(modality_map=[SimpleNamespace(modality="text", start=15, end=17)])
    prefill = SimpleNamespace(state=SimpleNamespace(past_key_values="original"))
    captured = SimpleNamespace(
        windows={i: (torch.arange(2 * 1 * 32).reshape(2, 1, 32).float(),) for i in range(2)}
    )
    cap = 2 * 2 * 128 * 4 * 20 + 37
    formats, proof, record = driver._topk_formats(adapter, prepared, prefill, captured, cap)
    assert all(f.shape == (1, 2, 20, 128) for f in formats)
    assert all(p["mandatory_preserved"] for p in proof)
    assert record["inventory"]["representation_storage_bytes"] == cap - 37
    assert record["allocation"]["physical_padding_bytes"] == 0
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for x, y in zip(source, original, strict=True)
        for a, b in zip(x, y, strict=True)
    )
    assert all(f.pair[0].raw for f in formats)


def test_request_injects_same_prefix_at_decode_only_and_caps_cache(driver, monkeypatch):
    import evicastkv_review_source_controls as previous

    from mosaickv.evaluation import storage_runs

    prefix = tuple(range(128))
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(128, 2), {}

    monkeypatch.setattr(storage_runs, "run", run)

    def parent_request(adapter, payload, cfg, arm):
        logits, _ = storage_runs.run(adapter, "prefill", ("formats",), tokens=128)
        return logits, dict(
            storage=dict(
                initial_active_kv_storage_bytes=50,
                final_active_kv_storage_bytes=60,
                peak_active_kv_storage_bytes=80,
            )
        )

    monkeypatch.setattr(previous, "_request", parent_request)
    logits, row = driver.request(
        "adapter", {}, "cfg", "source3_attention_optional", prefix=prefix, cap=100
    )
    assert calls == [dict(tokens=128, forced=prefix, initial_physical_byte_cap=100)]
    assert logits.shape == (128, 2)
    assert row["fixed_FullKV_prefix"]
    assert row["selection_prefix_access"] is False
    assert row["exact_realized_byte_match_claim"] is False
    assert row["realized_initial_bytes"] == 50


def test_reference_is_greedy_and_uncapped_clone_is_forced(driver, monkeypatch):
    import evicastkv_review_source_controls as previous

    from mosaickv.evaluation import storage_runs

    calls = []
    monkeypatch.setattr(
        storage_runs,
        "run",
        lambda *args, **kwargs: (calls.append(kwargs) or torch.zeros(128, 2), {}),
    )

    def parent_request(adapter, payload, cfg, arm):
        assert arm == "fullkv_native_reference"
        logits, _ = storage_runs.run(adapter, "prefill", (), tokens=128)
        return logits, dict(
            storage=dict(
                initial_active_kv_storage_bytes=50,
                final_active_kv_storage_bytes=60,
                peak_active_kv_storage_bytes=80,
            )
        )

    monkeypatch.setattr(previous, "_request", parent_request)
    native_calls = []
    monkeypatch.setattr(
        driver,
        "native_reference",
        lambda adapter, payload: (native_calls.append(payload) or torch.zeros(128, 2), {}),
    )
    driver.request("a", {}, "cfg", "fullkv_native_reference", prefix=None, cap=10)
    driver.request("a", {}, "cfg", "fullkv_forced_clone_control", prefix=(0,) * 128, cap=10)
    assert calls == [dict(tokens=128, forced=(0,) * 128)]
    assert native_calls == [{}]


def test_actual_native_reference_uses_unmodified_cache_and_new_returned_states(driver, monkeypatch):
    import evicastkv_request_validation as parent_runtime

    monkeypatch.setattr(parent_runtime, "load_messages", lambda payload: payload)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class Event:
        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Event())
    initial = tuple(torch.zeros(1, 1, 5, 4).bfloat16() for _ in range(2))
    inputs = []

    def state(pair, position):
        return SimpleNamespace(
            past_key_values=(pair,), logical_sequence_length=5, next_decode_position=position
        )

    def decode(token, previous, **kwargs):
        inputs.append((token.item(), previous.next_decode_position))
        assert kwargs == dict(capture_queries=False)
        pair = previous.past_key_values[0]
        pair = tuple(
            torch.cat((tensor, torch.zeros_like(tensor[..., :1, :])), dim=2) for tensor in pair
        )
        return SimpleNamespace(
            state=state(pair, previous.next_decode_position + 1),
            next_token_id=torch.tensor([[1]]),
            logits=torch.tensor([[0.0, 1.0]]),
        )

    adapter = SimpleNamespace(
        prepare_inputs=lambda payload: payload,
        prefill=lambda prepared, **kwargs: SimpleNamespace(
            state=state(initial, 5),
            logits=torch.tensor([[0.0, 1.0]]),
            next_token_id=torch.tensor([[1]]),
        ),
        _legacy_layers=lambda cache: (cache, None),
        decode_one_token=decode,
    )
    logits, row = driver.native_reference(adapter, {})
    assert logits.shape == (128, 2)
    assert len(inputs) == 127
    assert inputs[0] == (1, 5) and inputs[-1] == (1, 131)
    assert row["fullkv_unmodified_HF_cache"]
    assert not row["source_codec_or_history_transport"]
    assert row["storage"]["initial_active_kv_storage_bytes"] == 80
    assert row["storage"]["final_active_kv_storage_bytes"] == 132 * 16
    assert row["final_next_position"] == 132


@pytest.mark.parametrize(
    "arm,prefix",
    [
        ("source3_attention_optional", None),
        ("fullkv_native_reference", (1,) * 128),
        ("unknown", (1,) * 128),
    ],
)
def test_reference_order_is_fail_closed(driver, arm, prefix):
    with pytest.raises(ValueError, match="reference first"):
        driver.request(None, {}, None, arm, prefix=prefix, cap=100)


def test_baseline_claims_are_not_upgraded_by_diagnostic(driver):
    import evicastkv_review_source_controls as previous

    assert all(
        not r["official_fidelity_for_this_new_diagnostic"] for r in previous.baseline_inventory()
    )


def test_no_gpu_query_on_module_import(driver):
    assert "torch" not in driver.__dict__
    assert "os" in driver.__dict__
    assert driver.DEFAULT_PARENT.name == "source"
