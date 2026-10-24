"""Synthetic CPU feasibility only; never pretrained answer-quality evidence."""

import inspect
import json

import numpy as np
import pytest
import torch

from mosaickv.cache_state import FullKVState
from mosaickv.selection.direct_error_exchange import (
    MAX_EXCHANGES,
    MAX_VERIFIED_PAIRS,
    LayerContext,
    build_direct_error_layer_plan,
    cpu_feasibility,
    validate_exchange_trace,
)
from mosaickv.selection.evidence_layer_allocation import (
    RECONSTRUCTION_ATOL,
    RECONSTRUCTION_RTOL,
    reconstruction_mse,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(*, heads=2, negative=False):
    values = (
        ([0.0, 2.0, 1.0, 1.0],) * 2
        if negative
        else ([0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    )
    length = len(values[0])
    capacity = length // 2
    layers = tuple(
        (
            torch.arange(length).float().reshape(1, 1, length, 1).repeat(1, heads, 1, 1),
            torch.tensor(row).reshape(1, 1, length, 1).repeat(1, heads, 1, 1),
        )
        for row in values
    )
    state = FullKVState.from_tensors(
        layers,
        block_size=1,
        mandatory_logical_positions=(capacity - 1,),
    )
    attention = {(i, h): np.ones(length) / length for i in range(2) for h in range(heads)}
    records = tuple(
        dict(layer=i, kv_head=h, logical_position=p, tier="exact")
        for i in range(2)
        for h in range(heads)
        for p in range(capacity)
    )
    return state, attention, records, capacity


@pytest.mark.parametrize("remove", [True, False])
def test_analytic_move_matches_bruteforce_renormalized_output(remove):
    generator = torch.Generator().manual_seed(292)
    mass = torch.rand(3, 12, generator=generator)
    values = torch.randn(3, 12, 7, generator=generator)
    positions = ((0, 2, 4, 6, 8, 10),) * 3
    mandatory = ((0,),) * 3
    context = LayerContext(mass, values, positions, mandatory)
    actual = context.best_move(remove=remove)
    choices = []
    for head in range(3):
        options = []
        for p in range(12):
            if remove and (p not in positions[head] or p in mandatory[head]):
                continue
            if not remove and p in positions[head]:
                continue
            row = set(positions[head])
            row.remove(p) if remove else row.add(p)
            error = reconstruction_mse(
                mass[head : head + 1], values[head : head + 1], (tuple(sorted(row)),)
            ).item()
            base = context.errors[head].item()
            if error <= base + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * base:
                options.append((error, p))
        choices.append(min(options) if options else None)
    if any(choice is None for choice in choices):
        assert actual is None
    else:
        assert actual.positions == tuple(choice[1] for choice in choices)
        assert actual.errors == pytest.approx(tuple(choice[0] for choice in choices), abs=1e-12)


def test_positive_feasibility_exact_byte_layer_exchange_and_source_isolation():
    state, attention, records, capacity = fixture(heads=32)
    saved = [(layer.key.clone(), layer.value.clone()) for layer in state.layers]
    plan = build_direct_error_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    trace = json.loads(json.dumps(plan.trace(), allow_nan=False))
    check = validate_exchange_trace(trace)
    search = trace["reconstruction_guard"]["exchange_search"]
    assert search["committed_exchanges"] > 0
    assert search["changed_from_anchor"]
    assert search["initial_safe_donor_layers"] > 0
    assert search["initial_safe_receiver_layers"] > 0
    assert search["context_builds"] == 2
    assert search["option_layer_refreshes"] == 2 + 2 * search["committed_exchanges"]
    first = search["transactions"][0]
    assert first["donor_layer"] == 0 and first["receiver_layer"] == 1
    assert first["capacities_after"] == [capacity - 1, capacity + 1]
    assert plan.prompt_bytes == state.active_bytes // 2
    assert sum(plan.capacities) == 2 * capacity
    assert check["final_mse_mean"] < check["anchor_mse_mean"]
    assert plan.trace()["padding_bytes"] == 0
    assert search["committed_exchanges"] <= MAX_EXCHANGES
    assert search["verified_pairs"] <= MAX_VERIFIED_PAIRS
    for heads in plan.positions:
        assert len(set(map(len, heads))) == 1
        for row in heads:
            assert capacity - 1 in row
            assert row == tuple(sorted(set(row)))
    for before, layer in zip(saved, state.layers, strict=True):
        torch.testing.assert_close(before[0], layer.key, atol=0, rtol=0)
        torch.testing.assert_close(before[1], layer.value, atol=0, rtol=0)


def test_cpu_feasibility_gate_is_explicitly_synthetic_and_exercises_no_move_case():
    result = cpu_feasibility()
    assert result["passed"]
    assert "not_pretrained_quality" in result["classification"]
    assert not result["benchmark_labels_accessed"]
    assert result["cases"]["random_32_layers_32_heads"]["committed_exchanges"] > 0
    assert result["cases"]["negative_no_safe_donor"]["committed_exchanges"] == 0


@pytest.mark.parametrize("mutation", ["bytes", "head_regression", "capacity", "changed", "gain"])
def test_trace_replay_rejects_corrupted_atomic_transactions(mutation):
    state, attention, records, capacity = fixture()
    plan = build_direct_error_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    trace = json.loads(json.dumps(plan.trace()))
    search = trace["reconstruction_guard"]["exchange_search"]
    event = search["transactions"][0]
    if mutation == "bytes":
        event["transferred_bytes"] += 1
    elif mutation == "head_regression":
        event["donor_mse_after"][0] += 10
    elif mutation == "capacity":
        event["capacities_after"][0] += 1
    elif mutation == "changed":
        search["changed_from_anchor"] = False
    else:
        event["summed_head_mse_gain"] += 10
    with pytest.raises(ValueError):
        validate_exchange_trace(trace)


def test_search_respects_explicit_verification_and_exchange_limits(monkeypatch):
    import mosaickv.selection.direct_error_exchange as exchange

    state, attention, records, capacity = fixture()
    monkeypatch.setattr(exchange, "MAX_EXCHANGES", 1)
    monkeypatch.setattr(exchange, "MAX_VERIFIED_PAIRS", 1)
    plan = build_direct_error_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    search = plan.reconstruction["exchange_search"]
    assert search["committed_exchanges"] == search["verified_pairs"] == 1
    assert search["stop_reason"] == "exchange_limit"
    assert sum(plan.capacities) == 2 * capacity


def test_zero_mass_and_mandatory_positions_cannot_be_evicted_to_create_empty_cache():
    context = LayerContext(torch.tensor([[0.0, 1.0]]), torch.zeros(1, 2, 1), ((1,),), ((),))
    assert context.best_move(remove=True) is None
    context = LayerContext(torch.ones(1, 3), torch.ones(1, 3, 1), ((0, 1),), ((0,),))
    assert context.best_move(remove=True).positions == (1,)
    assert context.best_move(remove=False).positions == (2,)


def test_direct_exchange_config_roundtrip_and_block_granularity_is_strict():
    from dataclasses import replace
    from pathlib import Path

    from mosaickv.config import (
        BoundedDecodeConfig,
        ConfigurationError,
        RunConfig,
        canonical_config,
        config_sha256,
        load_config,
    )

    base = load_config(
        Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    )
    cfg = replace(base, bounded_decode=BoundedDecodeConfig(32, 4, 0.2, "direct_error_exchange_v1"))
    assert config_sha256(RunConfig.from_mapping(canonical_config(cfg))) == config_sha256(cfg)
    assert "__direct_error_exchange_layers" in cfg.bounded_decode.effective_method(cfg.method)
    with pytest.raises(ConfigurationError, match="one-token"):
        replace(cfg, cache=replace(cfg.cache, block_size=4))


def test_exchanges_preserve_noncontiguous_logical_positions_and_next_decode_position():
    state, attention, records, capacity = fixture()
    mapped = tuple(3 * p for p in range(8))
    remapped = FullKVState.from_tensors(
        tuple((layer.key, layer.value) for layer in state.layers),
        block_size=1,
        logical_positions=mapped,
        original_logical_sequence_length=24,
        next_decode_position=30,
        mandatory_logical_positions=(9,),
    )
    mapped_records = tuple(
        dict(row, logical_position=3 * row["logical_position"]) for row in records
    )
    identity_plan = build_direct_error_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    mapped_plan = build_direct_error_layer_plan(
        remapped,
        attention,
        mapped_records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    assert identity_plan.positions == mapped_plan.positions
    assert remapped.logical_positions.physical_to_logical == mapped
    assert remapped.logical_positions.original_logical_sequence_length == 24
    assert remapped.logical_positions.next_decode_position == 30
    for heads in mapped_plan.positions:
        for positions in heads:
            logical = remapped.logical_positions.gather(positions)
            assert logical == tuple(sorted(logical))
            assert 9 in logical


def test_negative_feasibility_preserves_anchor_when_no_safe_donor_exists():
    state, attention, records, capacity = fixture(negative=True)
    plan = build_direct_error_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=capacity,
    )
    search = plan.reconstruction["exchange_search"]
    assert search["initial_safe_donor_layers"] == 0
    assert search["stop_reason"] == "no_safe_donor"
    assert search["committed_exchanges"] == 0
    assert not search["changed_from_anchor"]
    assert plan.positions == (((0, 1),) * 2,) * 2
    assert plan.prompt_bytes == state.active_bytes // 2


def test_deterministic_positions_and_no_benchmark_label_api():
    state, attention, records, capacity = fixture()

    def run():
        return build_direct_error_layer_plan(
            state,
            attention,
            records,
            prompt_budget_bytes=state.active_bytes // 2,
            uniform_capacity=capacity,
        )

    a, b = run(), run()
    assert a.positions == b.positions
    assert (
        a.reconstruction["exchange_search"]["transactions"]
        == b.reconstruction["exchange_search"]["transactions"]
    )
    params = inspect.signature(build_direct_error_layer_plan).parameters
    assert not {"answers", "references", "scores", "labels"} & set(params)
    with pytest.raises(TypeError):
        build_direct_error_layer_plan(
            state,
            attention,
            records,
            prompt_budget_bytes=1,
            uniform_capacity=capacity,
            answers=["forbidden"],
        )


@pytest.mark.parametrize("failure", ["budget", "missing_mandatory", "duplicates", "nan"])
def test_invalid_inputs_fail_closed(failure):
    state, attention, records, capacity = fixture()
    budget = state.active_bytes // 2
    if failure == "budget":
        budget += 1
    elif failure == "missing_mandatory":
        records = tuple(
            dict(r, logical_position=4) if r["logical_position"] == 3 else r for r in records
        )
    elif failure == "duplicates":
        records = (*records, records[0])
    else:
        attention[0, 0][0] = float("nan")
    with pytest.raises(ValueError):
        build_direct_error_layer_plan(
            state, attention, records, prompt_budget_bytes=budget, uniform_capacity=capacity
        )
