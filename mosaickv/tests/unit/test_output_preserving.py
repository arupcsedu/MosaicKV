"""Synthetic correctness checks; these numbers are not benchmark results."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mosaickv.backends.output_preserving import (  # noqa: E402
    select_output_preserving,
    topk_anchor_capture,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan  # noqa: E402
from mosaickv.config import (  # noqa: E402
    ConfigurationError,
    OutputPreservingConfig,
    load_config,
)
from mosaickv.selection.output_preserving import (  # noqa: E402
    corrective_swaps,
    prepare_evidence,
    reconstruct,
)


def test_frozen_validation_merges_only_reviewed_unexecuted_disjoint_ledgers(tmp_path):
    from prepare_output_preserving_frozen_validation import merge_reviewed_ledgers
    from run_prefixkv_decode_preflight import digest

    route = {
        "schema_version": 1,
        "task": "coco_caption",
        "dataset_id": "lmms-lab/COCO-Caption",
        "dataset_revision": "c3213b54e173ae86517d481de17f31439d6f8c72",
        "split": "val",
    }
    paths = []
    jobs = []
    for index in range(2):
        path = tmp_path / f"{index}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    **route,
                    "samples": [
                        {
                            "sample_id": f"coco_caption:{index}",
                            "prompt": "caption",
                            "references": ["reference"],
                            "media": [],
                        }
                    ],
                }
            )
        )
        paths.append(path)
        jobs.append(
            {
                "ledger": str(path),
                "ledger_sha": digest(path),
                "output": str(tmp_path / f"unused-{index}"),
            }
        )
    merged, bindings = merge_reviewed_ledgers(
        paths, parent_plan={"jobs": jobs}, excluded_ids=set(), expected_samples=2
    )
    assert [sample["sample_id"] for sample in merged["samples"]] == [
        "coco_caption:0",
        "coco_caption:1",
    ]
    assert len(bindings) == 2
    with pytest.raises(ValueError, match="overlap inspected"):
        merge_reviewed_ledgers(
            paths,
            parent_plan={"jobs": jobs},
            excluded_ids={"coco_caption:1"},
            expected_samples=2,
        )
    executed = Path(jobs[0]["output"])
    executed.mkdir()
    (executed / "worker.json").write_text("{}")
    with pytest.raises(ValueError, match="already has execution output"):
        merge_reviewed_ledgers(
            paths, parent_plan={"jobs": jobs}, excluded_ids=set(), expected_samples=2
        )


def test_frozen_validation_protocol_cannot_authorize_paper_campaign():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_draft_frozen_validation_v1.yaml"
    ).read_text()
    assert "sample_count: 128" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol
    assert "minimum_memory_reduction: 0.30" in protocol
    assert "required_ci_lower_bound: 0.0" in protocol


def test_capacity_analysis_flattens_point_identity_for_parquet():
    from analyze_output_preserving_capacity_sweep import _parquet_comparison_rows

    actual = _parquet_comparison_rows(
        [
            {
                "reference": ("prompt_attention_topk", 40.0),
                "candidate": ("output_preserving_swaps2", 25.5),
                "candidate_minus_reference": -1.0,
            }
        ]
    )
    assert actual == [
        {
            "candidate_minus_reference": -1.0,
            "reference_label": "prompt_attention_topk",
            "reference_cap_mib": 40.0,
            "candidate_label": "output_preserving_swaps2",
            "candidate_cap_mib": 25.5,
        }
    ]


def test_reconstruction_matches_recomputed_selected_softmax():
    torch.manual_seed(12)
    scores, values = torch.randn(6, 9, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64)
    scores[0, 3:] = -torch.inf
    attention, values = prepare_evidence(scores.softmax(-1), values)
    selected = (0, 2, 4, 6)
    actual = reconstruct(attention, values, selected)
    expected = scores[:, selected].softmax(-1) @ values[list(selected)]
    torch.testing.assert_close(actual.output, expected)
    assert not torch.allclose(actual.numerator, expected)
    assert torch.equal(attention[0, 3:], torch.zeros(6, dtype=torch.float64))


def test_swaps_correct_value_distortion_and_preserve_capacity():
    # Top mass picks +10 but ignores the -10 cancellation in the full output.
    attention, values = prepare_evidence(
        torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64),
        torch.tensor([[10.0], [-10.0], [0.0], [0.0]], dtype=torch.float64),
    )
    options = OutputPreservingConfig(shortlist_size=4, max_swaps_per_head=1)
    a_before, v_before = attention.clone(), values.clone()
    result = corrective_swaps(attention, values, (0, 3), (3,), options)
    assert result.positions == (2, 3)
    assert result.final_error < result.initial_error
    assert len(result.swaps) == 1
    torch.testing.assert_close(attention, a_before, rtol=0, atol=0)
    torch.testing.assert_close(values, v_before, rtol=0, atol=0)
    anchor = corrective_swaps(
        attention, values, (0, 3), (3,), replace(options, max_swaps_per_head=0)
    )
    assert anchor.positions == (0, 3) and anchor.swaps == ()


def test_draft_gate_accepts_only_a_swap_that_improves_both_objectives():
    torch.manual_seed(34)
    prompt, values = prepare_evidence(
        torch.rand(3, 6, dtype=torch.float64),
        torch.randn(6, 2, dtype=torch.float64),
    )
    draft, _ = prepare_evidence(torch.rand(2, 6, dtype=torch.float64), values)
    options = OutputPreservingConfig(
        shortlist_size=6,
        max_swaps_per_head=1,
        pair_chunk_size=4,
        minimum_improvement=0,
    )
    prompt_only = corrective_swaps(prompt, values, (0, 1, 5), (5,), options)
    joint = corrective_swaps(
        prompt,
        values,
        (0, 1, 5),
        (5,),
        replace(options, draft_validation=True),
        validation_attention=draft,
    )
    assert prompt_only.positions == (1, 3, 5)
    assert joint.positions == (0, 2, 5)
    assert joint.final_error < joint.initial_error
    assert joint.validation_final_error < joint.validation_initial_error
    assert joint.rejected_validation_pairs > 0


def test_individual_queries_do_not_disappear_into_a_pooled_objective():
    attention = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    values = torch.tensor([[1.0], [-1.0]])
    individual, v = prepare_evidence(attention, values)
    pooled, _ = prepare_evidence(attention, values, pooled=True)
    assert individual.shape == (2, 2) and pooled.shape == (1, 2)
    assert reconstruct(individual, v, (0,)).error != reconstruct(pooled, v, (0,)).error


@pytest.mark.parametrize("seed", range(8))
def test_full_shortlist_matches_best_single_swap_and_is_repeatable(seed):
    torch.manual_seed(seed)
    a, v = prepare_evidence(
        torch.rand(4, 7, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    )
    options = OutputPreservingConfig(
        shortlist_size=7, max_swaps_per_head=1, pair_chunk_size=2, minimum_improvement=0.0
    )
    selected = (0, 1, 6)
    result = corrective_swaps(a, v, selected, (6,), options)
    feasible = [s for s in combinations(range(7), 3) if 6 in s and len(set(s) - set(selected)) <= 1]
    best = min(reconstruct(a, v, s).error for s in feasible)
    assert result.final_error == pytest.approx(best, abs=1e-12)
    assert result == corrective_swaps(a, v, selected, (6,), options)
    assert len(result.positions) == 3 and 6 in result.positions
    full = corrective_swaps(a, v, tuple(range(7)), (), options)
    assert full.positions == tuple(range(7)) and full.final_error < 1e-25


def test_zero_mass_is_rejected_not_floored_and_mandatory_is_enforced():
    a, v = prepare_evidence(torch.eye(2), torch.ones(2, 1))
    with pytest.raises(ValueError, match="attention mass"):
        reconstruct(a, v, (1,))
    with pytest.raises(ValueError, match="mandatory"):
        corrective_swaps(a, v, (0,), (1,), OutputPreservingConfig())
    with pytest.raises(ValueError, match="unique"):
        reconstruct(a, v, (0, 0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_attention_fails_closed(bad):
    with pytest.raises(ValueError):
        prepare_evidence(torch.tensor([[bad, 1.0]]), torch.ones(2, 1))


@pytest.mark.parametrize(
    "mapping",
    [
        {"max_swaps_per_head": -1},
        {"shortlist_size": 65},
        {"max_routed_heads": True},
        {"query_evidence": "draft"},
        {"minimum_query_mass": 0},
        {"minimum_error": float("nan")},
        {"unknown": 1},
    ],
)
def test_strict_candidate_config(mapping):
    with pytest.raises(ConfigurationError):
        OutputPreservingConfig.from_mapping(mapping)


def test_effective_method_distinguishes_corrective_capacity():
    assert (
        OutputPreservingConfig(max_swaps_per_head=2).effective_method
        == "mosaickv_exact__output_preserving_v1"
    )
    assert (
        OutputPreservingConfig(max_swaps_per_head=4).effective_method
        == "mosaickv_exact__output_preserving_v1__swaps4"
    )
    assert (
        OutputPreservingConfig(max_swaps_per_head=8).effective_method
        == "mosaickv_exact__output_preserving_v1__swaps8"
    )
    assert (
        OutputPreservingConfig(draft_validation=True).effective_method
        == "mosaickv_exact__output_preserving_v1__draft_gate"
    )
    with pytest.raises(ConfigurationError, match="individual query evidence"):
        OutputPreservingConfig(query_evidence="pooled", draft_validation=True)


def test_bridge_routes_by_error_preserves_exact_storage_and_positions():
    torch.manual_seed(79)
    keys = torch.randn(1, 2, 8, 3)
    values = torch.randn(1, 2, 8, 3)
    full = FullKVState.from_tensors(
        ((keys, values),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    capture = torch.rand(1, 4, 2, 8)
    capture /= capture.sum(-1, keepdim=True)
    before = (keys.clone(), values.clone(), capture.clone())
    options = OutputPreservingConfig(max_routed_heads=1, shortlist_size=8)
    scores = capture.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten().numpy()
    config = SimpleNamespace(
        output_preserving=options, forecasting=SimpleNamespace(prompt_window=2)
    )
    plan = select_output_preserving(
        full,
        (capture,),
        config,
        position_limits={(0, 0): 4, (0, 1): 4},
        target_bytes=192,
        prompt_attention_scores=scores,
    )
    worst = max(plan.routes, key=lambda r: r["initial_error"])
    assert worst["routed"]
    assert sum(r["routed"] for r in plan.routes) == 1
    assert plan.selected_bytes == 192
    state = plan.construct(full)
    assert state.statistics.active_kv_bytes == 192
    assert not state.prototypes.source_blocks and not state.residuals.source_blocks
    assert state.next_decode_position == full.next_decode_position
    assert state.original_logical_sequence_length == full.original_logical_sequence_length
    for block, key, value in zip(
        state.exact.blocks, state.exact.key_blocks, state.exact.value_blocks, strict=True
    ):
        h, p = block.kv_head, block.physical_cache_indices[0]
        torch.testing.assert_close(key, keys[:, h : h + 1, p : p + 1], rtol=0, atol=0)
        torch.testing.assert_close(value, values[:, h : h + 1, p : p + 1], rtol=0, atol=0)
    for after, original in zip((keys, values, capture), before, strict=True):
        torch.testing.assert_close(after, original, rtol=0, atol=0)
    for route in plan.routes:
        assert route["final_error"] <= route["initial_error"]
        assert route["evaluated_pairs"] <= options.max_swaps_per_head * options.shortlist_size**2


def test_fp16_anchor_pooling_matches_baseline_capture_rounding():
    source = SimpleNamespace(layers=(SimpleNamespace(kv_heads=2, sequence_length=8),))
    torch.manual_seed(30)
    raw = torch.rand(1, 8, 3, 8, dtype=torch.float16)
    expected = raw.reshape(1, 2, 4, 3, 8).mean(2)
    actual = topk_anchor_capture(source, (raw,))[0]
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual.float(), raw.float().reshape(1, 2, 4, 3, 8).mean(2))


def test_development_yaml_and_unsupported_backend_fail_closed():
    config = load_config(
        Path(__file__).resolve().parents[2] / "configs/development/output_preserving_exact_v1.yaml"
    )
    assert config.output_preserving.query_evidence == "individual"
    assert config.cache.retention_ratio > 0.99 and config.cache.retention_ratio < 1
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, execution=replace(config.execution, attention_implementation="sdpa"))
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, cache=replace(config.cache, block_size=4))
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, prototypes=replace(config.prototypes, enabled=True))
