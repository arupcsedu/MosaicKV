"""CPU audit fixtures are implementation controls, not measured quality data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from evicastkv_cpu_audit import PHASES, Evidence, guard_summary, paper_name, scalar_phases

from mosaickv.backends.hf_runtime import (
    _attention_uncertainty_by_head,
    _prompt_token_attention_by_head,
    _uncertain_head_routes,
)
from mosaickv.baselines.prefixkv import _select_positions, prefixkv_attention_scores


@pytest.mark.parametrize("numpy_path", [True, False])
def test_layer_head_mapping_and_entropy_winner_are_not_hardcoded(numpy_path):
    layers = [SimpleNamespace(kv_heads=2, sequence_length=6) for _ in range(3)]
    attention = []
    for layer in range(3):
        values = torch.zeros(1, 4, 3, 6)
        values[..., 0] = 1
        if layer == 2:
            values[:, 2:] = 1 / 6
        attention.append(values.numpy() if numpy_path else values)
    scores = _prompt_token_attention_by_head(
        SimpleNamespace(layers=layers),
        SimpleNamespace(attention_weights=tuple(attention)),
        prompt_window=2,
    )
    expected = {(layer, head) for layer in range(3) for head in range(2)}
    assert set(scores) == expected
    np.testing.assert_allclose(scores[2, 1], np.full(6, 1 / 6), rtol=1e-6)
    np.testing.assert_allclose(scores[1, 1], [1, 0, 0, 0, 0, 0], atol=1e-300)
    entropy = _attention_uncertainty_by_head(scores)
    assert entropy[2, 1] == pytest.approx(1)
    assert _uncertain_head_routes(
        expected, entropy, fraction=1 / 256, threshold=0, label="CPU audit"
    ) == ((2, 1),)


def test_entropy_ranking_is_input_sensitive_and_ties_are_lexicographic():
    heads = {(0, 4): np.array([0.5, 0.5]), (2, 1): np.array([1.0, 0.0])}
    for expected in ((0, 4), (2, 1)):
        assert _uncertain_head_routes(
            heads,
            _attention_uncertainty_by_head(heads),
            fraction=0.1,
            threshold=0,
            label="CPU audit",
        ) == (expected,)
        heads = {(0, 4): heads[2, 1], (2, 1): heads[0, 4]}
    assert _uncertain_head_routes(
        heads, dict.fromkeys(heads, 0.5), fraction=0.1, threshold=0, label="CPU audit"
    ) == ((0, 4),)


def test_router_rejects_incomplete_head_score_mapping():
    with pytest.raises(RuntimeError, match="one uncertainty value"):
        _uncertain_head_routes(
            {(0, 0), (1, 0)}, {(0, 0): 0.5}, fraction=0.1, threshold=0, label="audit"
        )


def test_renaming_changes_display_only_and_does_not_merge_variants():
    assert paper_name("mosaickv_exact") == "EviCastKV exact"
    assert paper_name("prefixkv_reimpl") == "prefixkv_reimpl"
    assert paper_name("mosaickv_exact") != paper_name("mosaickv_full")
    assert paper_name("unrecognized_fallback") == "unrecognized_fallback"


def trace_fixture():
    return {
        "experimental_evidence_policy": {
            "grouped_query_heads_averaged": False,
            "installed_output_guard": {
                "changed_positions": 3,
                "whole_cache_fallback": False,
                "routes": [
                    {
                        "layer": 2,
                        "kv_head": 1,
                        "proposal_accepted": True,
                        "rolled_back": False,
                        "mass_guard_passed": True,
                        "output_guard_passed": True,
                    }
                ],
            },
        },
        "candidate_graph_routing": {"graph_head_routes": [[2, 1]]},
    }


def test_guard_summary_uses_actual_route_identity_without_label_fields():
    summary = guard_summary([trace_fixture()])
    assert summary["selected_routes"] == {"2:1": 1}
    assert summary["counts"]["changed_positions"] == 3
    assert summary["counts"]["whole_cache_fallbacks"] == 0


def test_guard_summary_rejects_mismatched_graph_and_installed_route():
    trace = trace_fixture()
    trace["candidate_graph_routing"]["graph_head_routes"] = [[0, 4]]
    with pytest.raises(ValueError, match="identities disagree"):
        guard_summary([trace])


def test_nested_phases_and_excluded_instrumentation_are_not_double_counted():
    trace = {
        "timing_breakdown": dict.fromkeys(PHASES, 1.0),
        "planner_subphases": {"nested": 1000.0},
    }
    trace["timing_breakdown"]["trace_instrumentation"] = 50
    result = scalar_phases(trace, len(PHASES))
    assert result["compression_phase_sum"] == len(PHASES)
    assert result["compression_accounting_difference"] == 0
    assert result["excluded_trace_instrumentation"] == 50


def test_input_binding_detects_mutation(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    evidence = Evidence()
    evidence.read(path)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="input changed"):
        evidence.read(path)


def test_pinned_upstream_full_query_scores_and_selected_positions(tmp_path, monkeypatch):
    """Direct upstream CPU fixture; not model/checkpoint parity evidence."""
    repository = Path(__file__).resolve().parents[3]
    upstream = repository / "third_party/PrefixKV/prefixkv.py"
    spec = importlib.util.spec_from_file_location("audited_prefixkv_upstream", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "confs").mkdir()
    (tmp_path / "confs/prefixkv_toy_0.5.json").write_text(json.dumps([0.25, 0.5]))
    generator = torch.Generator().manual_seed(51)
    attention = []
    kv = []
    for _ in range(2):
        weights = torch.rand(1, 2, 8, 8, generator=generator)
        weights *= torch.ones(8, 8).tril()
        weights /= weights.sum(dim=-1, keepdim=True)
        attention.append(weights)
        keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1).expand(1, 2, 8, 1)
        kv.append((keys.clone(), keys.clone()))
    official = module.PrefixKV(model_name="toy", start_size=1, ratio=0.5, layer_num=2)
    result = official(kv, num_of_token=8, attentions=attention)
    local_scores = prefixkv_attention_scores(tuple(attention))
    expected_scores = torch.stack([a.mean(dim=1) for a in attention]).float().sum(dim=-2)
    for layer, retained in enumerate((6, 4)):
        np.testing.assert_array_equal(local_scores[layer], expected_scores[layer, 0].numpy())
        selected, protected = _select_positions(local_scores[layer], retained, 1, 1)
        assert tuple(int(v) for v in result[layer][0][0, 0, :, 0]) == selected
        assert protected == (0, 7)
        assert result[layer][0].shape[2] == retained
