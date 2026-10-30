"""Pure rule and independent original-KV representation checks; no labels."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.evaluation.delimiter_control import source_sha
from mosaickv.evaluation.lossless_prompt_cache import construct_cache
from mosaickv.evaluation.projection_archive import ProjectionCapture, build_archive
from mosaickv.evaluation.projection_cache import construct_projection_cache
from mosaickv.evaluation.union_restoration import evaluate_union, forced_prefix, union_positions


def test_union_rule_is_deterministic_monotonic_and_no_donor_is_removed():
    original = (((0, 2), (1, 3)), ((0, 3), (0, 3)))
    expanded = union_positions(original)
    assert expanded == (((0, 1, 2, 3), (0, 1, 2, 3)), ((0, 3), (0, 3)))
    assert expanded == union_positions(original)
    assert all(
        set(old) <= set(new)
        for a, b in zip(original, expanded, strict=True)
        for old, new in zip(a, b, strict=True)
    )
    with pytest.raises(ValueError, match="uniform"):
        union_positions((((1, 0), (0, 2)),))


def test_union_reference_and_source_decoder_match_even_when_first_layer_is_shorter():
    torch.manual_seed(40)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("union-source").messages))
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    before = source_sha(pairs)
    original = (((0, 3), (0, 3)), ((0, 2), (1, 3)))
    expanded = union_positions(original)
    # Shared maps alias one union allocation rather than H copies.
    archive = build_archive(capture.records[1], expanded[1], encoded=True)
    assert (
        archive.selected.untyped_storage().data_ptr() == archive.union.untyped_storage().data_ptr()
    )
    assert archive.selected.stride(0) == 0
    raw = construct_cache(adapter, prefill.state.past_key_values, expanded, encode=False)
    reference, _ = forced_prefix(adapter, prefill, raw, [3 + i % 5 for i in range(32)])
    source = construct_projection_cache(adapter, prefill, capture, expanded, policy="source_all")
    actual, storage = forced_prefix(adapter, prefill, source, [3 + i % 5 for i in range(32)])
    assert torch.equal(reference, actual)
    assert source_sha(pairs) == before
    assert storage["outstanding_layer_leases"] == 0


def test_union_covering_full_source_passes_unchanged_guard_without_accessing_labels():
    from mosaickv.config import DraftAllocationGuardConfig
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference

    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("union-guard").messages))
    full_positions = (((0, 1, 2, 3), (0, 1, 2, 3)),) * 2
    full = standard_greedy_reference(adapter, prefill, full_positions)
    targets = full.argmax(-1).tolist()
    original = (((0, 2), (1, 3)),) * 2
    raw = construct_cache(adapter, prefill.state.past_key_values, original, encode=False)
    anchor, _ = forced_prefix(adapter, prefill, raw, targets)
    result = evaluate_union(
        adapter,
        prefill,
        capture,
        original,
        targets,
        full_logits=full,
        anchor_logits=anchor,
        guard_config=DraftAllocationGuardConfig(),
        maximum_reference_kl=0.02,
        maximum_logit_difference=1.0,
    )
    assert result["unchanged_relative_guard"]["accepted"]
    assert result["absolute_fullkv_limits_passed"]
    assert result["expanded_head_positions"] == 8
    assert result["no_donor_eviction"]


def test_disabling_workspace_audit_does_not_change_source_decode_or_claim_peak():
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("timed-source").messages))
    selections = (((0, 2), (1, 3)),) * 2
    values = []
    for audit in (True, False):
        cache = construct_projection_cache(
            adapter, prefill, capture, selections, policy="source_all", audit_workspace=audit
        )
        logits, record = forced_prefix(adapter, prefill, cache, [3 + i % 5 for i in range(32)])
        assert record["reconstruction_tensor_results_observed"] == audit
        assert bool(record["reconstruction_operator_observations"]) == audit
        values.append(logits)
    assert torch.equal(*values)
