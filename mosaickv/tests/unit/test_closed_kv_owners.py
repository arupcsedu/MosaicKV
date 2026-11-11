"""Synthetic CPU ownership proof tests; no measured model certificate."""

import ast
import dataclasses
import hashlib
import json

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.graphed_storage import SharedSourceCache, SharedSourceLayer
from mosaickv.evaluation.lossless_source_control import LosslessSourceFormat
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache
from mosaickv.measurements import closed_kv_owners as closed
from mosaickv.measurements.native_event_ledger import NativeEventLedger


def row(action, addr=100, size=64):
    return {"action": action, "addr": addr, "size": size, "device": 0, "pool": [0, 0]}


def capture(*events):
    return {
        "schema": "native_numeric_allocator_trace_v1",
        "complete": True,
        "overflow": False,
        "count": len(events),
        "capacity": max(1, len(events)),
        "synthetic_test_only": False,
        "events": [dict(event, sequence=i) for i, event in enumerate(events)],
    }


def witness(cursor, addr=100, size=64, end=None, roles=("persistent",)):
    return {
        "begin_cursor": cursor,
        "end_cursor": cursor if end is None else end,
        "owners": [{"device": "cuda:0", "addr": addr, "bytes": size, "roles": list(roles)}],
    }


def proof(tmp_path):
    contract = closed.source_contract()
    observer = tmp_path / "synthetic_observer_NOT_model_evidence.py"
    observer.write_text("# Synthetic CPU fixture, not a model escape review.\nx = 1\n")
    bindings = {
        str(observer): {
            "sha256": hashlib.sha256(observer.read_bytes()).hexdigest(),
            "ast_sha256": hashlib.sha256(
                ast.dump(ast.parse(observer.read_bytes()), include_attributes=False).encode()
            ).hexdigest(),
            "escape_sites": list(closed.ESCAPE_SITES),
        }
    }
    path = tmp_path / "explicit-source-review.json"
    path.write_text(
        json.dumps(
            {
                "source_contract": contract,
                "observer_source_bindings": bindings,
                "clauses": list(closed.CLAUSES),
                "reviewer_kind": "explicit_source_review_not_human_approval",
                "reviewer": "CPU synthetic fixture ONLY",
                "all_escape_sites_reviewed": True,
            }
        )
    )
    return {
        "schema": "closed_persistent_KV_escape_proof_v1",
        "source_contract": contract,
        "observer_source_bindings": bindings,
        "review_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, path


def test_live_views_deduplicate_full_backing_and_do_not_retain_owner():
    base = torch.empty(32)
    view = base[5:7]
    result = closed.live_witness(
        [(view, "persistent", "packet.source"), (base, "cache_workspace", "constructor")], lambda: 4
    )
    assert len(result["owners"]) == 1
    assert result["owners"][0]["bytes"] == 128
    assert result["owners"][0]["roles"] == ["persistent", "cache_workspace"]
    assert result["strong_references_released_on_return"]
    assert not any(isinstance(value, torch.Tensor) for value in result.values())


def test_live_cursor_and_role_errors():
    tensor = torch.empty(1)
    ticks = iter((3, 2))
    with pytest.raises(ValueError, match="backwards"):
        closed.live_witness([(tensor, "persistent", "x")], lambda: next(ticks))
    with pytest.raises(ValueError, match="semantic role"):
        closed.live_witness([(tensor, "automatic_nonKV", "x")], lambda: 1)


def test_early_allocation_and_pointer_reuse_bind_distinct_generations():
    raw = capture(row("alloc"), row("free_requested"), row("free_completed"), row("alloc"))
    records = NativeEventLedger().ingest(raw).records
    assert closed.bind_witnesses(records, [witness(1)]) == {0}
    assert closed.bind_witnesses(records, [witness(4)]) == {1}
    with pytest.raises(ValueError, match="exactly one"):
        closed.bind_witnesses(records, [witness(2, end=4)])


def test_alias_witnesses_union_roles_without_double_charge(tmp_path):
    raw = capture(row("alloc"))
    reviewed, artifact = proof(tmp_path)
    result = closed.reduce_metric_A(
        raw,
        baseline=[],
        final_blocks=[dict(row("alloc"))],
        witnesses=[witness(1), witness(1)],
        proof=reviewed,
        review_artifact=artifact,
        complete_escape_coverage=True,
    )
    assert result["conditional_metric_A"]["peak_requested_bytes"] == 64
    assert result["persistent_generations"] == 1
    assert result["metric_B_unknown_preserved"]
    assert result["exact_peak_certified"] is False
    assert result["exact_source_KV_peak_certified"] is False
    assert result["exact_whole_request_KV_peak_certified"] is False


def test_overlap_late_free_and_unknown_transient(tmp_path):
    raw = capture(
        row("alloc"),
        row("alloc", 200, 96),
        row("free_requested"),
        row("alloc", 300, 1024),
        row("free_requested", 300, 1024),
        row("free_completed", 300, 1024),
        row("free_completed"),
    )
    reviewed, artifact = proof(tmp_path)
    result = closed.reduce_metric_A(
        raw,
        baseline=[],
        final_blocks=[row("alloc", 200, 96)],
        witnesses=[witness(1), witness(2, 200, 96)],
        proof=reviewed,
        review_artifact=artifact,
        complete_escape_coverage=True,
    )
    assert result["conditional_metric_A"]["peak_requested_bytes"] == 160
    assert result["conditional_metric_A"]["final_requested_bytes"] == 96
    assert result["other_generations"] == 1
    assert result["metric_B_unknown_preserved"]


def test_missing_review_coverage_source_tamper_and_synthetic_refused(tmp_path):
    raw = capture(row("alloc"))
    reviewed, artifact = proof(tmp_path)
    arguments = dict(
        baseline=[],
        final_blocks=[row("alloc")],
        witnesses=[witness(1)],
        proof=reviewed,
        review_artifact=artifact,
    )
    with pytest.raises(ValueError, match="escape sites"):
        closed.reduce_metric_A(raw, **arguments)
    bad = dict(reviewed, source_contract={})
    with pytest.raises(ValueError, match="source/schema"):
        closed.validate_escape_proof(bad, artifact)
    with pytest.raises(ValueError, match="synthetic"):
        closed.reduce_metric_A(
            dict(raw, synthetic_test_only=True), **arguments, complete_escape_coverage=True
        )
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="source/schema"):
        closed.validate_escape_proof(reviewed, artifact)


def test_incomplete_final_baseline_and_overflow_refused(tmp_path):
    raw = capture(row("alloc"))
    reviewed, artifact = proof(tmp_path)
    arguments = dict(
        baseline=[],
        final_blocks=[],
        witnesses=[witness(1)],
        proof=reviewed,
        review_artifact=artifact,
        complete_escape_coverage=True,
    )
    with pytest.raises(ValueError, match="final requested"):
        closed.reduce_metric_A(raw, **arguments)
    with pytest.raises(ValueError, match="overflow"):
        closed.reduce_metric_A(dict(raw, overflow=True), **arguments)


def test_observer_source_and_ast_must_be_bound_not_coverage_boolean(tmp_path):
    reviewed, artifact = proof(tmp_path)
    with pytest.raises(ValueError, match="observer source/AST"):
        closed.validate_escape_proof(dict(reviewed, observer_source_bindings={}), artifact)
    source = next(iter(reviewed["observer_source_bindings"]))
    from pathlib import Path

    Path(source).write_text("x = 'unreviewed mutation'\n")
    with pytest.raises(ValueError, match="source/AST/site binding changed"):
        closed.validate_escape_proof(reviewed, artifact)


def packet():
    n, h, d = 2, 4, 128
    normalized = torch.zeros((1, n, h * d), dtype=torch.bfloat16)
    phases = tuple(torch.zeros((1, n, d), dtype=torch.bfloat16) for _ in range(2))
    patches = tuple(
        SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
        for _ in range(2)
    )
    return LosslessSourceFormat(
        normalized,
        *phases,
        torch.arange(n, dtype=torch.int32),
        patches,
        (1, h, n, d),
        2 * h * n * d * 2,
        (),
    )


def source_cache():
    config = LlamaConfig(
        hidden_size=512, num_attention_heads=4, num_key_value_heads=4, intermediate_size=1024
    )
    config._attn_implementation = "eager"
    attention = LlamaAttention(config, layer_idx=0).eval()
    cache = SharedSourceCache(audit_workspace=True, constructor="direct", graphs=False)
    layer = SharedSourceLayer(
        packet(),
        attention,
        observe=cache.observe,
        direct=True,
        updates=128,
        audit=True,
        cache=cache,
    )
    cache.layers.append(layer)
    cache.shared_tensors = layer.representation.shared()
    state = DecodeState(
        cache, torch.ones((1, 2), dtype=torch.long), 2, 2, 2, (), {"mosaickv_cache_all_valid": True}
    )
    return cache, layer, state


def test_retiring_cache_schema_and_extra_owner_refused():
    cache = RetiringPrefillCache(1)
    assert closed.audit_schema(cache)["missing_state_roots"]
    factory = cache.layer_classes
    cache.layer_classes = object
    with pytest.raises(ValueError, match="foreign layer allocation factory"):
        closed.audit_schema(cache)
    cache.layer_classes = factory
    cache.hidden_tensor = torch.ones(1)
    with pytest.raises(ValueError, match="field drift"):
        closed.audit_schema(cache)


def test_source_schema_includes_decode_buffers_and_excludes_only_known_lease():
    cache, layer, state = source_cache()
    lease = tuple(torch.empty((1, 4, 130, 128), dtype=torch.bfloat16) for _ in range(2))
    cache.workspace = lease
    layer.leased = tuple(value[..., :2, :] for value in lease)
    cache.shared_tensors += lease
    state.model_state["mosaickv_decode_attention_mask_buffer"] = torch.ones((1, 512))
    state.model_state["mosaickv_decode_position_buffer"] = torch.arange(512)
    report = closed.audit_schema(cache, states=(state,))
    assert not report["missing_state_roots"]
    tagged = report["tagged_owners"]
    assert all(
        role == "cache_workspace"
        for value, role, _ in tagged
        if value.untyped_storage().data_ptr() == lease[0].untyped_storage().data_ptr()
    )
    assert any(
        "decode_position_buffer" in path and role == "persistent" for _, role, path in tagged
    )
    assert report["whole_request_escape_review_passed"] is False
    assert report["metric_B_classified"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "hidden_packet",
        "subclass",
        "shared_unknown",
        "state_unknown",
        "graph",
        "hook",
        "nested_metadata_tensor",
        "model_buffer",
    ],
)
def test_closed_schema_escape_violations(defect):
    cache, layer, state = source_cache()
    if defect == "hidden_packet":
        object.__setattr__(layer.representation, "hidden", torch.ones(1))
    elif defect == "subclass":

        class Foreign(type(layer.representation)):
            pass

        layer.representation = Foreign(
            **{
                field.name: getattr(layer.representation, field.name)
                for field in dataclasses.fields(layer.representation)
            }
        )
    elif defect == "shared_unknown":
        cache.shared_tensors += (torch.ones(1),)
    elif defect == "state_unknown":
        state.model_state["hidden"] = torch.ones(1)
    elif defect == "graph":
        cache.graphs_enabled = True
    elif defect == "hook":
        layer.attention.register_forward_hook(lambda *args: None)
    elif defect == "model_buffer":
        layer.attention.register_buffer("unregistered", torch.ones(1))
    else:
        cache.build_records = [{"looks_like_metadata": torch.ones(1)}]
    with pytest.raises(ValueError):
        closed.audit_schema(cache, states=(state,))
