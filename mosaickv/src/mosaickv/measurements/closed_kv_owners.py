"""Fail-closed primitives for a separately reviewed persistent-KV boundary.

This is not an observer or a certificate for the current runtime. Live marks
identify generations; ONLY an independently registered, source-bound exhaustive
escape proof can establish that unmarked generations are not persistent KV.
Constructor/reader/validation workspace (metric B) remains unknown here.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import types
import weakref
from enum import Enum
from pathlib import Path

from mosaickv.measurements.native_event_ledger import NativeEventLedger

BOUNDARY = "persistent_KV_representation_plus_metadata"
CLAUSES = (
    "all_persistent_owner_escape_sites_reviewed_not_observation_counts",
    "all_prefill_and_decode_state_roots_including_masks_and_positions",
    "every_persistent_publication_marked_while_backing_is_strongly_alive",
    "fresh_empty_cache_no_preexisting_persistent_KV_owners",
    "all_pending_and_retired_persistent_generations_charged_until_free_completed",
    "fixed_known_classes_no_hidden_tensor_fields_graphs_offload_or_foreign_hooks",
    "unmarked_transients_are_not_metric_A_only_metric_B_stays_unknown",
)
ESCAPE_SITES = (
    "original_prefill_KV_publication",
    "prompt_packet_publication",
    "generated_raw_tail_publication",
    "normalized_tail_publication",
    "phase_tail_publication",
    "history_packet_publication",
    "history_phase_clones_publication",
    "prefill_state_metadata_publication",
    "decode_mask_and_position_replacement",
    "request_terminal_frees",
)
_SOURCE_NAMES = (
    "evaluation/lossless_source_control.py",
    "evaluation/exact_storage.py",
    "evaluation/lossless_bf16.py",
    "evaluation/projection_archive.py",
    "evaluation/dual_projection.py",
    "evaluation/encoded_projection.py",
    "evaluation/streaming_lossless_source.py",
    "evaluation/streaming_prefill.py",
    "evaluation/source_history.py",
    "evaluation/sealed_history.py",
    "evaluation/graphed_storage.py",
    "evaluation/lossless_prompt_cache.py",
    "evaluation/storage_runs.py",
    "adapters/huggingface/types.py",
    "adapters/huggingface/base.py",
    "measurements/closed_kv_owners.py",
    "measurements/native_event_ledger.py",
    "measurements/snapshot_states.py",
    "measurements/native_allocator_events.py",
    "measurements/csrc/native_allocator_events.cpp",
)
_COMMON_CACHE = {
    "layers",
    "layer_classes",
    "layer_init_kwargs",
    "num_hidden_layers",
    "cache_processor",
}
_RETIRING = _COMMON_CACHE | {
    "next_layer",
    "peak_bytes",
    "audit",
    "observations",
    "original_refs",
    "_extra_tensors",
    "_owned_storage",
    "_owned_device_bytes",
    "peak_device_bytes",
    "_layer_storage",
    "_extra_storage",
    "_catalog_rebuilds",
    "_catalog_identity_calls",
    "_independent_checks",
}
_SOURCE_CACHE = _COMMON_CACHE | {
    "initial_bytes",
    "peak_bytes",
    "observations",
    "build_records",
    "shared_tensors",
    "shared_bytes",
    "audit_workspace",
    "source_constructor",
    "source_check_flags_checked",
    "source_check_syncs",
    "source_history_seals",
    "phase_tail",
    "phase_identity",
    "history_position_chunks",
    "history_rotary_chunks",
    "consolidated_phase_versions",
    "source_consolidation_layers",
    "source_consolidation_epochs",
}
_SHARED_CACHE = _SOURCE_CACHE | {
    "graphs_enabled",
    "workspace",
    "workspace_allocations",
    "graph_captures",
    "graph_replays",
    "graph_private_allocation_bytes",
    "graph_private_live_bytes",
    "graph_private_accounting_verified",
    "workspace_capacity",
}
_LAYER = {
    "keys",
    "values",
    "representation",
    "attention",
    "direct",
    "audit",
    "workspace_operations",
    "weights",
    "prompt_shape",
    "prompt_length",
    "history_length",
    "history_capacity",
    "history",
    "leased",
    "resident_bytes",
    "history_bytes",
    "lease_bytes",
    "observe",
    "eviction_count",
    "fused_unpack",
    "maximum_decode_updates",
    "sealed",
    "tail_used",
    "normalized_tail",
    "source_sealed",
    "source_pending",
}
_STATE_TENSORS = {
    "mosaickv_decode_attention_mask_buffer",
    "mosaickv_decode_position_buffer",
}
_STATE_METADATA = {
    "mosaickv_prefill_attention_collection",
    "mosaickv_cache_all_valid",
    "mosaickv_packed_all_valid",
    "mosaickv_decode_attention_mask_capacity",
    "mosaickv_decode_position_capacity",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_contract():
    """Registration template, NOT an assertion that the escape review passed."""
    import transformers.cache_utils as installed_cache

    package = Path(__file__).resolve().parents[1]
    paths = [package / name for name in _SOURCE_NAMES] + [Path(installed_cache.__file__)]
    return {
        "boundary": BOUNDARY,
        "source_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "schema_sha256": _digest(
            {
                "retiring": sorted(_RETIRING),
                "source": sorted(_SOURCE_CACHE),
                "shared": sorted(_SHARED_CACHE),
                "layer": sorted(_LAYER),
                "state_tensors": sorted(_STATE_TENSORS),
                "state_metadata": sorted(_STATE_METADATA),
            }
        ),
        "required_clauses": list(CLAUSES),
        "required_escape_sites": list(ESCAPE_SITES),
        "review_is_still_required": True,
    }


def validate_escape_proof(proof, review_artifact):
    """Hash/source-bound explicit review; booleans/counts alone never suffice."""
    path = Path(review_artifact)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError("bounded regular source-review artifact required")
    raw = path.read_bytes()
    review = json.loads(raw)
    bindings = proof.get("observer_source_bindings")
    if not isinstance(bindings, dict) or not 1 <= len(bindings) <= 8:
        raise ValueError(
            "reviewed actual observer source/AST bindings required, not a coverage boolean"
        )
    covered = set()
    for name, binding in bindings.items():
        source = Path(name)
        if (
            not source.is_absolute()
            or source.is_symlink()
            or not source.is_file()
            or source.stat().st_size > 2**20
        ):
            raise ValueError("bounded immutable observer source required")
        text = source.read_bytes()
        structural = ast.dump(ast.parse(text), include_attributes=False)
        sites = binding.get("escape_sites")
        if (
            hashlib.sha256(text).hexdigest() != binding.get("sha256")
            or hashlib.sha256(structural.encode()).hexdigest() != binding.get("ast_sha256")
            or not isinstance(sites, list)
            or set(sites) - set(ESCAPE_SITES)
        ):
            raise ValueError("observer source/AST/site binding changed")
        covered.update(sites)
    if covered != set(ESCAPE_SITES):
        raise ValueError("registered observer omits a persistent escape site")
    if (
        proof.get("schema") != "closed_persistent_KV_escape_proof_v1"
        or proof.get("review_artifact_sha256") != hashlib.sha256(raw).hexdigest()
        or proof.get("source_contract") != source_contract()
        or review.get("source_contract") != proof["source_contract"]
        or review.get("observer_source_bindings") != bindings
        or review.get("clauses") != list(CLAUSES)
        or review.get("reviewer_kind") != "explicit_source_review_not_human_approval"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or review.get("all_escape_sites_reviewed") is not True
    ):
        raise ValueError("explicit complete source/schema-bound escape review required")
    return _digest(proof)


def _primitive(value, *, depth=0):
    import torch

    if depth > 32:
        raise ValueError("metadata nesting exceeds bound")
    if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, Enum):
        return
    if isinstance(value, (torch.dtype, torch.device)):
        return
    if type(value) in (tuple, list, set, frozenset):
        for item in value:
            _primitive(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            _primitive(key, depth=depth + 1)
            _primitive(item, depth=depth + 1)
        return
    raise ValueError(f"unregistered nonprimitive metadata/hidden owner: {type(value).__name__}")


def _fields(value, expected):
    actual = set(vars(value)) if hasattr(value, "__dict__") else set()
    if dataclasses.is_dataclass(value):
        actual.update(field.name for field in dataclasses.fields(value))
    if actual != set(expected):
        raise ValueError(
            f"closed schema field drift: {type(value).__name__}: {actual ^ set(expected)}"
        )


def _tensor(value):
    import torch

    if type(value) is not torch.Tensor:
        raise ValueError("exact torch.Tensor owners required; subclasses are unsupported")
    storage = value.untyped_storage()
    return {
        "device": str(value.device),
        "addr": storage.data_ptr(),
        "bytes": storage.nbytes(),
        "storage_cdata": storage._cdata,
    }


def live_witness(tagged_owners, cursor):
    """Keep strong owners through both native cursors; retain metadata ONLY.

    ``tagged_owners`` consists of (tensor, role, source_path) entries returned
    by the schema audit or a reviewed pre-publication escape wrapper.
    """
    strong = tuple(tagged_owners)
    before = cursor()
    if type(before) is not int or before < 0:
        raise ValueError("native integer cursor required")
    rows = {}
    for value, role, source in strong:
        if (
            role not in {"persistent", "cache_workspace"}
            or not isinstance(source, str)
            or not source
        ):
            raise ValueError("explicit registered semantic role and source path required")
        metadata = _tensor(value)
        if not metadata["bytes"]:
            continue
        key = metadata["device"], metadata["addr"]
        row = rows.setdefault(key, dict(metadata, roles=[], source_paths=[]))
        if any(row[field] != metadata[field] for field in ("bytes", "storage_cdata")):
            raise ValueError("inconsistent live backing aliases")
        if role not in row["roles"]:
            row["roles"].append(role)
        if source not in row["source_paths"]:
            row["source_paths"].append(source)
    after = cursor()
    if type(after) is not int or after < before:
        raise ValueError("native cursor went backwards")
    for value, _, _ in strong:
        metadata = _tensor(value)
        if metadata["bytes"] and metadata != {
            key: rows[(metadata["device"], metadata["addr"])][key] for key in metadata
        }:
            raise ValueError("storage changed while capturing its live witness")
    return {
        "begin_cursor": before,
        "end_cursor": after,
        "owners": list(rows.values()),
        "strong_references_released_on_return": True,
    }


def audit_schema(cache, *, states=()):
    """Enumerate the exact supported root graph; reject drift, not ignore it.

    Include BOTH retained prefill and active decode states. A cache-only audit
    is explicitly incomplete for the whole-request metadata boundary.
    """
    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.cache_state import ModalitySpan
    from mosaickv.evaluation.dual_projection import DualFormat
    from mosaickv.evaluation.encoded_projection import EncodedFormat
    from mosaickv.evaluation.exact_storage import ExactFormat, SparseBitPatch
    from mosaickv.evaluation.graphed_storage import SharedSourceCache, SharedSourceLayer
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_source_control import LosslessSourceFormat
    from mosaickv.evaluation.projection_archive import ProjectionArchive
    from mosaickv.evaluation.source_history import SourceHistoryCache, SourceHistoryLayer
    from mosaickv.evaluation.streaming_prefill import RetiringLayer, RetiringPrefillCache

    tagged, visited = [], set()

    def tensors(values, path, role="persistent"):
        if not isinstance(values, (tuple, list)):
            values = (values,)
        for index, value in enumerate(values):
            _tensor(value)
            tagged.append((value, role, f"{path}[{index}]"))

    packet_types = {
        LosslessSourceFormat,
        ExactFormat,
        DualFormat,
        EncodedFormat,
        ProjectionArchive,
        PackedBF16,
        SparseBitPatch,
    }

    def packet(value, path):
        if type(value) not in packet_types:
            raise ValueError("unregistered packet type/subclass")
        if id(value) in visited:
            return
        visited.add(id(value))
        fields = {field.name for field in dataclasses.fields(type(value))}
        _fields(value, fields)
        for name in fields:
            item = getattr(value, name)
            target = f"{path}.{name}"
            if name in {
                "normalized",
                "cos",
                "sin",
                "logical_positions",
                "indices",
                "values",
                "union",
                "selected",
                "slots",
                "output_slots",
                "literal_checkpoints",
            }:
                if item is not None:
                    tensors(item, target)
            elif name in {"archive", "source"}:
                if item is not None:
                    packet(item, target)
            elif name in {"pair", "patches"}:
                if type(item) is not tuple:
                    raise ValueError("immutable exact packet children required")
                for index, child in enumerate(item):
                    packet(child, f"{target}[{index}]")
            elif name == "buffers":
                tensors(item, target)
            else:
                _primitive(item)

    def attention(value):
        import torch
        from transformers.models.llama.modeling_llama import LlamaAttention

        if type(value) is not LlamaAttention or value.training or value.num_key_value_groups != 1:
            raise ValueError("pinned eval/full-MHA LlamaAttention required")
        # This primitive has no authority to approve active hook closures.
        # Caller must audit at a quiescent point, not silently trust a closure.
        internals = {
            "training",
            "_parameters",
            "_buffers",
            "_non_persistent_buffers_set",
            "_backward_pre_hooks",
            "_backward_hooks",
            "_is_full_backward_hook",
            "_forward_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
            "_modules",
        }
        for module in value.modules():
            if any(
                getattr(module, key)
                for key in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                    "_state_dict_hooks",
                    "_state_dict_pre_hooks",
                    "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks",
                )
            ):
                raise ValueError("active hook context requires a separate reviewed escape proof")
            if module is not value and type(module) is not torch.nn.Linear:
                raise ValueError("foreign projection module can retain unknown owners")
            if (
                module._buffers
                or (
                    module is value
                    and (
                        module._parameters
                        or set(module._modules) != {"q_proj", "k_proj", "v_proj", "o_proj"}
                    )
                )
                or (
                    module is not value
                    and (
                        set(module._parameters) != {"weight", "bias"}
                        or module._parameters["bias"] is not None
                        or module._modules
                    )
                )
            ):
                raise ValueError(
                    "unexpected model/projection buffer, parameter or hidden module owner"
                )
            for key, item in vars(module).items():
                if key in internals:
                    continue
                _primitive(vars(item) if key == "config" else item)

    def layer(value, parent, path):
        if type(value) is RetiringLayer:
            _fields(value, {"keys", "values", "owner", "index", "length", "packet"})
            if value.owner is not parent or (value.keys is None) != (value.values is None):
                raise ValueError("retiring original-owner relation changed")
            if value.keys is not None:
                tensors((value.keys, value.values), path + ".original_KV")
            if value.packet is not None:
                packet(value.packet, path + ".packet")
            _primitive(value.index)
            _primitive(value.length)
            return
        if type(value) not in {SourceHistoryLayer, SharedSourceLayer}:
            raise ValueError("unregistered source layer/subclass")
        expected = _LAYER | (
            {"cache", "reconstruction_graph", "graph_private_bytes"}
            if type(value) is SharedSourceLayer
            else set()
        )
        _fields(value, expected)
        if value.keys is not None or value.values is not None or value.sealed:
            raise ValueError("unexpected independent raw/ordinary sealed cache owner")
        if type(value) is SharedSourceLayer and (
            value.cache is not parent
            or value.reconstruction_graph is not None
            or value.graph_private_bytes
        ):
            raise ValueError("graph or foreign shared-cache ownership unsupported")
        if not value.direct or value.eviction_count or value.fused_unpack:
            raise ValueError("only the registered direct exact source-history policy is supported")
        attention(value.attention)
        packet(value.representation, path + ".representation")
        tensors(value.history, path + ".history")
        tensors(value.leased, path + ".leased", "cache_workspace")
        if value.normalized_tail is not None:
            tensors(value.normalized_tail, path + ".normalized_tail")
        for index, item in enumerate(value.source_sealed):
            packet(item, f"{path}.source_sealed[{index}]")
        for name in expected - {
            "representation",
            "attention",
            "history",
            "leased",
            "observe",
            "normalized_tail",
            "source_sealed",
            "cache",
        }:
            _primitive(getattr(value, name))
        if not isinstance(value.observe, types.MethodType) or value.observe.__self__ is not parent:
            raise ValueError("foreign cache observer may retain unenumerated owners")

    if type(cache) not in {RetiringPrefillCache, SourceHistoryCache, SharedSourceCache}:
        raise ValueError("unregistered cache/subclass/offload route")
    expected = (
        _RETIRING
        if type(cache) is RetiringPrefillCache
        else _SHARED_CACHE
        if type(cache) is SharedSourceCache
        else _SOURCE_CACHE
    )
    _fields(cache, expected)
    from transformers.cache_utils import DynamicLayer

    expected_factory = RetiringLayer if type(cache) is RetiringPrefillCache else DynamicLayer
    if cache.layer_classes is not expected_factory:
        raise ValueError("foreign layer allocation factory is outside the closed schema")
    if cache.cache_processor is not None:
        raise ValueError("offload/processor owners are unsupported")
    if type(cache) is SharedSourceCache and (
        cache.graphs_enabled
        or cache.graph_private_live_bytes
        or cache.graph_captures
        or cache.graph_replays
    ):
        raise ValueError("graph ownership is outside this closed schema")
    if type(cache) is RetiringPrefillCache:
        if cache.extra_tensors or any(
            type(ref) is not weakref.ReferenceType for ref in cache.original_refs
        ):
            raise ValueError("unexpected extra prefill owner or strong reference")
        metadata = expected - {"layers", "layer_classes", "original_refs"}
    else:
        if cache.source_constructor != "direct" or cache.consolidated_phase_versions:
            raise ValueError("unregistered history constructor/consolidation")
        persistent = []
        if cache.phase_tail is not None:
            persistent.extend(cache.phase_tail)
        persistent.extend(cache.history_position_chunks)
        persistent.extend(t for pair in cache.history_rotary_chunks for t in pair)
        tensors(persistent, "cache.history_metadata")
        workspace = cache.workspace if type(cache) is SharedSourceCache else ()
        tensors(workspace, "cache.workspace", "cache_workspace")
        # Resolve the mixed shared list by exact backing identities of known
        # source fields, not by container name or a broad workspace exclusion.
        for index, item in enumerate(cache.layers):
            layer(item, cache, f"cache.layers[{index}]")
        classified = {_key(_tensor(value)) for value, _, _ in tagged}
        for value in cache.shared_tensors:
            if _key(_tensor(value)) not in classified:
                raise ValueError("shared cache contains an unexplained persistent/lease owner")
        metadata = expected - {
            "layers",
            "layer_classes",
            "shared_tensors",
            "phase_tail",
            "history_position_chunks",
            "history_rotary_chunks",
            "workspace",
        }
    if type(cache) is RetiringPrefillCache:
        for index, item in enumerate(cache.layers):
            layer(item, cache, f"cache.layers[{index}]")
    for name in metadata:
        _primitive(getattr(cache, name))
    for index, state in enumerate(states):
        if type(state) is not DecodeState:
            raise ValueError("exact explicit prefill/decode state roots required")
        _fields(state, {field.name for field in dataclasses.fields(DecodeState)})
        if state.past_key_values is not None and state.past_key_values is not cache:
            raise ValueError("unregistered retained cache state")
        tensors(state.attention_mask, f"states[{index}].attention_mask")
        if type(state.model_state) is not dict or set(state.model_state) - (
            _STATE_TENSORS | _STATE_METADATA
        ):
            raise ValueError("unregistered DecodeState.model_state key/owner")
        for name, value in state.model_state.items():
            if name in _STATE_TENSORS:
                tensors(value, f"states[{index}].model_state.{name}")
            else:
                _primitive(value)
        for name in ("active_cache_length", "logical_sequence_length", "next_decode_position"):
            _primitive(getattr(state, name))
        for span in state.modality_map:
            if type(span) is not ModalitySpan:
                raise ValueError("unregistered modality metadata")
            for field in dataclasses.fields(span):
                _primitive(getattr(span, field.name))
    return {
        "tagged_owners": tuple(tagged),
        "schema_sha256": source_contract()["schema_sha256"],
        "state_root_count": len(states),
        "whole_request_escape_review_passed": False,
        "missing_state_roots": not bool(states),
        "metric_B_classified": False,
    }


def _key(owner):
    return owner["device"], owner["addr"], owner["bytes"]


def bind_witnesses(records, witnesses):
    """Resolve full backing across BOTH cursors; pointer reuse stays distinct."""
    chosen = set()
    for witness in witnesses:
        begin, end = witness["begin_cursor"], witness["end_cursor"]
        if type(begin) is not int or type(end) is not int or not 0 <= begin <= end:
            raise ValueError("invalid live cursor range")
        for owner in witness["owners"]:
            if "persistent" not in owner["roles"]:
                continue
            device = owner["device"]
            if (
                not isinstance(device, str)
                or not device.startswith("cuda:")
                or not device[5:].isdigit()
            ):
                raise ValueError("only explicit indexed CUDA backings can bind native generations")
            matches = [
                i
                for i, row in enumerate(records)
                if row["device"] == int(device[5:])
                and row["addr"] == owner["addr"]
                and row["size"] == owner["bytes"]
                and row["start"] < begin
                and (row["end"] is None or row["end"] >= end)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "live witness does not identify exactly one complete native generation"
                )
            chosen.add(matches[0])
    return chosen


def reduce_metric_A(
    capture,
    *,
    baseline,
    final_blocks,
    witnesses,
    proof,
    review_artifact,
    complete_escape_coverage=False,
):
    """Conditional arithmetic, NEVER a runtime ownership certificate.

    Runtime publisher/hook completeness is not implemented by this primitive.
    A boolean and a reviewed JSON cannot turn that missing evidence into a
    production certificate. Unknown B is NEVER converted into non-KV.
    """
    proof_sha = validate_escape_proof(proof, review_artifact)
    if complete_escape_coverage is not True:
        raise ValueError("all registered escape sites and retained state roots must be covered")
    if (
        capture.get("schema") != "native_numeric_allocator_trace_v1"
        or capture.get("synthetic_test_only") is not False
    ):
        raise ValueError("actual complete native capture required, not synthetic injection")
    ledger = NativeEventLedger(baseline).ingest(capture)
    final = ledger.verify_snapshot(final_blocks)
    selected = bind_witnesses(ledger.records, witnesses)
    if not selected or any(ledger.records[index]["baseline"] for index in selected):
        raise ValueError(
            "fresh-empty-cache review contradicts missing/preexisting persistent owners"
        )
    changes = {}
    for index in selected:
        row = ledger.records[index]
        changes[row["start"]] = changes.get(row["start"], 0) + row["size"]
        if row["end"] is not None:
            changes[row["end"]] = changes.get(row["end"], 0) - row["size"]
    current = peak = 0
    for sequence in sorted(changes):
        current += changes[sequence]
        peak = max(peak, current)
    return {
        "boundary": BOUNDARY,
        "exact_peak_certified": False,
        "exact_source_KV_peak_certified": False,
        "exact_whole_request_KV_peak_certified": False,
        "conditional_exact_A_under_declared_reviewed_escape_assumptions": True,
        "conditional_metric_A": {"peak_requested_bytes": peak, "final_requested_bytes": current},
        "persistent_generations": len(selected),
        "other_generations": len(ledger.records) - len(selected),
        "escape_proof_sha256": proof_sha,
        "conditional_unmarked_not_A_requires_runtime_exhaustive_escape_proof": True,
        "runtime_publisher_and_every_mutation_evidence_wired": False,
        "actual_unknown_generations_remain_unclassified": True,
        "metric_B_unknown_preserved": True,
        "whole_device_or_reserved_memory_certified": False,
        "native_final_snapshot": final,
        "quality_latency_or_superiority_promoted": False,
    }
