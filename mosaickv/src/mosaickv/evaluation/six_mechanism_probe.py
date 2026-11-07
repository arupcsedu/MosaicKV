"""Diagnostic-only observers; wrap rather than replace the six-component runtime."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch


def tensor_sha(value: Any) -> str:
    import torch

    data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@contextmanager
def observe_runtime(adapter: Any, config: Any) -> Iterator[dict[str, Any]]:
    """Capture executed work and independent tensor/layout checks, not benchmark labels.

    These observers synchronize and hash values. Their timings are diagnostic,
    not uninstrumented performance claims. No result tensor is changed.
    """
    import torch

    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    counts: Counter[str] = Counter()
    record: dict[str, Any] = {
        "decode_checks": [],
        "rope_position_checks": [],
        "decode_logits_sha256": [],
        "prefill_logits_sha256": [],
        "packing_records": [],
        "plan_invariants": False,
        "packed_storage_checks": False,
        "repair_physical_cap_checks": True,
        "promoted_kv_exact_checks": [],
        "benchmark_labels_accessed": False,
        "instrumented_diagnostic_not_headline_latency": True,
    }
    original_prefill = adapter.prefill
    original_decode = adapter.decode_one_token
    original_plan = runtime.build_compression_plan
    original_pack = runtime.pack_runtime_cache
    expected_rope_position: list[int | None] = [None]

    def rotary_hook(_module: Any, args: Any, kwargs: Any) -> None:
        expected_position = expected_rope_position[0]
        if expected_position is None:
            return
        positions = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        valid = positions is not None and positions.detach().reshape(-1).tolist() == [
            expected_position
        ]
        record["rope_position_checks"].append(valid)
        assert valid, "actual model RoPE position differs from original logical next position"

    def prefill(*args: Any, **kwargs: Any) -> Any:
        counts["prefill"] += 1
        if kwargs.get("capture_attentions"):
            # Registered correctness-first collection: retain native full
            # eager probabilities, not an optimized query-window kernel.
            kwargs["attention_query_window"] = None
            record["prefill_attention_capture"] = "full_original_eager"
        output = original_prefill(*args, **kwargs)
        record["prefill_logits_sha256"].append(tensor_sha(output.logits))
        return output

    def decode(token: Any, state: Any, **kwargs: Any) -> Any:
        logical, next_position, active = (
            state.logical_sequence_length,
            state.next_decode_position,
            state.active_cache_length,
        )
        expected_rope_position[0] = next_position
        try:
            output = original_decode(token, state, **kwargs)
        finally:
            expected_rope_position[0] = None
        record["decode_checks"].append(
            bool(
                output.state.logical_sequence_length == logical + 1
                and output.state.next_decode_position == next_position + 1
                and output.state.active_cache_length == active + 1
                and torch.isfinite(output.logits).all().item()
            )
        )
        record["decode_logits_sha256"].append(tensor_sha(output.logits))
        return output

    def plan(*args: Any, **kwargs: Any) -> Any:
        output = original_plan(*args, **kwargs)
        full = output.full_state
        exact = set(output.construction.exact_node_ids)
        required = {i for i, block in enumerate(full.blocks) if block.mandatory}
        state = output.construction.state
        physical_source = full.logical_positions
        assert state.logical_positions == physical_source
        assert required <= exact, "mandatory source membership was removed"
        bank = getattr(state.exact, "_compact_bank", None)
        exact_bytes = (
            sum(
                (bank.keys[block.layer].position_bytes + bank.values[block.layer].position_bytes)
                * block.position_count
                for block in state.exact.blocks
            )
            if bank is not None
            else sum(
                int(t.numel()) * int(t.element_size())
                for t in (*state.exact.key_blocks, *state.exact.value_blocks)
            )
        )
        actual = exact_bytes + sum(
            int(t.numel()) * int(t.element_size())
            for t in (
                *state.prototypes.prototype_keys,
                *state.prototypes.prototype_values,
            )
        )
        assert actual == state.statistics.active_kv_bytes
        assert actual <= output.active_budget_value
        assert output.selection.budget.unit.value == "bytes"
        counts["prototype_members"] += sum(len(p.members) for p in output.construction.prototypes)
        record["plan_invariants"] = True
        record["planner_nodes"] = len(output.graph.nodes)
        record["source_nodes"] = len(full.blocks)
        record["routed_selected_source_ids"] = sorted(
            output.source_node_id(i) for i in output.selection.selected_node_ids
        )
        record["selected_source_sha256"] = hashlib.sha256(
            str(tuple(sorted(exact))).encode()
        ).hexdigest()
        record["prototype_membership"] = [
            dict(
                id=p.prototype_id,
                layer=p.layer,
                head=p.kv_head,
                source_ids=list(p.assigned_node_ids),
                weights=[m.normalized_weight for m in p.members],
            )
            for p in output.construction.prototypes
        ]
        record["source_logical_sequence_length"] = full.original_logical_sequence_length
        return output

    def pack(*args: Any, **kwargs: Any) -> Any:
        output = original_pack(*args, **kwargs)
        full = args[1]
        layers = tuple((layer.key, layer.value) for layer in output.snapshot.layers)
        payload, backing = CacheLifecycle.count(layers)
        expected = sum(int(t.numel()) * int(t.element_size()) for pair in layers for t in pair)
        assert payload == expected and backing >= payload
        bytes_per_position = full.active_bytes // full.original_logical_sequence_length
        prompt_bytes = output.prompt_capacity * bytes_per_position
        positions: dict[tuple[int, int], list[int]] = defaultdict(list)
        exact_positions: dict[tuple[int, int], set[int]] = defaultdict(set)
        for row in output.slot_records:
            layer, head, position = row["layer"], row["kv_head"], row["logical_position"]
            assert isinstance(layer, int) and isinstance(head, int) and isinstance(position, int)
            identity = layer, head
            positions[identity].append(position)
            if row["tier"] == "exact":
                exact_positions[identity].add(position)
        for identity, indices in positions.items():
            assert indices == sorted(indices)
            assert min(indices) >= 0 and max(indices) < full.original_logical_sequence_length
            assert set(full.mandatory_logical_positions) <= exact_positions[identity]
        if full.mandatory_logical_positions == (0, full.original_logical_sequence_length - 1):
            for original_layer, packed_layer in zip(
                full.layers, output.snapshot.layers, strict=True
            ):
                for original, packed_tensor in (
                    (original_layer.key, packed_layer.key),
                    (original_layer.value, packed_layer.value),
                ):
                    assert torch.equal(original[..., 0, :], packed_tensor[..., 0, :])
                    assert torch.equal(
                        original[..., -1, :], packed_tensor[..., output.prompt_capacity - 1, :]
                    )
        cap_ok = prompt_bytes <= config.cache.budget_value
        record["repair_physical_cap_checks"] &= cap_ok
        assert cap_ok, f"physical prompt cap exceeded: {prompt_bytes} > {config.cache.budget_value}"
        record["packed_storage_checks"] = True
        record["packing_records"].append(
            dict(
                payload_bytes=payload,
                backing_bytes=backing,
                prompt_payload_bytes=prompt_bytes,
                prompt_capacity=output.prompt_capacity,
                active_sequence_length=output.snapshot.active_sequence_length,
                all_slots_valid=output.all_slots_valid,
                prototype_slots=len(output.prototype_slots),
            )
        )
        return output

    def observe_call(name: str, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            result = original(*args, **kwargs)
            if name == "graph":
                counts["cross_modal_edges"] += sum(
                    result.nodes[s].block.modality != result.nodes[t].block.modality
                    for s, t in zip(result.row_indices, result.column_indices, strict=True)
                )
            if name == "repair" and result.event.triggered:
                full = result.state.full_state
                expected = full.gather_exact_blocks(
                    tuple(full.blocks[node_id] for node_id in result.event.restored_block_ids),
                    gather_mode="per_head_reference",
                )
                exact = result.state.mosaic_state.exact
                offsets = {
                    (block.layer, block.kv_head, block.physical_cache_indices): index
                    for index, block in enumerate(exact.blocks)
                }
                for block, key, value in zip(
                    expected.blocks, expected.key_blocks, expected.value_blocks, strict=True
                ):
                    offset = offsets[(block.layer, block.kv_head, block.physical_cache_indices)]
                    restored_key, restored_value = (
                        exact.key_blocks[offset],
                        exact.value_blocks[offset],
                    )
                    valid = torch.equal(key, restored_key) and torch.equal(value, restored_value)
                    record["promoted_kv_exact_checks"].append(valid)
                    assert valid, "promoted GPU K/V differ from original source entries"
            return result

        return wrapped

    with ExitStack() as stack:
        language = adapter.get_profiling_modules().language_model
        rotary = next(
            (
                getattr(model, "rotary_emb", None)
                for model in (language, getattr(language, "model", None))
                if getattr(model, "rotary_emb", None) is not None
            ),
            None,
        )
        if rotary is None:
            raise RuntimeError("registered LLaVA rotary module unavailable to position audit")
        handle = rotary.register_forward_pre_hook(rotary_hook, with_kwargs=True)
        stack.callback(handle.remove)
        for _name, attr, replacement in (
            ("prefill", "prefill", prefill),
            ("decode", "decode_one_token", decode),
        ):
            stack.enter_context(patch.object(adapter, attr, replacement))
        stack.enter_context(
            patch.object(runtime, "build_compression_plan", observe_call("planner", plan))
        )
        stack.enter_context(patch.object(runtime, "pack_runtime_cache", pack))
        for name, attr in (
            ("forecast", "forecast_with_rollout"),
            ("graph", "build_evidence_graph"),
            ("utility", "compute_block_utilities"),
            ("selector", "lazy_greedy_select"),
            ("repair", "repair_decode_step"),
        ):
            stack.enter_context(
                patch.object(runtime, attr, observe_call(name, getattr(runtime, attr)))
            )
        try:
            yield record
        finally:
            record.update(counts)
