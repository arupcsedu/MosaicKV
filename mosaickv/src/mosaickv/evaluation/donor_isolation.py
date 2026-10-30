"""Consumed-input donor-availability ablation, NOT physical cache compression.

Reuse the saved equal-capacity transaction verbatim. Block its donor attention
without restoring anything or changing dense K/V storage. No online policy is
changed, and no model/annotation evidence is used to choose new donors.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from types import TracebackType
from typing import Any
from unittest.mock import patch

from mosaickv.evaluation.early_prefix import retained_positions


def donor_slots(
    positions: Sequence[Sequence[Sequence[int]]], transaction: dict[str, Any]
) -> list[list[list[int]]]:
    """Validate a frozen transaction and map original donors to physical slots."""
    mandatory = set(transaction["mandatory_positions"])
    delimiter = set(transaction["delimiter_positions"])
    slots: list[list[list[int]]] = [[[] for _ in heads] for heads in positions]
    seen: set[tuple[int, int]] = set()
    total = 0
    for event in transaction["events"]:
        layer, head = event["layer"], event["kv_head"]
        if (
            type(layer) is not int
            or type(head) is not int
            or not 0 <= layer < len(positions)
            or not 0 <= head < len(positions[layer])
            or (layer, head) in seen
        ):
            raise ValueError("invalid or repeated saved donor route")
        seen.add((layer, head))
        row = positions[layer][head]
        if len(row) != len(set(row)) or not mandatory.issubset(row):
            raise ValueError("unique mandatory-preserving anchor required")
        removed, added = event["removed_positions"], event["added_positions"]
        if (
            not removed
            or len(removed) != len(added)
            or len(set(removed)) != len(removed)
            or len(set(added)) != len(added)
            or not set(removed).issubset(row)
            or set(removed) & (mandatory | delimiter)
            or set(added) & set(row)
            or set(added) != delimiter - set(row)
            or len(removed) >= len(row)
            or event["position_bytes"] <= 0
            or event["replaced_bytes"] != len(removed) * event["position_bytes"]
        ):
            raise ValueError("saved donors/restores must be feasible, equal-byte and unchanged")
        slots[layer][head] = [row.index(p) for p in removed]
        total += event["replaced_bytes"]
    if total != transaction["total_replaced_bytes"] or len(seen) != transaction["changed_routes"]:
        raise ValueError("saved transaction totals differ")
    return slots


def make_masks(
    original: Sequence[Any],
    slots: Sequence[Sequence[Sequence[int]]],
    query_heads: Sequence[int],
    *,
    block: bool,
) -> tuple[Any, ...]:
    """Head-aware additive masks; GQA shares each donor across its query group."""
    import torch

    if len(original) != len(slots) or len(original) != len(query_heads):
        raise ValueError("complete layer masks required")
    result = []
    for base, heads, count in zip(original, slots, query_heads, strict=True):
        if (
            tuple(base.shape[:3]) != (1, 1, 1)
            or base.ndim != 4
            or not heads
            or count % len(heads)
            or count < len(heads)
            or not base.dtype.is_floating_point
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError("audited batch-one bounded masks and valid GQA required")
        mask = base.expand(1, count, 1, -1).clone()
        for head, indices in enumerate(heads):
            if len(indices) != len(set(indices)) or any(
                type(i) is not int or not 0 <= i < base.shape[-1] for i in indices
            ):
                raise ValueError("donor physical slot outside mask")
            if block:
                group = count // len(heads)
                mask[:, head * group : (head + 1) * group, :, indices] = -torch.inf
        result.append(mask)
    return tuple(result)


class DonorMaskControl:
    """Scoped mask-only diagnostic; preserves selected tensors and logical IDs."""

    def __init__(
        self, adapter: Any, runtime_module: Any, transaction: dict[str, Any], *, mode: str
    ) -> None:
        if mode not in {"zero_mask", "donor_only"}:
            raise ValueError("zero-mask or frozen donor-only control required")
        self.adapter, self.module, self.transaction, self.mode = (
            adapter,
            runtime_module,
            transaction,
            mode,
        )
        self.stack = ExitStack()
        self.slots: list[list[list[int]]] | None = None
        self.positions: list[list[list[int]]] | None = None
        self.prompt_sha: str | None = None
        self.banks: tuple[Any, ...] | None = None
        self.records: list[dict[str, Any]] = []
        self.mask_calls = 0
        self.additional_mask_backing_bytes = 0

    def __enter__(self) -> DonorMaskControl:
        import torch

        import mosaickv.adapters.huggingface.bounded_history_cache as bounded
        from mosaickv.backends.hf_allocation_guard import packed_positions_sha
        from mosaickv.evaluation.delimiter_control import source_sha

        original_pack, original_decode = (
            self.module._state_from_packed,
            self.adapter.decode_one_token,
        )
        original_masks = bounded.bounded_history_masks

        def packed(adapter: Any, state: Any, choice: Any) -> Any:
            if self.positions is not None or not choice.all_slots_valid or choice.prototype_slots:
                raise ValueError("one unpadded exact anchor required")
            self.positions = retained_positions(choice)
            self.slots = donor_slots(self.positions, self.transaction)
            if packed_positions_sha(choice) != self.transaction["before_positions_sha256"]:
                raise ValueError("anchor positions differ from saved donor transaction")
            source, _ = adapter._legacy_layers(state.past_key_values)
            if source_sha(source) != self.transaction["source_KV_sha256"]:
                raise ValueError("original source K/V differs from saved transaction")
            self.prompt_sha = source_sha(tuple((x.key, x.value) for x in choice.snapshot.layers))
            return original_pack(adapter, state, choice)

        def masks(state: Any) -> tuple[Any, ...]:
            if self.slots is None or not state.model_state.get("mosaickv_incremental_history"):
                raise ValueError("frozen packed prompt with incremental history required")
            baseline = original_masks(state)
            if self.banks is None:
                full = state.model_state["mosaickv_bounded_history_masks"]
                heads = [
                    int(x.self_attn.config.num_attention_heads)
                    for x in self.adapter._language_layers()
                ]
                self.banks = make_masks(full, self.slots, heads, block=self.mode == "donor_only")
                self.additional_mask_backing_bytes = sum(
                    m.untyped_storage().nbytes() for m in self.banks
                )
            # Original buffers must remain all-valid throughout this specific
            # no-history-eviction replay. Never silently override another mask.
            if any(bool((m != 0).any()) for m in baseline):
                raise ValueError("unexpected nonzero underlying bounded mask")
            self.mask_calls += 1
            return tuple(
                bank[..., : old.shape[-1]] for bank, old in zip(self.banks, baseline, strict=True)
            )

        def decode(token: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
            if self.positions is None or self.slots is None:
                raise ValueError("packed prompt mapping missing before decode")
            kwargs["capture_attentions"] = True  # identical in zero/donor arms
            output = original_decode(token, state, *args, **kwargs)
            if len(output.attention_weights) != len(self.positions):
                raise ValueError("complete actual attention capture required")
            mass, donor_max, generated_mass = 0.0, 0.0, 0.0
            for probabilities, heads, selected in zip(
                output.attention_weights, self.slots, self.positions, strict=True
            ):
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("nonfinite causal attention")
                count = probabilities.shape[1]
                if count % len(heads):
                    raise ValueError("actual GQA differs from audited mask")
                group = count // len(heads)
                for head, indices in enumerate(heads):
                    if indices:
                        p = probabilities[0, head * group : (head + 1) * group, 0, indices]
                        mass += float(p.float().sum())
                        donor_max = max(donor_max, float(p.float().max()))
                prompt_length = len(selected[0])
                if any(len(row) != prompt_length for row in selected):
                    raise ValueError("ragged heads not supported by this dense diagnostic")
                generated_mass += float(probabilities[..., prompt_length:].float().sum())
                if any(i >= prompt_length for row in heads for i in row):
                    raise ValueError("donor mask reached generated history")
            if self.mode == "donor_only" and donor_max != 0:
                raise ValueError("masked donor still receives attention")
            if any(layer.eviction_count for layer in output.state.past_key_values.layers):
                raise ValueError("history eviction confounds donor-only control")
            prompt = tuple(
                (layer.keys[..., : len(rows[0]), :], layer.values[..., : len(rows[0]), :])
                for layer, rows in zip(
                    output.state.past_key_values.layers, self.positions, strict=True
                )
            )
            # Hash once at the final step: masks must not alter original prompt K/V.
            if len(self.records) == 30 and source_sha(prompt) != self.prompt_sha:
                raise ValueError("donor-availability ablation changed prompt K/V")
            self.records.append(
                dict(
                    prediction=len(self.records) + 2,
                    donor_attention_mass_sum_over_layers_and_query_heads=mass,
                    maximum_donor_attention_probability=donor_max,
                    generated_attention_mass_sum_over_layers_and_query_heads=generated_mass,
                )
            )
            return output

        self.stack.enter_context(patch.object(self.module, "_state_from_packed", packed))
        self.stack.enter_context(patch.object(bounded, "bounded_history_masks", masks))
        self.stack.enter_context(patch.object(self.adapter, "decode_one_token", decode))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.stack.__exit__(exc_type, exc_value, traceback)

    def validate_complete(self, horizon: int = 32) -> None:
        if len(self.records) != horizon - 1 or self.mask_calls != horizon - 1:
            raise ValueError("incomplete donor-only mask/attention trace")

    def record(self) -> dict[str, Any]:
        return dict(
            mode=self.mode,
            availability_only_not_physical_eviction=True,
            physical_KV_bytes_saved=0,
            restore_positions_added=[],
            frozen_donor_events=self.transaction["events"],
            frozen_masked_donor_payload_bytes=self.transaction["total_replaced_bytes"]
            if self.mode == "donor_only"
            else 0,
            additional_mask_backing_bytes=self.additional_mask_backing_bytes,
            prompt_KV_sha256=self.prompt_sha,
            original_KV_and_logical_positions_unchanged=True,
            generated_positions_never_masked=True,
            mandatory_positions_never_masked=True,
            attention=self.records,
            no_annotation_or_future_logits_used_for_construction=True,
        )
