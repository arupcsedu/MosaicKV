"""Consumed-only causal control: exact delimiter restoration at fixed capacity.

No production selector changes. Donors use only raw prefill-window attention
times source-value norm, identically for Top-K and EviCastKV. Future logits and
annotation answers are unavailable to the transaction.
"""

import hashlib
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

from mosaickv.evaluation.early_prefix import retained_positions


def delimiter_positions(token_ids, tokenizer):
    ids = tuple(int(x) for x in token_ids)
    for count in range(1, min(16, len(ids)) + 1):
        if tokenizer.decode(list(ids[-count:])) == "ASSISTANT:":
            return tuple(range(len(ids) - count, len(ids)))
    raise ValueError("complete generation delimiter cannot be identified exactly")


def compare_prefix_logits(reference, anchor, candidate):
    import torch

    if (
        reference.shape != anchor.shape
        or reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[0] != 32
    ):
        raise ValueError("complete paired 32-prefix logits required")
    if not all(bool(torch.isfinite(t).all()) for t in (reference, anchor, candidate)):
        raise ValueError("nonfinite causal replay logits")
    records = []
    for step, (full, a, b) in enumerate(zip(reference, anchor, candidate, strict=True), 1):
        logp = full.float().log_softmax(-1)
        p = logp.exp()
        fp64 = full.double().log_softmax(-1)
        row = dict(step=step, reference_argmax_tie_count=int((full == full.max()).sum()))
        for name, value in (("anchor", a), ("candidate", b)):
            logq = value.float().log_softmax(-1)
            row[name] = dict(
                logit_kl=float((p * (logp - logq)).sum()),
                maximum_logit_difference=float((full.float() - value.float()).abs().max()),
                greedy_token_agreement=bool(full.argmax() == value.argmax()),
                logit_kl_fp64_shadow=float(
                    (fp64.exp() * (fp64 - value.double().log_softmax(-1))).sum()
                ),
            )
        records.append(row)
    return records


def restore_positions(selected, delimiter, mandatory, donor_scores):
    """Deterministic equal-count replacement in one layer/KV-head route."""
    import math

    selected, delimiter, mandatory = tuple(selected), tuple(delimiter), frozenset(mandatory)
    if tuple(sorted(set(selected))) != selected or not mandatory.issubset(selected):
        raise ValueError("sorted exact anchor preserving all mandatory positions required")
    if not delimiter or len(set(delimiter)) != len(delimiter):
        raise ValueError("complete unique delimiter positions required")
    if any(
        type(p) is not int or not 0 <= p < len(donor_scores)
        for p in (*selected, *delimiter, *mandatory)
    ):
        raise ValueError("source position outside donor-score domain")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in donor_scores):
        raise ValueError("finite nonnegative prompt-only donor scores required")
    added = tuple(sorted(set(delimiter) - set(selected)))
    pool = sorted(
        set(selected) - set(delimiter) - mandatory, key=lambda p: (float(donor_scores[p]), p)
    )
    if len(pool) < len(added):
        raise ValueError("delimiter plus mandatory positions cannot fit existing route capacity")
    removed = tuple(pool[: len(added)])
    result = tuple(sorted((set(selected) - set(removed)) | set(added)))
    if (
        len(result) != len(selected)
        or not mandatory.issubset(result)
        or not set(delimiter).issubset(result)
    ):
        raise RuntimeError("equal-capacity delimiter transaction invariant failed")
    return result, dict(
        added_positions=list(added),
        removed_positions=list(removed),
        removed_prompt_impact_scores=[float(donor_scores[p]) for p in removed],
        restored_prompt_impact_scores=[float(donor_scores[p]) for p in added],
    )


def source_sha(pairs):
    """Bitwise source-state check, deliberately excluded from latency claims."""
    import torch

    digest = hashlib.sha256()
    for key, value in pairs:
        for tensor in (key, value):
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def build_control(adapter, state, anchor, attention, delimiter, mandatory, *, restore):
    import torch

    from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
    from mosaickv.backends.hf_allocation_guard import packed_positions_sha
    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    if not anchor.all_slots_valid or anchor.prototype_slots or anchor.layerwise_logical_positions:
        raise ValueError("control requires unpadded exact per-head prompt mapping")
    before = retained_positions(anchor)
    original, _ = adapter._legacy_layers(state.past_key_values)
    original_sha = source_sha(original)
    if len(attention) != len(original):
        raise ValueError("complete raw prompt-attention capture required")
    positions, events = [], []
    for layer, ((key, value), heads, probabilities) in enumerate(
        zip(original, before, attention, strict=True)
    ):
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("audited batch-one matching BHSD K/V required")
        batch, query_heads, queries, length = probabilities.shape
        kv_heads = key.shape[1]
        if batch != 1 or query_heads % kv_heads or length != key.shape[2] or queries < 1:
            raise ValueError("raw prompt-attention geometry differs from source")
        mass = (
            probabilities.float()
            .reshape(1, kv_heads, query_heads // kv_heads, queries, length)
            .mean((0, 2, 3))
        )
        scores = (mass * value[0].float().norm(dim=-1)).detach().cpu().tolist()
        rows = []
        for head, selected in enumerate(heads):
            if restore:
                row, event = restore_positions(selected, delimiter, mandatory, scores[head])
                if event["added_positions"]:
                    cost = (
                        key.shape[-1] * key.element_size() + value.shape[-1] * value.element_size()
                    )
                    events.append(
                        dict(
                            layer=layer,
                            kv_head=head,
                            position_bytes=cost,
                            replaced_bytes=cost * len(event["added_positions"]),
                            **event,
                        )
                    )
            else:
                row = tuple(selected)
            rows.append(row)
        positions.append(tuple(rows))
    snapshot = gather_layer_local_snapshot(adapter, state.past_key_values, tuple(positions))
    records = tuple(
        dict(layer=layer, kv_head=h, slot=s, logical_position=p, physical_position=p, tier="exact")
        for layer, heads in enumerate(positions)
        for h, row in enumerate(heads)
        for s, p in enumerate(row)
    )
    result = replace(anchor, snapshot=snapshot, slot_records=records)
    old_bytes = CacheLifecycle.count(tuple((x.key, x.value) for x in anchor.snapshot.layers))[1]
    new_bytes = CacheLifecycle.count(tuple((x.key, x.value) for x in result.snapshot.layers))[1]
    if old_bytes != new_bytes or any(
        a.key.shape != b.key.shape
        for a, b in zip(anchor.snapshot.layers, snapshot.layers, strict=True)
    ):
        raise RuntimeError("delimiter transaction changed physical capacity")
    if not restore and any(
        not torch.equal(a.key, b.key) or not torch.equal(a.value, b.value)
        for a, b in zip(anchor.snapshot.layers, snapshot.layers, strict=True)
    ):
        raise RuntimeError("null repacking changed exact K/V values")
    if source_sha(original) != original_sha:
        raise RuntimeError("delimiter transaction mutated original prefill K/V")
    trace = dict(
        mode="restore_delimiter" if restore else "null_repack",
        delimiter_positions=list(delimiter),
        mandatory_positions=list(mandatory),
        events=events,
        changed_routes=len(events),
        total_replaced_bytes=sum(e["replaced_bytes"] for e in events),
        before_positions_sha256=packed_positions_sha(anchor),
        after_positions_sha256=packed_positions_sha(result),
        source_KV_sha256=original_sha,
        source_KV_unchanged=True,
        original_KV_values_and_positions_preserved=True,
        initial_physical_bytes=old_bytes,
        constructed_physical_bytes=new_bytes,
        padding_bytes=0,
        donor_evidence="raw_prompt_window_attention_mean_times_source_value_L2_norm",
        future_queries_or_logits_used=False,
        annotation_targets_used=False,
        runtime_selector_trace_describes_anchor_not_intervention=True,
        effective_selection_injected_only_through_scoped_evaluation_transaction=True,
    )
    return result, trace


class DelimiterTransaction:
    def __init__(
        self, adapter, runtime_module, *, mode, prompt_window=16, mandatory_policy="method"
    ):
        if mode not in {"anchor", "null_repack", "restore_delimiter"} or mandatory_policy not in {
            "method",
            "last_token",
        }:
            raise ValueError("bounded control mode and audited exact mandatory policy required")
        self.adapter, self.module, self.mode = adapter, runtime_module, mode
        self.prompt_window = prompt_window
        self.stack = ExitStack()
        self.attention = None
        self.delimiter = None
        self.mandatory = None
        self.record = None

    def __enter__(self):
        original_prefill = self.adapter.prefill
        original_pack = self.module._state_from_packed

        def prefill(prepared, *args, **kwargs):
            if self.attention is not None:
                raise RuntimeError("one original prefill required")
            ids = prepared.model_inputs["input_ids"][0].tolist()
            self.delimiter = delimiter_positions(ids, self.adapter.processor.tokenizer)
            self.mandatory = (len(ids) - 1,)
            if any(
                not any(
                    s.start <= p < s.end and str(s.modality) == "text"
                    for s in prepared.modality_map
                )
                for p in self.delimiter
            ):
                raise ValueError("generation delimiter must lie wholly in source text")
            with self.adapter.capture_attention_weights(query_window=self.prompt_window) as capture:
                output = original_prefill(prepared, *args, **kwargs)
            self.attention = capture.result()
            return output

        def packed(adapter, state, choice):
            if self.record is not None or self.attention is None:
                raise RuntimeError("one fixed prompt transaction after prefill required")
            if self.mode == "anchor":
                self.record = dict(
                    mode="anchor",
                    delimiter_positions=list(self.delimiter),
                    events=[],
                    changed_routes=0,
                )
                result = choice
            else:
                result, self.record = build_control(
                    adapter,
                    state,
                    choice,
                    self.attention,
                    self.delimiter,
                    self.mandatory,
                    restore=self.mode == "restore_delimiter",
                )
            return original_pack(adapter, state, result)

        self.stack.enter_context(patch.object(self.adapter, "prefill", prefill))
        self.stack.enter_context(patch.object(self.module, "_state_from_packed", packed))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
