"""Evaluation-only, scoped per-token NLL/eviction capture; never a selector input.

Single-request eager decoding only. No tensor/cache clones, model-weight edits,
or changed eviction decisions. Repeated CE and scalar transfers contaminate
timing, so captured requests are not systems benchmarks.
"""

from contextlib import ExitStack
from math import isfinite
from unittest.mock import patch


class TeacherTokenCapture:
    def __init__(self, adapter, target_ids, runtime_module):
        if not target_ids or any(type(x) is not int or x < 0 for x in target_ids):
            raise ValueError("nonempty teacher target token IDs required")
        self.adapter = adapter
        self.target_ids = tuple(target_ids)
        self.module = runtime_module
        self.records = []
        self.native_events = []
        self.source_length = None
        self.prefills = 0
        self._native_prompt = []
        self._native_generated = []
        self._stack = ExitStack()

    def _score(self, logits, state, *, incremental_evictions, removed):
        import torch

        index = len(self.records)
        if index >= len(self.target_ids):
            raise RuntimeError("unexpected extra probe/decode outside teacher trajectory")
        x = logits.reshape(-1, logits.shape[-1])[-1].float()
        target = self.target_ids[index]
        nll = float(
            torch.nn.functional.cross_entropy(
                x.reshape(1, -1),
                torch.tensor([target], dtype=torch.long, device=x.device),
                reduction="sum",
            ).item()
        )
        if not isfinite(nll):
            raise RuntimeError("nonfinite diagnostic NLL")
        counts = [a + b for a, b in zip(incremental_evictions, self._native_generated, strict=True)]
        self.records.append(
            dict(
                prediction_position=index + 1,
                target_token_id=target,
                greedy_token_id=int(x.argmax().item()),
                nll=nll,
                decode_updates=index,
                next_logical_position_after_forward=int(state.next_decode_position),
                generated_evictions_visible_per_layer=counts,
                prompt_evictions_visible_per_layer=list(self._native_prompt),
                removed_generated_positions_in_this_forward=removed,
                native_cleanup_after_current_logits_not_yet_visible=True,
            )
        )

    def __enter__(self):
        original_prefill = self.adapter.prefill
        original_decode = self.adapter.decode_one_token
        original_maintain = self.module.maintain_prefixkv_decode_cache

        def prefill(*args, **kwargs):
            if self.prefills:
                raise RuntimeError("diagnostic requires one original prefill and no draft")
            output = original_prefill(*args, **kwargs)
            self.prefills += 1
            self.source_length = int(output.state.logical_sequence_length)
            n = len(output.state.past_key_values.layers)
            self._native_prompt, self._native_generated = [0] * n, [0] * n
            self._score(
                output.logits,
                output.state,
                incremental_evictions=[0] * n,
                removed=[[] for _ in range(n)],
            )
            return output

        def decode(token, state, *args, **kwargs):
            index = len(self.records)
            if not index or index >= len(self.target_ids):
                raise RuntimeError("non-contiguous teacher decode trajectory")
            if token.numel() != 1 or int(token.item()) != self.target_ids[index - 1]:
                raise RuntimeError("teacher prefix differs from registered target")
            before = [
                tuple(getattr(layer, "_generated_positions", ()))
                for layer in state.past_key_values.layers
            ]
            output = original_decode(token, state, *args, **kwargs)
            layers = output.state.past_key_values.layers
            removed = [
                sorted(set(old) - set(getattr(layer, "_generated_positions", ())))
                for old, layer in zip(before, layers, strict=True)
            ]
            self._score(
                output.logits,
                output.state,
                incremental_evictions=[
                    int(getattr(layer, "eviction_count", 0)) for layer in layers
                ],
                removed=removed,
            )
            return output

        def maintain(*args, **kwargs):
            state, events = original_maintain(*args, **kwargs)
            for event in events:
                position = event.get("removed_logical_position")
                if position is None:
                    continue
                layer = int(event["layer"])
                prompt = position < self.source_length
                (self._native_prompt if prompt else self._native_generated)[layer] += 1
                self.native_events.append(
                    dict(
                        event,
                        removed_modality_boundary="prompt" if prompt else "generated",
                        first_prediction_it_can_affect=int(event["step"]) + 2,
                    )
                )
            return state, events

        self._stack.enter_context(patch.object(self.adapter, "prefill", prefill))
        self._stack.enter_context(patch.object(self.adapter, "decode_one_token", decode))
        self._stack.enter_context(
            patch.object(self.module, "maintain_prefixkv_decode_cache", maintain)
        )
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def record(self):
        if self.prefills != 1 or len(self.records) != len(self.target_ids):
            raise RuntimeError("incomplete per-token teacher capture")
        return dict(
            schema_version=1,
            measurement_type="evaluation_only_token_eviction_diagnostic",
            source_prompt_logical_length=self.source_length,
            target_token_count=len(self.target_ids),
            records=self.records,
            native_post_forward_cleanup_events=self.native_events,
            sum_nll=sum(x["nll"] for x in self.records),
            prediction_positions_are_one_based=True,
            prefill_prediction_precedes_compression=True,
            timing_not_comparable_to_primary_latency=True,
            used_by_online_selector=False,
        )


def first_divergence(a, b):
    if len(a) != len(b):
        raise ValueError("equal fixed-length token trajectories required")
    return next((i + 1 for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)


def compare_nll(records, reference, epsilon=1e-6):
    """Report raw first increase plus fixed 16-token window/cumulative summaries.

    Epsilon is a preregistered descriptive noise filter, NOT a relaxed quality
    acceptance threshold. NLL is in nats; no cross-task metric average is used.
    """
    if len(records) != len(reference):
        raise ValueError("paired complete target lengths required")
    deltas = []
    for index, (a, b) in enumerate(zip(records, reference, strict=True), start=1):
        if (
            a["prediction_position"] != index
            or b["prediction_position"] != index
            or a["target_token_id"] != b["target_token_id"]
        ):
            raise ValueError("teacher targets/position alignment mismatch")
        d = a["nll"] - b["nll"]
        if not isfinite(d):
            raise ValueError("nonfinite NLL comparison")
        deltas.append(d)
    first_eviction = next(
        (
            x["prediction_position"]
            for x in records
            if any(x["generated_evictions_visible_per_layer"])
        ),
        None,
    )
    first_increase = next((i + 1 for i, d in enumerate(deltas) if d > epsilon), None)
    cumulative, first_cumulative = 0.0, None
    for i, d in enumerate(deltas, start=1):
        cumulative += d
        if first_cumulative is None and cumulative > epsilon:
            first_cumulative = i
    return dict(
        first_token_nll_increase=first_increase,
        first_cumulative_nll_increase=first_cumulative,
        first_teacher_greedy_divergence=next(
            (
                a["prediction_position"]
                for a, b in zip(records, reference, strict=True)
                if "greedy_token_id" in a
                and "greedy_token_id" in b
                and a["greedy_token_id"] != b["greedy_token_id"]
            ),
            None,
        ),
        first_prompt_eviction_affected_prediction=next(
            (
                x["prediction_position"]
                for x in records
                if any(x.get("prompt_evictions_visible_per_layer", ()))
            ),
            None,
        ),
        first_generated_eviction_affected_prediction=first_eviction,
        descriptive_epsilon_nats=epsilon,
        nll_delta_total=sum(deltas),
        nll_delta_before_first_generated_eviction=sum(deltas[: first_eviction - 1])
        if first_eviction
        else sum(deltas),
        nll_delta_from_first_generated_eviction=sum(deltas[first_eviction - 1 :])
        if first_eviction
        else None,
        fixed_windows=[
            dict(first=i + 1, last=min(i + 16, len(deltas)), nll_delta=sum(deltas[i : i + 16]))
            for i in range(0, len(deltas), 16)
        ],
        per_token_deltas=deltas,
        temporal_alignment_is_not_causal_proof=True,
    )
