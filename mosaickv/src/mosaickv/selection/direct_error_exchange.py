"""Bounded exact-byte layer exchanges using renormalized output error.

Prompt-derived tensors only. Reconstruction is a surrogate, not answer fidelity.
No model probes, benchmark references, replacement values or ragged head padding.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaickv.selection.evidence_layer_allocation import (
    RECONSTRUCTION_ATOL,
    RECONSTRUCTION_RTOL,
    EvidenceLayerPlan,
    reconstruction_guard,
    reconstruction_mse,
)

# Versioned policy constants, frozen before successor identities are selected.
MAX_EXCHANGES = 16
MAX_VERIFIED_PAIRS = 64
CANDIDATE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class SlotMove:
    positions: tuple[int, ...]
    errors: tuple[float, ...]
    gain: float


class LayerContext:
    """Build source products once; refresh only layers changed by a transaction."""

    def __init__(self, mass: Any, values: Any, positions, mandatory):
        import torch

        # Reuse the audited reconstruction kernel's finite/positive checks.
        baseline = reconstruction_mse(mass, values, positions)
        self.mass = mass.detach()
        self.weighted = (mass.unsqueeze(-1) * values).detach()
        self.target = self.weighted.sum(1, dtype=torch.float64) / mass.sum(
            -1, keepdim=True, dtype=torch.float64
        )
        self.anchor_errors = baseline
        self.mask = torch.zeros_like(mass, dtype=torch.bool)
        self.mandatory = torch.zeros_like(self.mask)
        for head, row in enumerate(positions):
            self.mask[head, list(row)] = True
            self.mandatory[head, list(mandatory[head])] = True
        if bool((self.mandatory & ~self.mask).any()):
            raise ValueError("exchange anchor omits a mandatory source position")
        self.refresh()

    def refresh(self):
        import torch

        self.retained_mass = (self.mass * self.mask).sum(-1, keepdim=True, dtype=torch.float64)
        self.numerator = (self.weighted * self.mask.unsqueeze(-1)).sum(1, dtype=torch.float64)
        self.errors = (self.numerator / self.retained_mass - self.target).square().mean(-1)

    def positions(self):
        return tuple(tuple(int(p) for p in row.nonzero().flatten().tolist()) for row in self.mask)

    def best_move(self, *, remove: bool) -> SlotMove | None:
        """Best safe position independently per head; one dense slot for a layer."""
        import torch

        eligible = (self.mask & ~self.mandatory) if remove else ~self.mask
        limit = torch.minimum(
            self.errors + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * self.errors,
            self.anchor_errors + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * self.anchor_errors,
        )
        best = torch.full_like(self.errors, float("inf"))
        indices = torch.full((self.mass.shape[0],), -1, dtype=torch.long, device=self.mass.device)
        sign = -1 if remove else 1
        for start in range(0, self.mass.shape[1], CANDIDATE_CHUNK_SIZE):
            end = min(start + CANDIDATE_CHUNK_SIZE, self.mass.shape[1])
            denominator = self.retained_mass + sign * self.mass[:, start:end].double()
            numerator = self.numerator.unsqueeze(1) + sign * self.weighted[:, start:end].double()
            output = numerator / denominator.clamp_min(1e-30).unsqueeze(-1)
            errors = (output - self.target.unsqueeze(1)).square().mean(-1)
            valid = (
                eligible[:, start:end]
                & (denominator > 0)
                & torch.isfinite(errors)
                & (errors <= limit.unsqueeze(-1))
            )
            errors = errors.masked_fill(~valid, float("inf"))
            candidate, index = errors.min(-1)
            # Strict replacement retains the first physical position on ties.
            improve = candidate < best
            best = torch.where(improve, candidate, best)
            indices = torch.where(improve, index + start, indices)
        if not bool(torch.isfinite(best).all()):
            return None
        return SlotMove(
            tuple(int(p) for p in indices.tolist()),
            tuple(float(e) for e in best.tolist()),
            float((self.errors - best).sum().item()),
        )

    def changed_positions(self, move: SlotMove, *, remove: bool):
        rows = [set(row) for row in self.positions()]
        for head, position in enumerate(move.positions):
            if remove:
                rows[head].remove(position)
            else:
                rows[head].add(position)
        return tuple(tuple(sorted(row)) for row in rows)

    def exact_errors(self, positions):
        import torch

        mask = torch.zeros_like(self.mask)
        for head, row in enumerate(positions):
            mask[head, list(row)] = True
        denominator = (self.mass * mask).sum(-1, keepdim=True, dtype=torch.float64)
        if bool((denominator <= 0).any()):
            raise ValueError("exchange removed all retained attention mass")
        numerator = (self.weighted * mask.unsqueeze(-1)).sum(1, dtype=torch.float64)
        return (numerator / denominator - self.target).square().mean(-1)

    def install_positions(self, positions):
        self.mask.zero_()
        for head, row in enumerate(positions):
            self.mask[head, list(row)] = True
        self.refresh()


def _safe(context: LayerContext, errors: Any) -> bool:
    import torch

    return bool(torch.isfinite(errors).all()) and bool(
        (
            (errors <= context.errors + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * context.errors)
            & (
                errors
                <= context.anchor_errors
                + RECONSTRUCTION_ATOL
                + RECONSTRUCTION_RTOL * context.anchor_errors
            )
        ).all()
    )


def build_direct_error_layer_plan(
    full_state: Any,
    token_attention: dict[tuple[int, int], Any],
    anchor_records: tuple[dict[str, Any], ...],
    *,
    prompt_budget_bytes: int,
    uniform_capacity: int,
) -> EvidenceLayerPlan:
    """One removed token/head in donor layer funds one added token/head elsewhere."""
    import torch

    started = time.perf_counter()
    layers = full_state.layers
    if not layers or type(uniform_capacity) is not int or uniform_capacity < 1:
        raise ValueError("exchange requires nonempty layers and a positive anchor capacity")
    costs = tuple(layer.byte_size // layer.sequence_length for layer in layers)
    if len(set(costs)) != 1 or uniform_capacity * sum(costs) != prompt_budget_bytes:
        raise ValueError("exchange requires equal layer-slot costs and exact anchor bytes")
    logical = full_state.logical_positions.physical_to_logical
    physical = {p: i for i, p in enumerate(logical)}
    selected: dict[tuple[int, int], set[int]] = {
        (i, h): set() for i, layer in enumerate(layers) for h in range(layer.kv_heads)
    }
    mandatory: dict[tuple[int, int], set[int]] = {key: set() for key in selected}
    for block in full_state.blocks:
        if block.mandatory:
            mandatory[block.layer, block.kv_head].update(block.physical_cache_indices)
    for row in anchor_records:
        if row.get("tier") != "exact":
            raise ValueError("exchange requires an exact-only anchor")
        route = int(row["layer"]), int(row["kv_head"])
        position = physical[int(row["logical_position"])]
        if position in selected[route]:
            raise ValueError("exchange anchor contains duplicate source membership")
        selected[route].add(position)
    contexts = []
    anchor_positions = []
    for i, layer in enumerate(layers):
        if (
            layer.key_head_dimension,
            layer.key_sequence_dimension,
            layer.value_head_dimension,
            layer.value_sequence_dimension,
        ) != (1, 2, 1, 2) or int(layer.value.shape[0]) != 1:
            raise ValueError("exchange supports audited batch-one BHSD source tensors only")
        head_positions = tuple(tuple(sorted(selected[i, h])) for h in range(layer.kv_heads))
        if any(len(row) != uniform_capacity for row in head_positions):
            raise ValueError("exchange anchor has padded or unequal per-head capacity")
        values = layer.value.detach()[0].float()
        mass = torch.as_tensor(
            np.stack([token_attention[i, h] for h in range(layer.kv_heads)]),
            device=values.device,
            dtype=torch.float32,
        )
        reconstruction_mse(mass, values, head_positions)
        mass = mass / mass.sum(-1, keepdim=True)
        contexts.append(
            LayerContext(
                mass,
                values,
                head_positions,
                tuple(tuple(sorted(mandatory[i, h])) for h in range(layer.kv_heads)),
            )
        )
        anchor_positions.append(head_positions)
    anchors = tuple(tuple(float(v) for v in c.anchor_errors.tolist()) for c in contexts)
    counts = [uniform_capacity] * len(layers)
    lows = [
        max(
            1,
            max(len(mandatory[i, h]) for h in range(layer.kv_heads)),
            math.floor(uniform_capacity * 0.5),
        )
        for i, layer in enumerate(layers)
    ]
    highs = [min(layer.sequence_length, math.ceil(uniform_capacity * 1.5)) for layer in layers]
    preparations = time.perf_counter() - started
    options = {}
    evaluations = 0

    def update(index):
        nonlocal evaluations
        c = contexts[index]
        options[index] = (
            c.best_move(remove=True) if counts[index] > lows[index] else None,
            c.best_move(remove=False) if counts[index] < highs[index] else None,
        )
        evaluations += 1

    for i in range(len(layers)):
        update(i)
    initial_donors = sum(row[0] is not None for row in options.values())
    initial_receivers = sum(row[1] is not None for row in options.values())
    transactions: list[dict[str, Any]] = []
    verified, rejected = 0, 0
    stop = "exchange_limit"
    while len(transactions) < MAX_EXCHANGES:
        pairs = sorted(
            (-(donor.gain + receiver.gain), d, r)
            for d, (donor, _) in options.items()
            if donor is not None
            for r, (_, receiver) in options.items()
            if receiver is not None and d != r and donor.gain + receiver.gain > 0
        )
        if not pairs:
            stop = (
                "no_safe_donor"
                if not any(x[0] is not None for x in options.values())
                else "no_safe_receiver"
                if not any(x[1] is not None for x in options.values())
                else "no_improving_pair"
            )
            break
        committed = False
        for _, donor, receiver in pairs:
            if verified >= MAX_VERIFIED_PAIRS:
                break
            verified += 1
            remove, add = options[donor][0], options[receiver][1]
            d, r = contexts[donor], contexts[receiver]
            dpos = d.changed_positions(remove, remove=True)
            rpos = r.changed_positions(add, remove=False)
            derr, rerr = d.exact_errors(dpos), r.exact_errors(rpos)
            gain = float((d.errors - derr).sum().item() + (r.errors - rerr).sum().item())
            if not _safe(d, derr) or not _safe(r, rerr) or not gain > 0:
                rejected += 1
                continue
            event = dict(
                donor_layer=donor,
                receiver_layer=receiver,
                transferred_bytes=costs[donor],
                removed_positions=remove.positions,
                added_positions=add.positions,
                donor_mse_before=d.errors.tolist(),
                donor_mse_after=derr.tolist(),
                receiver_mse_before=r.errors.tolist(),
                receiver_mse_after=rerr.tolist(),
                summed_head_mse_gain=gain,
            )
            d.install_positions(dpos)
            r.install_positions(rpos)
            counts[donor] -= 1
            counts[receiver] += 1
            if sum(n * c for n, c in zip(counts, costs, strict=True)) != prompt_budget_bytes:
                raise ValueError("exchange violated exact physical byte conservation")
            event["capacities_after"] = tuple(counts)
            transactions.append(event)
            update(donor)
            update(receiver)
            committed = True
            break
        if not committed:
            stop = "verification_limit" if verified >= MAX_VERIFIED_PAIRS else "no_verified_pair"
            break
    positions = tuple(c.positions() for c in contexts)
    # Recompute from original source tensors, not the analytic option scores.
    final_errors = tuple(
        tuple(
            float(v)
            for v in reconstruction_mse(
                c.mass,
                layer.value.detach()[0].float(),
                heads,
            ).tolist()
        )
        for c, layer, heads in zip(contexts, layers, positions, strict=True)
    )
    guard = reconstruction_guard(anchors, final_errors)
    guard["proposed_layer_capacities"] = tuple(counts)
    guard["proposed_positions_sha256"] = hashlib.sha256(
        json.dumps(positions, separators=(",", ":")).encode()
    ).hexdigest()
    guard["anchor_positions_sha256"] = hashlib.sha256(
        json.dumps(tuple(anchor_positions), separators=(",", ":")).encode()
    ).hexdigest()
    if not guard["accepted"]:
        positions = tuple(anchor_positions)
        counts = [uniform_capacity] * len(layers)
        stop = "final_guard_rollback"
    guard["host_wall_seconds_including_device_transfers"] = time.perf_counter() - started
    guard["exchange_search"] = dict(
        policy="direct_error_single_slot_layer_exchange_v1",
        max_exchanges=MAX_EXCHANGES,
        max_verified_pairs=MAX_VERIFIED_PAIRS,
        candidate_chunk_size=CANDIDATE_CHUNK_SIZE,
        context_builds=len(contexts),
        option_layer_refreshes=evaluations,
        preparation_seconds=preparations,
        verified_pairs=verified,
        rejected_pairs=rejected,
        initial_safe_donor_layers=initial_donors,
        initial_safe_receiver_layers=initial_receivers,
        initial_layer_capacities=tuple(uniform_capacity for _ in layers),
        layer_slot_bytes=costs[0],
        tentative_exchanges=len(transactions),
        committed_exchanges=len(transactions) if guard["accepted"] else 0,
        changed_from_anchor=positions != tuple(anchor_positions),
        stop_reason=stop,
        transactions=transactions,
        original_anchor_guard_rechecked_after_all_exchanges=True,
    )
    masses = tuple(
        float((c.mass * c.mask).sum(-1).mean().item())
        if guard["accepted"]
        else float(np.mean([c.mass[h, list(row)].sum().item() for h, row in enumerate(heads)]))
        for c, heads in zip(contexts, positions, strict=True)
    )
    return EvidenceLayerPlan(
        positions,
        tuple(counts),
        prompt_budget_bytes,
        tuple(float(np.mean(np.sqrt(row))) for row in anchors),
        masses,
        tuple(
            max(len(mandatory[i, h]) for h in range(layer.kv_heads))
            for i, layer in enumerate(layers)
        ),
        guard,
    )


def validate_exchange_trace(allocation: dict[str, Any]) -> dict[str, Any]:
    """Replay serialized atomic transactions against unchanged original-anchor bounds."""
    from mosaickv.selection.evidence_layer_allocation import validate_reconstruction_trace

    check = validate_reconstruction_trace(allocation)
    guard = allocation["reconstruction_guard"]
    search = guard["exchange_search"]
    events = search["transactions"]
    if (
        search["max_exchanges"] != MAX_EXCHANGES
        or search["max_verified_pairs"] != MAX_VERIFIED_PAIRS
        or search["candidate_chunk_size"] != CANDIDATE_CHUNK_SIZE
        or search["tentative_exchanges"] != len(events)
        or not len(events) <= search["verified_pairs"] <= MAX_VERIFIED_PAIRS
        or len(events) > MAX_EXCHANGES
    ):
        raise ValueError("exchange trace violates registered search limits")
    anchors = guard["anchor_mse_by_layer_head"]
    current = [list(row) for row in anchors]
    counts = list(search["initial_layer_capacities"])
    slot_bytes = search["layer_slot_bytes"]
    if not counts or slot_bytes <= 0 or sum(counts) * slot_bytes != allocation["prompt_bytes"]:
        raise ValueError("exchange trace initial byte accounting mismatch")
    for event in events:
        d, r = event["donor_layer"], event["receiver_layer"]
        if d == r or not 0 <= d < len(counts) or not 0 <= r < len(counts):
            raise ValueError("exchange trace has invalid donor/receiver routes")
        if event["transferred_bytes"] != slot_bytes:
            raise ValueError("exchange trace transfer changes physical byte cost")
        gain = 0.0
        for index, role, positions in (
            (d, "donor", event["removed_positions"]),
            (r, "receiver", event["added_positions"]),
        ):
            before, after = event[f"{role}_mse_before"], event[f"{role}_mse_after"]
            if (
                before != current[index]
                or len(after) != len(before)
                or len(positions) != len(before)
            ):
                raise ValueError("exchange trace per-head transaction continuity mismatch")
            for old, new, anchor in zip(before, after, anchors[index], strict=True):
                if (
                    not math.isfinite(new)
                    or new < 0
                    or (
                        new > old + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * old
                        or new > anchor + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * anchor
                    )
                ):
                    raise ValueError("exchange trace violated current/original per-head guard")
                gain += old - new
            current[index] = after
        if not gain > 0 or not math.isclose(
            gain, event["summed_head_mse_gain"], abs_tol=1e-12, rel_tol=1e-9
        ):
            raise ValueError("exchange trace lacks strictly positive reconstruction gain")
        counts[d] -= 1
        counts[r] += 1
        if counts != list(event["capacities_after"]) or min(counts) < 1:
            raise ValueError("exchange trace capacity transaction mismatch")
        if sum(counts) * slot_bytes != allocation["prompt_bytes"]:
            raise ValueError("exchange trace violated exact byte cap")
    committed = len(events) if guard["accepted"] else 0
    changed = allocation["selected_positions_sha256"] != guard["anchor_positions_sha256"]
    if search["committed_exchanges"] != committed or search["changed_from_anchor"] != changed:
        raise ValueError("exchange trace final committed/changed status mismatch")
    if bool(committed) != changed:
        raise ValueError("exchange trace claims transactions without a changed cache")
    if guard["accepted"] and counts != list(allocation["layer_capacities"]):
        raise ValueError("exchange trace final layer capacities mismatch")
    for expected, actual in zip(current, guard["proposed_mse_by_layer_head"], strict=True):
        if any(
            not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-9)
            for a, b in zip(expected, actual, strict=True)
        ):
            raise ValueError("exchange trace final error differs from replayed transactions")
    return dict(
        **check,
        exchange_committed=committed,
        exchange_changed=changed,
        exchange_verified_pairs=search["verified_pairs"],
        exchange_initial_donors=search["initial_safe_donor_layers"],
        exchange_stop_reason=search["stop_reason"],
    )


def cpu_feasibility() -> dict[str, Any]:
    """Fixed synthetic gate: executable safe moves AND explicit safe-no-move case.

    This never loads model weights, datasets or answers; it is NOT quality data.
    The architecture-sized fixture checks 32 layers / 32 heads, with smaller
    synthetic sequence and value dimensions than the real checkpoint.
    """
    import torch

    from mosaickv.cache_state import FullKVState

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        results = {}
        generator = torch.Generator().manual_seed(292)
        for name in ("positive_exact", "negative_no_safe_donor", "random_32_layers_32_heads"):
            started = time.perf_counter()
            if name == "random_32_layers_32_heads":
                layers = tuple(
                    (
                        torch.randn(1, 32, 128, 16, generator=generator),
                        torch.randn(1, 32, 128, 16, generator=generator),
                    )
                    for _ in range(32)
                )
                capacity, retained = 32, (*range(31), 127)
            else:
                rows = (
                    ([0.0, 2.0, 1.0, 1.0],) * 2
                    if name == "negative_no_safe_donor"
                    else (
                        [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                    )
                )
                layers = tuple(
                    (
                        torch.zeros(1, 32, len(row), 1),
                        torch.tensor(row).reshape(1, 1, len(row), 1).repeat(1, 32, 1, 1),
                    )
                    for row in rows
                )
                capacity = len(rows[0]) // 2
                retained = tuple(range(capacity))
            state = FullKVState.from_tensors(
                layers,
                block_size=1,
                mandatory_logical_positions=(retained[-1],),
            )
            attention = {
                (i, h): np.ones(layer.sequence_length) / layer.sequence_length
                for i, layer in enumerate(state.layers)
                for h in range(layer.kv_heads)
            }
            records = tuple(
                dict(layer=i, kv_head=h, logical_position=p, tier="exact")
                for i, layer in enumerate(state.layers)
                for h in range(layer.kv_heads)
                for p in retained
            )
            budget = (
                sum(layer.byte_size // layer.sequence_length for layer in state.layers) * capacity
            )
            plan = build_direct_error_layer_plan(
                state,
                attention,
                records,
                prompt_budget_bytes=budget,
                uniform_capacity=capacity,
            )
            trace = json.loads(json.dumps(plan.trace(), allow_nan=False))
            check = validate_exchange_trace(trace)
            if plan.reconstruction is None:
                raise ValueError("CPU feasibility lacks reconstruction proof")
            search = plan.reconstruction["exchange_search"]
            passed = bool(plan.prompt_bytes == budget and check["accepted"])
            if name == "negative_no_safe_donor":
                passed &= (
                    search["committed_exchanges"] == 0 and search["stop_reason"] == "no_safe_donor"
                )
            else:
                passed &= search["committed_exchanges"] > 0 and search["changed_from_anchor"]
                passed &= check["final_mse_mean"] < check["anchor_mse_mean"]
            results[name] = dict(
                passed=bool(passed),
                committed_exchanges=search["committed_exchanges"],
                initial_safe_donor_layers=search["initial_safe_donor_layers"],
                stop_reason=search["stop_reason"],
                exact_bytes=plan.prompt_bytes == budget,
                wall_seconds=time.perf_counter() - started,
            )
        return dict(
            classification="synthetic_CPU_feasibility_not_pretrained_quality_evidence",
            passed=all(row["passed"] for row in results.values()),
            benchmark_labels_accessed=False,
            cases=results,
        )
    finally:
        torch.set_num_threads(previous)
