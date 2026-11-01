"""Source-funded exact visual-token trajectories, no donors or weight updates.

One source position is restored wherever absent across layers/heads. Each K/V
item remains layer/head-local; the cross-layer group is a transaction, NOT a
mixed-layer KV block or a prototype. Existing decoded cache bits stay unchanged.
"""

from dataclasses import dataclass

import torch

from mosaickv.evaluation.cache_cycles import compile_cycles
from mosaickv.evaluation.exact_storage import decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@torch.inference_mode()
def rank_tokens(attention_steps, source, packets, memberships, attentions, visual):
    if len(attention_steps) != 4 or not source or not visual:
        raise ValueError("four draft queries and authoritative visual positions required")
    n = source[0][0].shape[-2]
    if tuple(sorted(set(visual))) != tuple(visual) or not 0 <= visual[0] <= visual[-1] < n:
        raise ValueError("sorted unique visual positions required")
    score = torch.zeros(n, device=source[0][0].device)
    for layer, ((key, value), packet, heads, attention) in enumerate(
        zip(source, packets, memberships, attentions, strict=True)
    ):
        h, nn, d = key.shape[1:]
        if nn != n or len(heads) != h or d != 128:
            raise ValueError("complete uniform full-MHA source geometry required")
        demand = torch.stack([step[layer][0, :, 0, :n].float() for step in attention_steps], 1)
        demand = demand / demand.sum(-1, keepdim=True).clamp_min(1e-30)
        ids = torch.tensor(heads, device=key.device)
        selected = demand.gather(2, ids[:, None].expand(-1, 4, -1))
        mass = selected.sum(-1, keepdim=True).clamp_min(1e-30)
        old_v = packet.materialize(attention)[1][0].float()
        old = torch.bmm(selected / mass, old_v)
        full = torch.bmm(demand, value[0].float())
        residual = old - full
        dot_residual = torch.bmm(residual, value[0].float().transpose(1, 2))
        dot_old = torch.bmm(old, value[0].float().transpose(1, 2))
        distance = (
            value[0].float().square().sum(-1)[:, None]
            + old.square().sum(-1, keepdim=True)
            - 2 * dot_old
        ).clamp_min(0)
        fraction = demand / (mass + demand)
        gain = (
            -2 * fraction * (dot_residual - (residual * old).sum(-1, keepdim=True))
            - fraction.square() * distance
        )
        missing = torch.ones(h, n, device=key.device, dtype=torch.bool)
        missing.scatter_(1, ids, False)
        score += (gain * missing[:, None]).mean((0, 1))
    if not torch.isfinite(score).all():
        raise ValueError("nonfinite joint forecast reconstruction gain")
    return tuple(
        sorted(
            (dict(position=p, forecast_gain=float(score[p])) for p in visual),
            key=lambda r: (-r["forecast_gain"], r["position"]),
        )
    )


@dataclass(frozen=True)
class VisualTokenFormat:
    base: object
    bodies: tuple
    flat_slots: object
    cycles: object
    valid_prompt_counts: object
    kind: str = "source_funded_visual_trajectory_diagnostic"

    @property
    def shape(self):
        return (*self.base.shape[:-2], self.base.shape[-2] + 1, self.base.shape[-1])

    def private(self):
        return (
            *self.base.private(),
            *(t for b in self.bodies for t in b.buffers),
            self.flat_slots,
            *self.cycles.tensors(),
            self.valid_prompt_counts,
        )

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.base.fill(attention, targets, direct=direct)
        n = self.shape[-2]
        for body, target in zip(self.bodies, targets, strict=True):
            target[..., n - 1 : n, :].zero_()
            if target.device.type == "cuda":
                from mosaickv.evaluation.head_local_precision_kernel import scatter

                scatter(body, self.flat_slots, target, prompt_length=n)
            else:
                heads, slots = self.flat_slots.long() // n, self.flat_slots.long() % n
                target[0, heads, slots] = decoded(body)[0, 0]
        self.cycles.apply(targets)

    def materialize(self, attention):
        result = tuple(
            torch.empty(self.shape, device=self.flat_slots.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, result, direct=True)
        return result


@torch.inference_mode()
def restore_token(packets, source, memberships, attentions, position):
    if type(position) is not int or not source or not 0 <= position < source[0][0].shape[-2]:
        raise ValueError("valid original source position required")
    result, proofs = [], []
    for layer, (packet, pair, heads, attention) in enumerate(
        zip(packets, source, memberships, attentions, strict=True)
    ):
        n, d = packet.shape[-2:]
        if len(heads) != packet.shape[1] or any(
            len(row) != n or tuple(row) != tuple(sorted(set(row))) for row in heads
        ):
            raise ValueError("uniform sorted original head memberships required")
        added_heads = tuple(h for h, row in enumerate(heads) if position not in row)
        if not added_heads:
            result.append(packet)
            proofs.append(dict(layer=layer, added_heads=[], added_KV_items=0))
            continue
        device = pair[0].device
        if any(t.dtype != torch.bfloat16 or t.shape != pair[0].shape for t in pair):
            raise ValueError("original same-shape BF16 K/V required")
        idx = torch.tensor(added_heads, device=device)
        original = packet.materialize(attention)
        bodies = tuple(encode(t[0, idx, position].reshape(1, 1, -1, d).contiguous()) for t in pair)
        flat = torch.tensor(
            [h * (n + 1) + n for h in added_heads], device=device, dtype=torch.int32
        )
        source_n = pair[0].shape[-2]
        physical = tuple(
            tuple((*row, position if h in added_heads else source_n)) for h, row in enumerate(heads)
        )
        cycles = compile_cycles(physical, device=device)
        counts = torch.tensor(
            [n + int(h in added_heads) for h in range(len(heads))], device=device, dtype=torch.int16
        )
        candidate = VisualTokenFormat(packet, bodies, flat, cycles, counts)
        actual = candidate.materialize(attention)
        for head, row in enumerate(heads):
            sorted_positions = sorted(physical[head])
            preserved = [sorted_positions.index(p) for p in row]
            for old, new, full in zip(original, actual, pair, strict=True):
                if not torch.equal(old[0, head], new[0, head, preserved]):
                    raise RuntimeError("visual restoration changed original selected K/V bits")
                if head in added_heads and not torch.equal(
                    new[0, head, sorted_positions.index(position)], full[0, head, position]
                ):
                    raise RuntimeError("restored visual K/V is not original BF16")
        result.append(candidate)
        proofs.append(
            dict(
                layer=layer,
                added_heads=added_heads,
                added_KV_items=len(added_heads),
                added_position=position,
                physical_padding_slots=len(heads) - len(added_heads),
                original_selected_KV_bitwise_preserved=True,
                original_added_KV_bitwise=True,
                no_donor_removal=True,
                candidate_component_bytes=candidate.component_bytes,
            )
        )
    return tuple(result), proofs
