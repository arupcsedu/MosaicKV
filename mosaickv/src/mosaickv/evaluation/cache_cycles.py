"""Bitwise in-place cache row permutation; disjoint cycles, no second KV lease."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CacheCycles:
    heads: Any
    offsets: Any
    rows: Any
    length: int

    def tensors(self):
        return self.heads, self.offsets, self.rows

    def apply(self, targets):
        if (
            len(targets) != 2
            or targets[0].shape != targets[1].shape
            or targets[0].ndim != 4
            or targets[0].shape[0] != 1
            or targets[0].shape[-2] < self.length
            or any(t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in targets)
        ):
            raise ValueError("matching feature-contiguous BF16 cache leases required")
        if not self.heads.numel():
            return
        if targets[0].device.type == "cuda":
            from mosaickv.evaluation.cache_cycles_kernel import apply

            apply(self, targets)
            return
        words = tuple(t.view(torch.int16) for t in targets)
        for cycle, head in enumerate(self.heads.tolist()):
            rows = self.rows[self.offsets[cycle] : self.offsets[cycle + 1]].tolist()
            temporary = [t[0, head, rows[0]].clone() for t in words]
            for row in (*rows[1:], rows[0]):
                for i, t in enumerate(words):
                    previous = t[0, head, row].clone()
                    t[0, head, row].copy_(temporary[i])
                    temporary[i] = previous


def compile_cycles(physical_positions, *, device):
    if (
        not physical_positions
        or len({len(h) for h in physical_positions}) != 1
        or not 1 <= len(physical_positions[0]) < 32768
        or len(physical_positions) >= 32768
        or any(
            len(set(h)) != len(h) or any(type(p) is not int or p < 0 for p in h)
            for h in physical_positions
        )
    ):
        raise ValueError("uniform int16-addressable distinct original positions required")
    heads, rows, offsets = [], [], [0]
    for h, positions in enumerate(physical_positions):
        rank = {p: j for j, p in enumerate(sorted(positions))}
        destination = [rank[p] for p in positions]
        seen = set()
        for start in range(len(positions)):
            if start in seen:
                continue
            cycle, current = [], start
            while current not in seen:
                cycle.append(current)
                seen.add(current)
                current = destination[current]
            if current != start:
                raise RuntimeError("invalid row permutation cycle")
            if len(cycle) > 1:
                heads.append(h)
                rows.extend(cycle)
                offsets.append(len(rows))
    return CacheCycles(
        torch.tensor(heads, device=device, dtype=torch.int16),
        torch.tensor(offsets, device=device, dtype=torch.int32),
        torch.tensor(rows, device=device, dtype=torch.int16),
        len(physical_positions[0]),
    )


def gpu_controls():
    generator = torch.Generator().manual_seed(808)
    positions = tuple(tuple(torch.randperm(129, generator=generator).tolist()) for _ in range(4))
    plan = compile_cycles(positions, device="cuda")
    words = torch.arange(4 * 146 * 128, device="cuda").short()
    original = words.view(torch.bfloat16).reshape(1, 4, 146, 128)
    index = torch.tensor([sorted(range(129), key=lambda j: p[j]) for p in positions], device="cuda")
    expected = original.view(torch.int16)[:, :, :129].gather(
        2, index[None, :, :, None].expand(1, 4, 129, 128)
    )
    for _ in range(3):
        targets = original.clone(), original.clone()
        plan.apply(targets)
        for t in targets:
            if not torch.equal(
                t[:, :, :129].view(torch.int16), expected.view(torch.int16)
            ) or not torch.equal(
                t[:, :, 129:].view(torch.int16), original[:, :, 129:].view(torch.int16)
            ):
                raise RuntimeError("cache permutation corrupted K/V or generated-history sentinel")
    return dict(
        all_65536_BF16_patterns_bitwise=True,
        history_sentinel_unchanged=True,
        repetitions=3,
        second_KV_buffer_allocated=False,
    )
