"""Training-free precision variant, shared with every compared selector.

No evictions, merges, positions, history policies, or model weights change.
Text protection is local to its head, not a union over unrelated head slots.
Lower-bit variants are explicitly lossy and need independent quality checks.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from mosaickv.evaluation import request_precision
from mosaickv.evaluation.exact_storage import ExactFormat, decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_corrections import fund_corrections, prompt_weights
from mosaickv.evaluation.precision_tier import PrecisionFormat, construct, pack
from mosaickv.evaluation.storage_baselines import baseline_payload

POLICIES = {
    "union3": (3, 3),
    "head3": (3, 3),
    "head3k2v": (3, 2),
    "head2k3v": (2, 3),
    "head2": (2, 2),
}


@dataclass(frozen=True)
class HeadLocalFormat:
    base: PrecisionFormat
    flat_tokens: Any
    protected: tuple[Any, ...]
    kind: str = "evicastkv_head_local_precision_variant"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base.shape

    def private(self) -> tuple[Any, ...]:
        return (
            *self.base.private(),
            self.flat_tokens,
            *(t for body in self.protected for t in body.buffers),
        )

    def shared(self) -> tuple[Any, ...]:
        return ()

    def tensors(self) -> tuple[Any, ...]:
        return self.private()

    @property
    def component_bytes(self) -> int:
        return backing_bytes(self.private())

    def fill(self, attention: Any, destinations: Any, *, direct: bool) -> None:
        self.base.fill(attention, destinations, direct=direct)
        if not self.protected:
            return
        n = self.shape[-2]
        for body, target in zip(self.protected, destinations, strict=True):
            if target.device.type == "cuda":
                from mosaickv.evaluation.head_local_precision_kernel import scatter

                scatter(body, self.flat_tokens, target, prompt_length=n)
            else:
                heads, slots = self.flat_tokens.long() // n, self.flat_tokens.long() % n
                target[0, heads, slots, :] = decoded(body)[0, 0]

    def materialize(self, attention: Any = None) -> tuple[Any, ...]:
        output = tuple(
            torch.empty(self.shape, device=b.codes.device, dtype=torch.bfloat16)
            for b in self.base.pair
        )
        self.fill(attention, output, direct=True)
        return output


def localize(
    fmt: PrecisionFormat, pair: Any, heads: Any, text: set[int], *, policy: str
) -> HeadLocalFormat:
    if policy not in POLICIES or policy == "union3" or not fmt.encoded_exact:
        raise ValueError("registered head-local policy and encoded exact reference required")
    h, n, d = fmt.shape[1:]
    if len(heads) != h or any(len(row) != n for row in heads):
        raise ValueError("selected per-head membership must match packet shape")
    common = tuple(sorted(set(range(min(4, n))) | set(range(max(0, n - 8), n))))
    common_set = set(common)
    extra = tuple(
        head * n + j
        for head, row in enumerate(heads)
        for j, p in enumerate(row)
        if p in text and j not in common_set
    )
    indices = torch.tensor(common, dtype=torch.int32, device=pair[0].device)
    flat = torch.tensor(extra, dtype=torch.int32, device=pair[0].device)
    base_exact = tuple(encode(t.index_select(2, indices.long()).contiguous()) for t in pair)
    protected = (
        tuple(
            encode(t.reshape(1, 1, h * n, d).index_select(2, flat.long()).contiguous())
            for t in pair
        )
        if extra
        else ()
    )
    bodies = tuple(
        b if b.bits == bits else pack(t, bits=bits, axis=b.axis)
        for b, t, bits in zip(fmt.pair, pair, POLICIES[policy], strict=True)
    )
    base = replace(fmt, pair=bodies, exact_indices=indices, exact_values=base_exact)
    return HeadLocalFormat(base, flat, protected)


def bundle(
    adapter: Any,
    prepared: Any,
    prefill: Any,
    capture: Any,
    cfg: Any,
    method: str,
    profile_path: str,
    ratio: float,
) -> tuple[Any, dict[str, Any]]:
    """Compute original selection/merge once; build all registered packets."""
    frozen, online = request_precision.online_frozen(
        prefill,
        tuple(json.loads(Path(profile_path).read_text())),
        ratio,
        method=method,
    )
    positions, transformed, proof = baseline_payload(
        adapter,
        prepared,
        prefill,
        cfg,
        method,
        frozen,
        profile_path,
        borrow_source=True,
    )
    memberships = (
        positions if positions is not None else proof["selected_source_positions_by_layer"]
    )
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    text = {
        p
        for span in prepared.modality_map
        if span.modality == "text"
        for p in range(span.start, span.end)
    }
    formats: dict[str, list[Any]] = {p: [] for p in POLICIES}
    records = []
    for i, ((key, value), heads) in enumerate(zip(source, memberships, strict=True)):
        if transformed is not None:
            pair = transformed[i]
        else:
            index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
                1, key.shape[1], len(heads[0]), key.shape[-1]
            )
            pair = key.gather(2, index), value.gather(2, index)
        if i < 2 or i >= len(source) - 2:
            exact = ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
            for packets in formats.values():
                packets.append(exact)
            continue
        union = tuple(sorted({j for row in heads for j, p in enumerate(row) if p in text}))
        fmt = construct(pair, bits=3, protected=union, encoded_exact=True)
        kw, vw = prompt_weights(
            adapter._language_layers()[i].self_attn,
            capture.records[i],
            prefill.attention_weights[i],
            heads,
            window=16,
        )
        fmt, corrections = fund_corrections(
            fmt, pair, fraction=0.005, key_weight=kw, value_weight=vw
        )
        formats["union3"].append(fmt)
        row = dict(
            layer=i,
            corrections=corrections,
            union_protected_rows=int(fmt.exact_indices.numel()) * key.shape[1],
            head_local_protected_rows=sum(
                len(
                    {j for j, p in enumerate(head) if p in text}
                    | set(range(min(4, len(head))))
                    | set(range(max(0, len(head) - 8), len(head)))
                )
                for head in heads
            ),
            component_bytes={"union3": fmt.component_bytes},
        )
        for policy in POLICIES:
            if policy == "union3":
                continue
            localized = localize(fmt, pair, heads, text, policy=policy)
            formats[policy].append(localized)
            row["component_bytes"][policy] = localized.component_bytes
        records.append(row)
    return {name: tuple(packets) for name, packets in formats.items()}, dict(
        selection=proof,
        online=online,
        layers=records,
        source_memberships=memberships,
        same_source_cap_not_encoded_byte_matching=True,
        only_storage_precision_and_protection_changes=True,
    )


def gpu_controls() -> list[dict[str, Any]]:
    """CPU oracle for 2/3-bit decoding plus bitwise flat-scatter sentinels."""
    from mosaickv.evaluation.head_local_precision_kernel import scatter

    generator = torch.Generator().manual_seed(31)
    pair = tuple(torch.randn(1, 3, 141, 128, generator=generator).bfloat16() for _ in range(2))
    heads = tuple(tuple(range(h, h + 141)) for h in range(3))
    text = {40, 72, 99}
    union = tuple(sorted({j for head in heads for j, p in enumerate(head) if p in text}))
    cpu = construct(pair, bits=3, protected=union, encoded_exact=True)
    device_pair = tuple(t.cuda() for t in pair)
    gpu = construct(device_pair, bits=3, protected=union, encoded_exact=True)
    records = []
    for policy in POLICIES:
        if policy == "union3":
            continue
        reference = localize(cpu, pair, heads, text, policy=policy).materialize()
        candidate = localize(gpu, device_pair, heads, text, policy=policy)
        destination = tuple(
            torch.full((1, 3, 151, 128), -7, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )
        candidate.fill(None, destination, direct=True)
        if not all(
            torch.equal(a[..., :141, :].cpu().view(torch.int16), b.view(torch.int16))
            and bool((a[..., 141:, :] == -7).all())
            for a, b in zip(destination, reference, strict=True)
        ):
            raise RuntimeError("head-local CUDA precision/scatter differs from CPU oracle")
        records.append(dict(policy=policy, bitwise_CPU_oracle=True, history_sentinel=True))
    data = torch.arange(65536, dtype=torch.int32, device="cuda").short().view(torch.bfloat16)
    data = data.reshape(1, 1, 512, 128)
    body = encode(data)
    flat = torch.arange(512, device="cuda", dtype=torch.int32) * 2
    target = torch.full((1, 4, 265, 128), -7, device="cuda", dtype=torch.bfloat16)
    scatter(body, flat, target, prompt_length=256)
    actual = target[0, flat.long() // 256, flat.long() % 256]
    if not torch.equal(actual.view(torch.int16), data[0, 0].view(torch.int16)):
        raise RuntimeError("flat protected scatter changed a BF16 bit pattern")
    if not bool((target[..., 256:, :] == -7).all()):
        raise RuntimeError("flat protected scatter overwrote generated history")
    records.append(dict(all_65536_BF16_patterns_bitwise=True, history_sentinel=True))
    return records
