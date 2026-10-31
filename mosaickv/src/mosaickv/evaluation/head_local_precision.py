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
from mosaickv.evaluation.exact_storage import ExactFormat, SparseBitPatch, decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_corrections import (
    fund_corrections,
    prompt_weights,
    stable_largest,
)
from mosaickv.evaluation.precision_tier import PrecisionFormat, construct, pack
from mosaickv.evaluation.storage_baselines import baseline_payload

POLICIES = {
    "union3": (3, 3),
    "head3": (3, 3),
    "head3k2v": (3, 2),
    "head2k3v": (2, 3),
    "head2": (2, 2),
    "head3k2v_refit": (3, 2),
    "head3k2v_headwise": (3, 2),
    "head3k2v_rows": (3, 2),
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
    *,
    policies: tuple[str, ...] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute original selection/merge once; build all registered packets."""
    requested = tuple(POLICIES) if policies is None else policies
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(p not in POLICIES for p in requested)
    ):
        raise ValueError("unique registered precision policies required")
    frozen, online = request_precision.online_frozen(
        prefill,
        tuple(json.loads(Path(profile_path).read_text())),
        ratio,
        method=method,
        priors=capture.priors if getattr(capture, "summaries", False) else None,
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
    formats: dict[str, list[Any]] = {p: [] for p in requested}
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
        if "union3" in formats:
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
        for policy in requested:
            if policy == "union3":
                continue
            localized: Any = localize(fmt, pair, heads, text, policy=policy)
            if policy in ("head3k2v_refit", "head3k2v_headwise", "head3k2v_rows"):
                localized, audit = refit_corrections(
                    localized,
                    pair,
                    key_weight=kw,
                    value_weight=vw,
                    per_head=policy == "head3k2v_headwise",
                )
                row.setdefault("refit", {})[policy] = audit
            if policy == "head3k2v_rows":
                from mosaickv.evaluation.value_row_precision import upgrade_rows

                localized, audit = upgrade_rows(
                    localized, pair, prefill.attention_weights[i], heads
                )
                row.setdefault("row_upgrades", {})[policy] = audit
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


def refit_corrections(
    packet: HeadLocalFormat,
    original: Any,
    *,
    key_weight: Any = None,
    value_weight: Any = None,
    per_head: bool = False,
) -> tuple[HeadLocalFormat, dict[str, Any]]:
    """Re-rank actual low-bit errors with exactly the old exception allowance."""
    old = packet.base.corrections
    if len(old) != 2 or type(per_head) is not bool:
        raise ValueError("two funded component budgets and explicit head policy required")
    uncorrected = replace(packet, base=replace(packet.base, corrections=()))
    decoded_ = uncorrected.materialize()
    previous = packet.materialize()
    h, n, d = packet.shape[1:]
    patches, records = [], []
    for reference, approximate, prior, budget, weight in zip(
        original, decoded_, previous, old, (key_weight, value_weight), strict=True
    ):
        if (
            tuple(reference.shape) != packet.shape
            or reference.dtype != torch.bfloat16
            or reference.device != approximate.device
        ):
            raise ValueError("same-shape original BF16 payload required")
        error = (reference.float() - approximate.float()).abs()
        score = error if weight is None else error * weight
        count = int(budget.indices.numel())
        if per_head:
            lower, remainder = divmod(count, h)
            indices = (
                torch.cat(
                    tuple(
                        stable_largest(score[0, head].flatten(), lower + int(head < remainder))
                        + head * n * d
                        for head in range(h)
                    )
                )
                .sort()
                .values
            )
        else:
            indices = stable_largest(score.flatten(), count)
        values = reference.contiguous().flatten().index_select(0, indices.long()).clone()
        patches.append(SparseBitPatch(indices, values))
        after = score.square().flatten()
        after[indices.long()] = 0
        prior_error = (reference.float() - prior.float()).abs()
        if weight is not None:
            prior_error = prior_error * weight
        records.append(
            dict(
                reference_exception_count=count,
                actual_exception_count=indices.numel(),
                weighted_error_old=float(prior_error.square().sum()),
                weighted_error_refit=float(after.sum()),
            )
        )
    result = replace(packet, base=replace(packet.base, corrections=tuple(patches)))
    if result.component_bytes > packet.component_bytes:
        raise RuntimeError("precision-specific corrections exceeded the same exception bytes")
    return result, dict(
        per_head=per_head,
        components=records,
        original_component_bytes=packet.component_bytes,
        refit_component_bytes=result.component_bytes,
        no_extra_exception_bytes=True,
        no_label_or_output_logit_access=True,
    )


def cpu_packet(packet: HeadLocalFormat) -> HeadLocalFormat:
    """Copy identical encoded bytes; do not requantize at CPU rounding ties."""

    def copied(tensor: Any) -> Any:
        return tensor.detach().cpu().clone()

    def body_cpu(body: Any) -> Any:
        return replace(body, buffers=tuple(copied(t) for t in body.buffers))

    base = packet.base
    return replace(
        packet,
        base=replace(
            base,
            pair=tuple(
                replace(b, codes=copied(b.codes), minimum=copied(b.minimum), scale=copied(b.scale))
                for b in base.pair
            ),
            exact_indices=copied(base.exact_indices),
            exact_map=copied(base.exact_map),
            exact_values=tuple(body_cpu(b) for b in base.exact_values),
            corrections=tuple(
                replace(p, indices=copied(p.indices), values=copied(p.values))
                for p in base.corrections
            ),
        ),
        flat_tokens=copied(packet.flat_tokens),
        protected=tuple(body_cpu(b) for b in packet.protected),
    )


def gpu_controls() -> list[dict[str, Any]]:
    """CPU oracle for 2/3-bit decoding plus bitwise flat-scatter sentinels."""
    from mosaickv.evaluation.head_local_precision_kernel import scatter

    generator = torch.Generator().manual_seed(31)
    pair = tuple(torch.randn(1, 3, 141, 128, generator=generator).bfloat16() for _ in range(2))
    heads = tuple(tuple(range(h, h + 141)) for h in range(3))
    text = {40, 72, 99}
    union = tuple(sorted({j for head in heads for j, p in enumerate(head) if p in text}))
    device_pair = tuple(t.cuda() for t in pair)
    gpu = construct(device_pair, bits=3, protected=union, encoded_exact=True)
    gpu, _ = fund_corrections(gpu, device_pair, fraction=0.005)
    records = []
    for policy in POLICIES:
        if policy == "union3":
            continue
        candidate: Any = localize(gpu, device_pair, heads, text, policy=policy)
        if policy in ("head3k2v_refit", "head3k2v_headwise", "head3k2v_rows"):
            candidate, _ = refit_corrections(
                candidate, device_pair, per_head=policy == "head3k2v_headwise"
            )
        # CPU/CUDA min-max division can round half-bin ties to different codes.
        # Decoder parity requires the SAME encoded packet, not two quantizers.
        if policy == "head3k2v_rows":
            from mosaickv.evaluation.value_row_precision import cpu_row_packet, upgrade_rows

            candidate, _ = upgrade_rows(
                candidate, device_pair, torch.full((1, 3, 16, 143), 1 / 143, device="cuda"), heads
            )
            reference = cpu_row_packet(candidate).materialize()
        else:
            reference = cpu_packet(candidate).materialize()
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
