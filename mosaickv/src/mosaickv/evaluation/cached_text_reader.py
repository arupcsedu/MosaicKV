"""Lossless caching at TEXT-OVERLAY granularity, not whole-layer granularity.

Preserve the original base, selected memberships and attention arithmetic.
Only replace the repeated high-precision text-source projection by exact rows
from that frozen decoder. Slots absent in a head are NEVER inserted into it.
"""

import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product

import torch

from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.hybrid_reader import ReaderOption
from mosaickv.evaluation.hybrid_reader_study import timed
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.text_source_storage import TextSourceFormat


@dataclass(frozen=True)
class CachedTextFormat(HeadLocalFormat):
    kind: str = "lossless_exact_text_overlay_execution_variant"

    def shared(self):
        # HeadLocalFormat's original base is PrecisionFormat (no shared data).
        # Here a shared-source base MUST retain/count its original rotary data.
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def materialize(self, attention):
        targets = tuple(
            torch.empty(
                self.shape, device=self.protected[0].buffers[0].device, dtype=torch.bfloat16
            )
            for _ in range(2)
        )
        self.fill(attention, targets, direct=True)
        return targets


def eligible(packet):
    return isinstance(packet, TextSourceFormat) and bool((packet.projection.slots >= 0).any())


@torch.inference_mode()
def build(packet, attention, modes):
    from mosaickv.evaluation.fused_lossless_attention import bits_equal
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    if modes == ("source", "source"):
        return packet
    if (
        not eligible(packet)
        or type(modes) is not tuple
        or len(modes) != 2
        or any(mode not in ("encoded", "resident") for mode in modes)
    ):
        raise ValueError("eligible text packet and two explicit lossless modes required")
    heads, length, dim = packet.shape[1:]
    slots = packet.projection.slots
    valid = slots >= 0
    if slots.ndim != 2 or slots.shape[0] != heads or bool((slots[valid] >= length).any()):
        raise ValueError("original valid head-local text slot map required")
    head_ids = torch.arange(heads, device=slots.device, dtype=torch.int32)[:, None].expand_as(slots)
    flat = (head_ids * length + slots.to(torch.int32))[valid].contiguous()
    if flat.numel() != flat.unique().numel() or not bool((flat[1:] > flat[:-1]).all()):
        raise ValueError("sorted unique per-head text memberships required")
    body = []
    for component, mode in enumerate(modes):
        original = torch.empty(packet.shape, device=slots.device, dtype=torch.bfloat16)
        fill_component(packet, attention, component, original)
        rows = original[0, head_ids[valid].long(), slots[valid].long()].reshape(1, 1, -1, dim)
        rows = rows.contiguous()
        encoded = (
            encode(rows) if mode == "encoded" else PackedBF16(tuple(rows.shape), True, (rows,))
        )
        from mosaickv.evaluation.exact_storage import decoded

        if not bits_equal(decoded(encoded), rows):
            raise RuntimeError("text-row storage changed an original decoder bit")
        body.append(encoded)
    return CachedTextFormat(packet.base, flat, tuple(body))


def convert(packets, attentions, policy):
    if (
        set(policy) != {"modes", "maximum_added_bytes", "base_private_bytes"}
        or type(policy["maximum_added_bytes"]) is not int
        or not 0 <= policy["maximum_added_bytes"] <= 6 * 2**20
        or type(policy["base_private_bytes"]) is not int
        or len(packets) != len(attentions)
        or len(packets) != len(policy["modes"])
    ):
        raise ValueError("complete bounded cached-text policy required; labels forbidden")
    old = backing_bytes(tuple(t for p in packets for t in p.private()))
    if old != policy["base_private_bytes"]:
        raise ValueError("cached-text policy bound to different input storage")
    result = tuple(
        build(p, a, tuple(m)) for p, a, m in zip(packets, attentions, policy["modes"], strict=True)
    )
    current = backing_bytes(tuple(t for p in result for t in p.private()))
    if current > old + policy["maximum_added_bytes"]:
        raise RuntimeError("actual cached-text backing exceeds its hard allowance")
    return result, dict(
        original_private_bytes=old,
        constructed_private_bytes=current,
        maximum_added_bytes=policy["maximum_added_bytes"],
        cached_overlay_layers=[i for i, p in enumerate(result) if isinstance(p, CachedTextFormat)],
        original_KV_bits_preserved=True,
        no_labels_or_query_feedback=True,
    )


@torch.inference_mode()
def profile(packets, attentions, *, seed):
    from mosaickv.evaluation.fused_lossless_attention import bits_equal
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    def identity(t):
        return str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()

    owners = Counter(k for p in packets for k in {identity(t) for t in p.private()})
    shared = {k for k, n in owners.items() if n > 1}

    def cost(p):
        return backing_bytes(tuple(t for t in p.private() if identity(t) not in shared))

    rng, routes, raw = random.Random(seed), [], []
    for index, (packet, attention) in enumerate(zip(packets, attentions, strict=True)):
        if not eligible(packet):
            # Excluded constant objective term, NEVER a measured zero latency.
            routes.append([asdict(ReaderOption(("source", "source"), cost(packet), 0, 0))])
            continue
        choices = [("source", "source"), *product(("encoded", "resident"), repeat=2)]
        rng.shuffle(choices)
        options = []
        for modes in choices:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            candidate = build(packet, attention, modes)
            torch.cuda.synchronize()
            construction = time.perf_counter_ns() - started if modes != ("source", "source") else 0
            observations = []
            for padding in (0, 64, 127):
                shape = (*packet.shape[:2], packet.shape[-2] + padding, packet.shape[-1])
                target = torch.full(
                    shape, -71, device=packet.private()[0].device, dtype=torch.bfloat16
                )
                oracle = torch.empty_like(target)
                for component in (0, 1):
                    fill_component(packet, attention, component, oracle)
                    fill_component(candidate, attention, component, target)
                    if not bits_equal(
                        oracle[..., : packet.shape[-2], :], target[..., : packet.shape[-2], :]
                    ) or not bool((target[..., packet.shape[-2] :, :] == -71).all()):
                        raise RuntimeError("cached text changed parent K/V or history sentinel")

                def call(candidate=candidate, target=target, attention=attention):
                    for component in (0, 1):
                        fill_component(candidate, attention, component, target)

                observations.extend(dict(padding=padding, **r) for r in timed(call))
                del target, oracle
            option = ReaderOption(
                modes,
                cost(candidate),
                construction,
                int(sum(x["wall_ns"] for x in observations) / len(observations)),
            )
            options.append(asdict(option))
            raw.append(
                dict(
                    layer=index,
                    modes=modes,
                    trials=observations,
                    parent_KV_word_exact=True,
                    history_sentinels_unchanged=True,
                )
            )
            del candidate
        routes.append(options)
    return dict(
        routes=routes,
        raw=raw,
        base_private_bytes=backing_bytes(tuple(t for p in packets for t in p.private())),
        noneligible_routes_excluded_constant_objective_not_measured_zero=True,
        operator_times_not_end_to_end_evidence=True,
        no_label_or_answer_access=True,
    )
