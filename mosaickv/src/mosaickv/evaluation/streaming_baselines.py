"""Stream the FROZEN local baseline precision variants, never their substitutes.

PrefixKV's profile and Top-K/LOOK-M capacities are known before prefill. VL-Cache
needs ALL layer densities: keep a temporary LOSSLESS prompt archive, not guessed
early capacities, then construct one selected layer at a time. No second prefill.
All final packets must pass an independent offline-payload/model-output gate.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from mosaickv.evaluation.projection_cache import ScratchAudit, identity
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache


@dataclass(frozen=True)
class SelectedEncoding:
    packet: object
    memberships: tuple[tuple[int, ...], ...]


class SelectedRetiringCache(RetiringPrefillCache):
    """Explicit selected-slot geometry; original logical positions never shrink."""

    def __init__(self, layer_count, *, audit=True):
        super().__init__(layer_count, audit=audit)
        self.memberships = {}

    def seal(self, index, *args, **kwargs):
        try:
            return super().seal(index, *args, **kwargs)
        except BaseException:
            self.memberships.pop(index, None)
            raise

    def validate_encoding(self, index, produced, original):
        if not isinstance(produced, SelectedEncoding):
            return super().validate_encoding(index, produced, original)
        packet, heads = produced.packet, produced.memberships
        b, h, n, d = original[0].shape
        shape = tuple(packet.shape)
        if (
            len(shape) != 4
            or shape[:2] != (b, h)
            or shape[-1] != d
            or not 1 <= shape[-2] <= n
            or len(heads) != h
            or any(
                len(row) != shape[-2]
                or any(type(p) is not int or not 0 <= p < n for p in row)
                or tuple(sorted(set(row))) != row
                or n - 1 not in row
                for row in heads
            )
        ):
            raise ValueError(
                "explicit monotonic selected-slot mapping including mandatory token required"
            )
        self.memberships[index] = heads
        return packet

    def finalize_lossless_layer(self, index, attention, build):
        """Charge pending archive + materialized layer + new packet before release."""
        from mosaickv.evaluation.exact_storage import ExactFormat

        if self.next_layer != len(self.layers) or index in self.memberships:
            raise RuntimeError("complete one-pass archive and unfinished selected layer required")
        layer = self.layers[index]
        old = layer.packet
        if (
            not isinstance(old, ExactFormat)
            or old.kind != "kv"
            or old.shape[-2] != layer.length
            or len(old.pair) != 2
            or old.archive is not None
            or old.slots is not None
            or old.patches
        ):
            raise ValueError("only full original lossless KV archive may be finalized")
        excluded = {identity(t) for t in self.owned_tensors()}
        excluded.update(identity(t) for t in attention.parameters())
        mode = (
            ScratchAudit(excluded=excluded, observe=self.observe) if self.audit else nullcontext()
        )
        with mode:
            original = old.materialize(attention)
            proposed = build(original)
            from mosaickv.evaluation.deferred_exact_encoding import finalize

            proposed = finalize(proposed)
            packet = self.validate_encoding(index, proposed, original)
            if any(identity(t) in {identity(x) for x in original} for t in packet.tensors()):
                self.memberships.pop(index, None)
                raise RuntimeError("selected archive aliases decoded temporary source")
            self.observe((*original, *packet.tensors()))
        layer.packet = packet
        self.layer_owners_changed(index)
        self.observe(original)


def source_capacities(profile, length, *, heads=32):
    from mosaickv.selection.profile_lifetime import allocate_profile_lifetime

    count, cost = len(profile), heads * 128 * 2 * 2
    from mosaickv.evaluation.paper_geometry import odd_length_allowed

    if not count or (length % 2 and not odd_length_allowed(length)):
        raise NotImplementedError("audited even-length BF16 full-MHA prompts only")
    capacity = allocate_profile_lifetime(
        tuple(profile),
        (length,) * count,
        (1,) * count,
        (cost,) * count,
        cap_bytes=int(count * length * cost * 0.5),
        generated_capacity=0,
    )
    return capacity.prompt_positions


def prefix_capacities(profile, length, target):
    from mosaickv.baselines.prefixkv import _bounded_apportion

    # Frozen local configuration: offline raw profile, global_apportion,
    # start/protect=1, legacy_clamped, append_only, no evidence adaptations.
    return _bounded_apportion(
        tuple(length * (1 - value) for value in profile),
        (2,) * len(profile),
        (length,) * len(profile),
        target,
    )


def vl_capacities(statistics, length, target):
    from mosaickv.baselines.vl_cache import _bounded_apportion, paper_layer_retention_ratios
    from mosaickv.config import VLCacheConfig

    config = VLCacheConfig(enabled=True)
    _, ratios = paper_layer_retention_ratios(
        tuple(s.layer_sparsity[0] for s in statistics),
        retention_ratio=target / (len(statistics) * length),
        minimum=config.min_layer_retention,
        maximum=config.max_layer_retention,
    )
    desired = tuple(r * length for r in ratios)
    scale = target / sum(desired)
    lower = max(1, math.floor(config.min_layer_retention * length + 1e-12))
    upper = min(length, max(lower, math.floor(config.max_layer_retention * length + 1e-12)))
    return _bounded_apportion(
        tuple(v * scale for v in desired),
        (lower,) * len(statistics),
        (upper,) * len(statistics),
        target,
    )


def selected_pair(original, heads):
    index = torch.tensor(heads, device=original[0].device, dtype=torch.long)[None, :, :, None]
    return tuple(
        t.gather(2, index.expand(1, t.shape[1], index.shape[2], t.shape[-1])) for t in original
    )


def encode_layer(index, count, attention, pair, heads, text_positions, record, probabilities, text):
    """Exact operation order of head3_boundary8 then text-source optimize."""
    from mosaickv.evaluation.direct_text_construction import optimize
    from mosaickv.evaluation.head_local_precision import localize, refit_corrections
    from mosaickv.evaluation.precision_corrections import fund_corrections, prompt_weights
    from mosaickv.evaluation.precision_tier import construct

    union = tuple(sorted({j for row in heads for j, p in enumerate(row) if p in text_positions}))
    boundary = index < 2 or index >= count - 2
    packet = construct(pair, bits=8 if boundary else 3, protected=union, encoded_exact=True)
    kw, vw = prompt_weights(attention, record, probabilities, heads, window=16)
    packet, _ = fund_corrections(packet, pair, fraction=0.005, key_weight=kw, value_weight=vw)
    packet = localize(
        packet,
        pair,
        heads,
        set(text_positions),
        policy="head3_boundary8",
        boundary_override=boundary,
    )
    if boundary:
        # The frozen boundary branch refits AFTER head-local exact overlays.
        packet, _ = refit_corrections(packet, pair, key_weight=kw, value_weight=vw)
    packet, _ = optimize(packet, text, heads, text_positions, attention, reference_pair=pair)
    return packet


def lookm_layer(pair, probabilities, prepared, capacity):
    """Use the ORIGINAL public reimplementation and actual merged K/V."""
    from mosaickv.adapters.huggingface.types import (
        CachedKeyState,
        CacheLayerSnapshot,
        CacheSnapshot,
    )
    from mosaickv.baselines.lookm import build_lookm_reimpl_plan
    from mosaickv.cache_state import FullKVState
    from mosaickv.config import CacheConfig, LookMConfig
    from mosaickv.types import BudgetUnit

    n, h = pair[0].shape[-2], pair[0].shape[1]
    full = FullKVState.from_cache_snapshot(
        CacheSnapshot(
            (CacheLayerSnapshot(*pair, 2),), tuple, "legacy", n, CachedKeyState.POST_ROPE
        ),
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
        block_size=1,
        original_logical_sequence_length=n,
        next_decode_position=n,
        mandatory_logical_positions=(n - 1,),
    )
    cache = CacheConfig(
        block_size=1,
        budget_unit=BudgetUnit.BYTES,
        budget_value=capacity * h * 128 * 4,
        retention_ratio=capacity / n,
        physical_budget_mandatory_policy="last_token",
    )
    plan = build_lookm_reimpl_plan(
        full,
        (probabilities,),
        LookMConfig(enabled=True),
        cache,
        physical_layer_capacities=(capacity,),
    )
    heads = tuple(head.selected_physical_positions for head in plan.heads)
    merged = tuple(
        torch.cat([getattr(head, component) for head in plan.heads], 1)
        for component in ("key", "value")
    )
    return heads, merged


class StreamingBaseline:
    """A one-request, label-blind encoder for explicitly frozen local variants."""

    def __init__(self, prepared, cache, profile, method, *, minimum_only=False):
        from mosaickv.evaluation.text_source_storage import TextSourceCapture

        if method not in (
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
        ):
            raise ValueError("registered four baseline precision variants only")
        if type(minimum_only) is not bool:
            raise ValueError("explicit lossless-minimum storage flag required")
        self.minimum_only = minimum_only
        self.prepared, self.cache, self.method = prepared, cache, method
        self.text_positions = TextSourceCapture(None, prepared).positions
        self.length, self.count = prepared.logical_sequence_length, len(profile)
        if cache is None or (hasattr(cache, "layers") and len(cache.layers) != self.count):
            raise ValueError("complete original decoder layer count required")
        original = source_capacities(profile, self.length)
        self.target = sum(original)
        low, extra = divmod(self.target, self.count)
        self.capacities = (
            prefix_capacities(profile, self.length, self.target)
            if method == "prefixkv_reimpl"
            else tuple(low + int(i < extra) for i in range(self.count))
        )
        media = [s.end for s in prepared.modality_map if s.modality in ("image", "video")]
        self.vision_end = max(media) if media else None
        if method == "vl_cache_reimpl" and (
            self.vision_end is None or self.vision_end >= self.length
        ):
            raise ValueError("authoritative visual span and post-vision language required")
        self.pending, self.statistics, self.memberships = {}, [], []
        self.phases, self.phase_identity, self.ids = None, None, None
        self.finished = False

    def _extra(self):
        self.cache.extra_tensors = (
            *(() if self.ids is None else (self.ids,)),
            *(() if self.phases is None else self.phases),
            *tuple(
                t
                for record, probabilities, text in self.pending.values()
                for t in (*record.values(), probabilities, *text.values())
            ),
        )

    def _heads_pair(self, index, original, probabilities):
        from mosaickv.baselines.prefixkv import _select_positions, prefixkv_attention_scores
        from mosaickv.evaluation.request_precision import ranked_positions

        count, n, h = self.capacities[index], self.length, original[0].shape[1]
        if self.method == "lookm_reimpl":
            return lookm_layer(original, probabilities, self.prepared, count)
        if self.method == "prompt_attention_topk":
            heads = ranked_positions(probabilities[0, :, -16:, :].float().mean(-2), count, (n - 1,))
        elif self.method == "prefixkv_reimpl":
            selected, _ = _select_positions(
                prefixkv_attention_scores((probabilities,))[0], count, 1, 1
            )
            heads = (selected,) * h
        else:
            from mosaickv.baselines.vl_cache import _select_positions as vl_select

            heads = tuple(
                vl_select(s, count, (n - 1,), 0.1)[0]
                for s in self.statistics[index].kv_head_token_scores[0]
            )
        return heads, selected_pair(original, heads)

    def _encode(self, *args):
        packet = encode_layer(*args)
        if self.minimum_only:
            from mosaickv.evaluation.exact_minimum_metadata import compact_format

            packet = compact_format(packet)
        return packet

    def __call__(self, index, attention, original, args, kwargs, output):
        from mosaickv.evaluation.streaming_prefill import lossless_kv_encoder

        if self.finished or index != len(self.memberships):
            raise ValueError("one ordered prefill layer per baseline encoder required")
        x, phases, probabilities = (
            kwargs.get("hidden_states", args[0] if args else None),
            kwargs.get("position_embeddings"),
            output[1],
        )
        n, h = self.length, original[0].shape[1]
        if (
            x is None
            or phases is None
            or x.shape != (1, n, 4096)
            or x.dtype != torch.bfloat16
            or h != 32
            or probabilities.shape != (1, h, n, n)
            or probabilities.dtype != torch.bfloat16
            or probabilities.device != x.device
        ):
            raise NotImplementedError("pinned complete BF16 LLaVA-1.5 eager prefill only")
        phase_identity = tuple((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in phases)
        if self.phases is None:
            self.ids = torch.tensor(self.text_positions, device=x.device)
            self.phases = tuple(p.index_select(1, self.ids).contiguous() for p in phases)
            self.phase_identity = phase_identity
        elif phase_identity != self.phase_identity:
            raise ValueError("original shared rotary phases changed across layers")
        record = dict(
            normalized=x[:, -16:].detach().clone(),
            cos=phases[0][:, -16:].detach().clone(),
            sin=phases[1][:, -16:].detach().clone(),
        )
        text = dict(
            normalized=x.index_select(1, self.ids).detach().contiguous(),
            cos=self.phases[0],
            sin=self.phases[1],
        )
        self._extra()
        if self.method == "vl_cache_reimpl":
            from mosaickv.baselines.vl_cache import post_vision_attention_statistics

            self.statistics.append(
                post_vision_attention_statistics(
                    (probabilities,), kv_heads_by_layer=(h,), post_vision_start=self.vision_end
                )
            )
            self.pending[index] = record, probabilities[..., -16:, :].detach().clone(), text
            self._extra()
            self.memberships.append(None)
            return lossless_kv_encoder(index, attention, original, args, kwargs, output)
        heads, pair = self._heads_pair(index, original, probabilities)
        packet = self._encode(
            index,
            self.count,
            attention,
            pair,
            heads,
            self.text_positions,
            record,
            probabilities,
            text,
        )
        self.memberships.append(heads)
        return SelectedEncoding(packet, heads)

    def finish(self, adapter):
        if self.finished or len(self.memberships) != self.count:
            raise RuntimeError("all original prefill layers must be processed exactly once")
        if self.method == "vl_cache_reimpl":
            self.capacities = vl_capacities(self.statistics, self.length, self.target)
            for i, block in enumerate(adapter._language_layers()):
                record, probabilities, text = self.pending[i]

                def build(
                    original,
                    *,
                    index=i,
                    probabilities=probabilities,
                    block=block,
                    record=record,
                    text=text,
                ):
                    heads, pair = self._heads_pair(index, original, probabilities)
                    packet = self._encode(
                        index,
                        self.count,
                        block.self_attn,
                        pair,
                        heads,
                        self.text_positions,
                        record,
                        probabilities,
                        text,
                    )
                    self.memberships[index] = heads
                    return SelectedEncoding(packet, heads)

                self.cache.finalize_lossless_layer(i, block.self_attn, build)
                del self.pending[i]
                self._extra()
        self.ids, self.finished = None, True
        self.cache.extra_tensors = ()
