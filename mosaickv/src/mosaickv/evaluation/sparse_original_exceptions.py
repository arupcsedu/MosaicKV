"""Bounded exact precision exceptions; no recoding, labels, offload or repair.

Different physical-memory points, not a byte-matched published-baseline claim.
Common to all local precision adaptations. Selected/merged values stay intact.
"""

from dataclasses import dataclass

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_source_format import apply_transport_patch


@dataclass(frozen=True)
class ExceptionFormat:
    base: object
    patches: tuple
    kind: str = "common_sparse_exact_precision_exceptions_variant"

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), *(t for p in self.patches for t in p.tensors))

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.base.fill(attention, targets, direct=direct)
        for patch, target in zip(self.patches, targets, strict=True):
            apply_transport_patch(patch, target, prompt_length=self.shape[-2])

    def materialize(self, attention=None):
        outputs = self.base.materialize(attention)
        for patch, target in zip(self.patches, outputs, strict=True):
            apply_transport_patch(patch, target, prompt_length=self.shape[-2])
        return outputs


@torch.inference_mode()
def construct(base, reference, actual, weights, *, budget_bytes):
    """Exact exceptions ranked by w*error^2; six bytes per int32/BF16 pair."""
    if (
        type(budget_bytes) is not int
        or budget_bytes < 0
        or len(reference) != 2
        or len(actual) != 2
        or len(weights) != 2
    ):
        raise ValueError("nonnegative physical bytes and two actual KV components required")
    if any(
        a.shape != base.shape
        or b.shape != a.shape
        or w.shape != a.shape
        or a.dtype != torch.bfloat16
        or b.dtype != a.dtype
        or any(t.device != a.device or not torch.isfinite(t).all() for t in (a, b, w))
        or (w < 0).any()
        for a, b, w in zip(reference, actual, weights, strict=True)
    ):
        raise ValueError(
            "finite compatible original/merged BF16 KV and nonnegative importance required"
        )
    scores = torch.cat(
        tuple(
            (w * (a.float() - b.float()).square()).flatten()
            for a, b, w in zip(reference, actual, weights, strict=True)
        )
    )
    count = min(budget_bytes // 6, int((scores > 0).sum()))
    chosen = torch.empty(0, device=scores.device, dtype=torch.long)
    if count:
        cutoff = torch.topk(scores, count, sorted=False).values.min()
        above = (scores > cutoff).nonzero().flatten()
        ties = (scores == cutoff).nonzero().flatten()[: count - above.numel()]
        chosen = torch.cat((above, ties)).sort().values
    patches = []
    offset = 0
    for original in reference:
        ids = chosen[(chosen >= offset) & (chosen < offset + original.numel())] - offset
        if original.numel() >= 2**31:
            raise ValueError("int32 patch geometry exceeded")
        patches.append(
            SparseBitPatch(ids.to(torch.int32), original.contiguous().flatten()[ids].clone())
        )
        offset += original.numel()
    used = backing_bytes(tuple(t for patch in patches for t in patch.tensors))
    if used > budget_bytes:
        raise RuntimeError("exact-exception physical budget exceeded")
    packet = ExceptionFormat(base, tuple(patches))
    return packet, dict(
        restored_scalars=count,
        exact_exception_bytes=used,
        budget_bytes=budget_bytes,
        weighted_error_before=float(scores.sum()),
        weighted_error_removed=float(scores[chosen].sum()),
        deterministic_cutoff_ties=True,
        body_and_existing_exact_values_unchanged=True,
        no_original_cache_alias=True,
        training_free=True,
    )


@torch.inference_mode()
def formats(
    adapter,
    prefill,
    capture,
    parents,
    original_pairs,
    memberships,
    forecast,
    *,
    cap_bytes,
    maximum_extra_bytes=2**20,
):
    from mosaickv.evaluation.lossless_affine_minimum import encode_format
    from mosaickv.evaluation.query_affine_formats import prompt_query
    from mosaickv.evaluation.source_query_importance import blend_families, kv_importance_cuda

    if adapter.device.type != "cuda":
        raise NotImplementedError("actual CUDA constructor required")
    original_parent_bytes = backing_bytes(tuple(t for f in parents for t in f.tensors()))
    original_parents = parents
    parents = tuple(encode_format(f) for f in parents)
    base_bytes = backing_bytes(tuple(t for f in parents for t in f.tensors()))
    if maximum_extra_bytes != 2**20 or cap_bytes < base_bytes:
        raise ValueError("fixed one-MiB allowance inside unchanged own hard cap required")
    per_layer = min(maximum_extra_bytes, cap_bytes - base_bytes) // len(parents)
    results = []
    proof = []
    n = prefill.state.logical_sequence_length
    for i, (base, pair, positions, block) in enumerate(
        zip(parents, original_pairs, memberships, adapter._language_layers(), strict=True)
    ):
        h, length = pair[0].shape[1:3]
        logical = torch.tensor(positions, device=adapter.device)
        exact = torch.zeros(h, length, device=adapter.device, dtype=torch.bool)
        query = prompt_query(block.self_attn, capture.records[i], h)
        prompt, _ = kv_importance_cuda(
            query,
            *pair,
            logical,
            torch.arange(n - query.shape[-2], n, device=adapter.device),
            exact,
        )
        future = forecast[i]
        r = future["query"].shape[-2]
        joint = tuple(
            torch.cat((a, b), dim=2) for a, b in zip(pair, future["history"], strict=True)
        )
        future_logical = torch.cat(
            (logical, torch.arange(n, n + r, device=adapter.device)[None].expand(h, r)), dim=1
        )
        future_mask = torch.cat(
            (exact, torch.ones(h, r, device=adapter.device, dtype=torch.bool)), dim=1
        )
        draft, _ = kv_importance_cuda(
            future["query"],
            *joint,
            future_logical,
            torch.arange(n, n + r, device=adapter.device),
            future_mask,
        )
        weights = tuple(
            blend_families(a, b[..., :length, :], query.shape[-2], r)
            for a, b in zip(prompt, draft, strict=True)
        )
        actual = base.materialize(block.self_attn)
        old_values = original_parents[i].materialize(block.self_attn)
        if not all(
            torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
            for a, b in zip(actual, old_values, strict=True)
        ):
            raise RuntimeError("minimum encoding changed a parent KV bit")
        del old_values
        packet, record = construct(base, pair, actual, weights, budget_bytes=per_layer)
        verified = packet.materialize(block.self_attn)
        for patch, value, old, original in zip(packet.patches, verified, actual, pair, strict=True):
            expected = old.contiguous().clone().flatten()
            expected[patch.indices.long()] = original.contiguous().flatten()[patch.indices.long()]
            if not torch.equal(
                value.contiguous().view(torch.int16).flatten(), expected.view(torch.int16)
            ):
                raise RuntimeError(
                    "exact exception changed an unselected scalar or failed restoration"
                )
        results.append(packet)
        proof.append(dict(layer=i, **record, actual_CUDA_bitwise_patch_verification=True))
    if backing_bytes(tuple(t for f in results for t in f.tensors())) > cap_bytes:
        raise RuntimeError("constructed exceptions exceed unchanged cap")
    return tuple(results), dict(
        layers=proof,
        parent_bytes=base_bytes,
        original_parent_bytes=original_parent_bytes,
        common_lossless_minimum_savings_bytes=original_parent_bytes - base_bytes,
        FP32_scale_unchanged=True,
        every_layer_KV_bitwise_before_exceptions=True,
        maximum_extra_bytes=maximum_extra_bytes,
        per_layer_allowance=per_layer,
        common_to_all_baselines=True,
        no_benchmark_labels_or_saved_reference_logits=True,
    )
