"""Coherent exact input columns shared by frozen K and V projections.

Diagnostic precision variant, not the six-component exact-selection method.
No label fitting, weight updates, additional prefill, prototype or repair.
"""

from dataclasses import dataclass

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class ChannelExceptions:
    lookup: object  # uint8[width], 255 means quantized source
    values: object  # original BF16[channel slot, original position]

    def __post_init__(self):
        if (
            self.lookup.ndim != 1
            or self.lookup.dtype != torch.uint8
            or self.values.ndim != 2
            or self.values.dtype != torch.bfloat16
            or self.values.device != self.lookup.device
            or self.values.shape[0] > 254
            or not self.lookup.is_contiguous()
            or not self.values.is_contiguous()
            or not torch.isfinite(self.values).all()
        ):
            raise ValueError(
                "contiguous same-device finite BF16 column values and uint8 map required"
            )
        slots = self.lookup[self.lookup != 255]
        if slots.numel() != self.values.shape[0] or not torch.equal(
            slots.sort().values.long(), torch.arange(slots.numel(), device=slots.device)
        ):
            raise ValueError("coherent source map must cover each stored column once")

    def tensors(self):
        return self.lookup, self.values


@torch.inference_mode()
def construct(original, decoded, importance, *, budget_bytes):
    if (
        type(budget_bytes) is not int
        or budget_bytes < 0
        or original.ndim != 4
        or original.shape[:2] != (1, 1)
        or not all(original.shape)
        or original.dtype != torch.bfloat16
        or decoded.dtype != original.dtype
        or decoded.shape != original.shape
        or importance.shape != original.shape
        or any(
            t.device != original.device or not torch.isfinite(t).all()
            for t in (decoded, importance)
        )
        or not torch.isfinite(original).all()
        or (importance < 0).any()
    ):
        raise ValueError(
            "finite BF16 sources, nonnegative sensitivities and physical budget required"
        )
    n, width = original.shape[2:]
    count = min(254, max(0, (budget_bytes - width) // (2 * n)))
    score = (importance * (original.float() - decoded.float()).square()).sum((0, 1, 2))
    count = min(count, int((score > 0).sum()))
    if not count:
        return None, dict(
            channels=0,
            exception_bytes=0,
            budget_bytes=budget_bytes,
            weighted_input_error_removed=0.0,
            weighted_input_error_before=float(score.sum()),
            source_codes_unchanged=True,
            original_positions_unchanged=True,
            K_and_V_share_corrected_input=True,
        )
    chosen = torch.argsort(score, descending=True, stable=True)[:count].sort().values
    lookup = torch.full((width,), 255, device=original.device, dtype=torch.uint8)
    lookup[chosen] = torch.arange(count, device=original.device, dtype=torch.uint8)
    values = original[0, 0].index_select(1, chosen).T.contiguous().clone()
    result = ChannelExceptions(lookup, values)
    used = backing_bytes(result.tensors())
    if used > budget_bytes:
        raise RuntimeError("coherent exception allowance exceeded")
    return result, dict(
        channels=count,
        exception_bytes=used,
        budget_bytes=budget_bytes,
        selected_channels=chosen.cpu().tolist(),
        weighted_input_error_removed=float(score[chosen].sum()),
        weighted_input_error_before=float(score.sum()),
        source_codes_unchanged=True,
        original_positions_unchanged=True,
        K_and_V_share_corrected_input=True,
        no_original_source_alias=True,
    )


def restore_cpu(exceptions, source):
    if exceptions is None:
        return source
    if source.device.type != "cpu" or source.ndim != 3 or source.shape[0] != 1:
        raise ValueError("CPU source oracle only")
    ids = (exceptions.lookup != 255).nonzero().flatten()
    source[0].index_copy_(1, ids, exceptions.values[exceptions.lookup[ids].long()].T)
    return source


def kernel_arguments(packet):
    exceptions = getattr(packet, "channel_exceptions", None)
    if exceptions is None:
        return packet.source.codes, packet.source.codes, False
    n, width = packet.source.shape[2:]
    if (
        exceptions.lookup.shape != (width,)
        or exceptions.lookup.dtype != torch.uint8
        or exceptions.values.dtype != torch.bfloat16
        or exceptions.values.ndim != 2
        or exceptions.values.shape[1] != n
        or exceptions.values.shape[0] > 254
        or not exceptions.lookup.is_contiguous()
        or not exceptions.values.is_contiguous()
        or any(t.device != packet.source.codes.device for t in exceptions.tensors())
    ):
        raise ValueError("audited same-device contiguous coherent-column buffers required")
    # Contents are validated exactly once by ChannelExceptions construction.
    # No tensor reductions, host synchronization, or model-wide scan in decode.
    return exceptions.lookup, exceptions.values, True


@torch.inference_mode()
def formats(
    adapter,
    prefill,
    capture,
    source_capture,
    text_capture,
    phase_capture,
    parents,
    forecast,
    *,
    head_group,
    cap_bytes,
):
    from dataclasses import replace

    from mosaickv.evaluation.lossless_affine_minimum import encode_format
    from mosaickv.evaluation.query_affine_formats import prompt_query
    from mosaickv.evaluation.shared_source_format import fit_transport
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction
    from mosaickv.evaluation.source_query_importance import (
        blend_families,
        future_importance_cuda,
        importance_cuda,
    )
    from mosaickv.evaluation.source_retile import retile
    from mosaickv.evaluation.text_source_storage import optimize

    if adapter.device.type != "cuda":
        raise NotImplementedError("actual CUDA coherent construction only")
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    results = []
    records = []
    for i, (parent, pair, block) in enumerate(
        zip(parents, original, adapter._language_layers(), strict=True)
    ):
        h, n = pair[0].shape[1:3]
        mask = torch.zeros(h * n, device=adapter.device, dtype=torch.bool)
        mask[parent.row_positions.long()] = True
        mask = mask.reshape(h, n)
        q = prompt_query(block.self_attn, capture.records[i], h)
        weight, wp = importance_cuda(
            q,
            *pair,
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
            torch.arange(n, device=adapter.device).repeat(h, 1),
            torch.arange(n - q.shape[-2], n, device=adapter.device),
            phase_capture.phases,
            mask,
        )
        future = forecast[i]
        fw = future_importance_cuda(
            future["query"],
            pair[1],
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
            phase_capture.phases,
            mask,
            future["probability"],
            future["output"],
        )
        weight = blend_families(weight, fw, q.shape[-2], future["query"].shape[-2])
        decoded = decode_source_for_construction(parent)[:, None]
        exceptions, proof = construct(
            source_capture.records[i][:, None],
            decoded,
            weight,
            budget_bytes=(2**20) // len(parents),
        )
        packet = replace(parent, channel_exceptions=exceptions, transport_patches=())
        packet, numerical = fit_transport(packet, block.self_attn)
        packet, text_proof = optimize(
            packet,
            text_capture.records[i],
            (tuple(range(n)),) * h,
            text_capture.positions,
            block.self_attn,
        )
        packet = encode_format(retile(packet, head_group))
        actual = packet.materialize(block.self_attn)
        ids = parent.row_positions.long()
        head, pos = ids // n, ids % n
        for value, reference in zip(actual, pair, strict=True):
            if not torch.equal(
                value[0, head, pos].contiguous().view(torch.int16),
                reference[0, head, pos].contiguous().view(torch.int16),
            ):
                raise RuntimeError("coherent input exception corrupted protected original KV")
        results.append(packet)
        records.append(
            dict(
                layer=i,
                **proof,
                importance=wp,
                numerical=numerical,
                text=text_proof,
                protected_original_KV_bitwise=True,
            )
        )
    size = backing_bytes(tuple(t for f in results for t in f.tensors()))
    if size > cap_bytes:
        raise RuntimeError("coherent input exceptions exceed unchanged own physical cap")
    return tuple(results), dict(
        layers=records,
        initial_bytes=size,
        cap_bytes=cap_bytes,
        maximum_exception_bytes=2**20,
        channels=sum(x["channels"] for x in records),
        header_encoding="lossless_bf16_minimum_fp32_scale",
        quantized_codes_unchanged=True,
        no_benchmark_label_or_saved_logit_access=True,
    )


@torch.inference_mode()
def gpu_controls():
    from dataclasses import replace
    from types import SimpleNamespace

    from mosaickv.evaluation.lossless_affine_minimum import encode_format
    from mosaickv.evaluation.precision_tier import unpack_cpu
    from mosaickv.evaluation.shared_source_format import construct as source
    from mosaickv.evaluation.shared_source_format import fit_transport
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

    if not torch.cuda.is_available():
        raise NotImplementedError("actual CUDA only")
    torch.manual_seed(20261101)
    h, n, d = 4, 40, 128
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    x = torch.randn(1, n, h * d, device="cuda").bfloat16()
    original = tuple(
        m(x).reshape(1, n, h, d).transpose(1, 2).contiguous()
        for m in (attention.k_proj, attention.v_proj)
    )
    phases = (
        torch.ones(1, n, d, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, n, d, device="cuda", dtype=torch.bfloat16),
    )
    packet = source(
        x,
        original,
        (tuple(range(n)),) * h,
        (0, n - 1),
        phases,
        (
            torch.arange(n, device="cuda", dtype=torch.int32),
            torch.arange(n, device="cuda", dtype=torch.int16)[None].expand(h, n),
        ),
        exclude_global_exact=True,
    )
    decoded = decode_source_for_construction(packet)
    body = replace(
        packet.source,
        codes=packet.source.codes.cpu(),
        minimum=packet.source.minimum.cpu(),
        scale=packet.source.scale.cpu(),
    )
    cpu = unpack_cpu(body).squeeze(1)
    if not torch.equal(cpu, decoded.cpu()):
        raise RuntimeError("zero-exception input transport drift")
    ex, proof = construct(
        x[:, None],
        decoded[:, None],
        torch.ones_like(x[:, None], dtype=torch.float32),
        budget_bytes=4096,
    )
    expected = restore_cpu(ChannelExceptions(ex.lookup.cpu(), ex.values.cpu()), cpu.clone())
    trial = replace(packet, channel_exceptions=ex)
    if not torch.equal(expected, decode_source_for_construction(trial).cpu()):
        raise RuntimeError("fused coherent input mismatch")
    trial, checks = fit_transport(trial, attention)
    grouped = replace(trial, source_head_group=4)
    baseline_values = trial.materialize(attention)
    for a, b in zip(baseline_values, grouped.materialize(attention), strict=True):
        if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
            raise RuntimeError("coherent h4 projection changed h1 bits")
    targets = tuple(
        torch.full((1, h, n + 5, d), -71, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    grouped.fill(attention, targets, direct=True)
    for actual, expected in zip(targets, baseline_values, strict=True):
        if not torch.equal(actual[..., :n, :], expected) or not (actual[..., n:, :] == -71).all():
            raise RuntimeError("coherent projection changed strided prefix or history sentinel")
    encoded = encode_format(trial)
    for a, b in zip(trial.materialize(attention), encoded.materialize(attention), strict=True):
        if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
            raise RuntimeError("coherent lossless header changed KV")
    return dict(
        actual_CUDA=True,
        zero_exception_input_bitwise=True,
        corrected_input_CPU_CUDA_bitwise=True,
        native_projection_frozen_tolerance=checks["frozen_construction_tolerance_pass"],
        lossless_minimum_KV_bitwise=True,
        construction=proof,
    )
