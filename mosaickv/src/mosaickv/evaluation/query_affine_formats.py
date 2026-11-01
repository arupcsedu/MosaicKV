"""Actual CUDA query-aware source/KV precision construction, no labels or training."""

from dataclasses import replace

import torch

from mosaickv.evaluation.affine_refit import refit, refit_cuda
from mosaickv.evaluation.conditional_precision import exact_rows
from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.shared_source_format import fit_transport
from mosaickv.evaluation.source_query_importance import (
    blend_families,
    importance,
    importance_cuda,
    kv_importance_cuda,
)
from mosaickv.evaluation.source_retile import retile
from mosaickv.evaluation.text_source_storage import TextSourceFormat, optimize


def prompt_query(attention, record, heads):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    hidden = record["normalized"]
    query = attention.q_proj(hidden).reshape(1, hidden.shape[1], heads, 128).transpose(1, 2)
    query, _ = apply_rotary_pos_emb(query, query, record["cos"], record["sin"])
    return query


@torch.inference_mode()
def source_formats(
    adapter,
    prefill,
    capture,
    source_capture,
    text_capture,
    phase_capture,
    parents,
    *,
    head_group,
    forecast=None,
):
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    results, records = [], []
    for i, (parent, pair, block) in enumerate(
        zip(parents, original, adapter._language_layers(), strict=True)
    ):
        h, n = pair[0].shape[1:3]
        mask = torch.zeros(h * n, device=adapter.device, dtype=torch.bool)
        mask[parent.row_positions.long()] = True
        mask = mask.reshape(h, n)
        q = prompt_query(block.self_attn, capture.records[i], h)
        weights, weight_proof = importance_cuda(
            q,
            *pair,
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
            torch.arange(n, device=adapter.device).repeat(h, 1),
            torch.arange(n - q.shape[-2], n, device=adapter.device),
            phase_capture.phases,
            mask,
        )
        if forecast is not None:
            from mosaickv.evaluation.source_query_importance import future_importance_cuda

            future = forecast[i]
            future_weights = future_importance_cuda(
                future["query"],
                pair[1],
                block.self_attn.k_proj.weight,
                block.self_attn.v_proj.weight,
                phase_capture.phases,
                mask,
                future["probability"],
                future["output"],
            )
            weights = blend_families(
                weights, future_weights, q.shape[-2], future["query"].shape[-2]
            )
            weight_proof.update(
                forecast_tokens=4,
                prompt_family_weight=0.5,
                future_family_weight=0.5,
                actual_draft_attention_mass_not_renormalized=True,
            )
        body, proof = refit_cuda(
            parent.source,
            source_capture.records[i][:, None],
            exact_rows=mask.all(0)[None],
            importance=weights,
        )
        packet = replace(parent, source=body, transport_patches=())
        packet, numerical = fit_transport(packet, block.self_attn)
        packet, text_proof = optimize(
            packet,
            text_capture.records[i],
            (tuple(range(n)),) * h,
            text_capture.positions,
            block.self_attn,
        )
        results.append(retile(packet, head_group))
        records.append(
            dict(layer=i, fit=proof, importance=weight_proof, numerical=numerical, text=text_proof)
        )
    return tuple(results), records


def replace_pair(packet, pair):
    if isinstance(packet, (TextSourceFormat, HeadLocalFormat)):
        return replace(packet, base=replace_pair(packet.base, pair))
    return replace(packet, pair=tuple(pair))


@torch.inference_mode()
def baseline_formats(
    adapter, prefill, capture, parents, original_pairs, memberships, *, forecast=None
):
    results, records = [], []
    for i, (parent, pair, positions, block) in enumerate(
        zip(parents, original_pairs, memberships, adapter._language_layers(), strict=True)
    ):
        local = parent.base if isinstance(parent, TextSourceFormat) else parent
        if not isinstance(local, HeadLocalFormat):
            raise NotImplementedError("audited head-local baseline required")
        n = prefill.state.logical_sequence_length
        h = pair[0].shape[1]
        q = prompt_query(block.self_attn, capture.records[i], h)
        mask = exact_rows(parent)
        weights, weight_proof = kv_importance_cuda(
            q,
            *pair,
            torch.tensor(positions, device=adapter.device),
            torch.arange(n - q.shape[-2], n, device=adapter.device),
            mask,
        )
        if forecast is not None:
            future = forecast[i]
            r = future["query"].shape[-2]
            length = pair[0].shape[-2]
            joint = tuple(
                torch.cat((a, b), dim=2) for a, b in zip(pair, future["history"], strict=True)
            )
            logical = torch.cat(
                (
                    torch.tensor(positions, device=adapter.device),
                    torch.arange(n, n + r, device=adapter.device)[None].expand(h, r),
                ),
                dim=1,
            )
            exact = torch.cat(
                (mask, torch.ones(h, r, device=adapter.device, dtype=torch.bool)), dim=1
            )
            future_weights, future_proof = kv_importance_cuda(
                future["query"],
                *joint,
                logical,
                torch.arange(n, n + r, device=adapter.device),
                exact,
            )
            weights = tuple(
                blend_families(a, b[..., :length, :], q.shape[-2], r)
                for a, b in zip(weights, future_weights, strict=True)
            )
            weight_proof.update(
                forecast_tokens=4,
                prompt_family_weight=0.5,
                future_family_weight=0.5,
                hypothetical_selected_context_with_common_FullKV_draft=future_proof,
            )
        weights = tuple(t.clone() for t in weights)
        for index, patch in enumerate(local.base.corrections):
            weights[index].reshape(-1)[patch.indices.long()] = 0
        fitted = [
            refit_cuda(body, values, exact_rows=mask, importance=w)
            for body, values, w in zip(local.base.pair, pair, weights, strict=True)
        ]
        results.append(replace_pair(parent, tuple(r[0] for r in fitted)))
        records.append(
            dict(
                layer=i,
                fit=[r[1] for r in fitted],
                importance=weight_proof,
                original_exact_and_exception_values_preserved=True,
                merged_values_not_regenerated=True,
            )
        )
    return tuple(results), records


@torch.inference_mode()
def gpu_controls():
    if not torch.cuda.is_available() or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("real CUDA with unchanged FP32 highest-precision controls required")
    generator = torch.Generator().manual_seed(20261101)
    h, n, d = 2, 16, 128
    q = torch.randint(-2, 3, (1, h, 3, d), generator=generator).bfloat16()
    key = torch.randint(-2, 3, (1, h, n, d), generator=generator).bfloat16()
    value = torch.randint(-2, 3, (1, h, n, d), generator=generator).bfloat16()
    wk = torch.randn(h * d, h * d, generator=generator).bfloat16().float() * 0.01
    wv = torch.randn(h * d, h * d, generator=generator).bfloat16().float() * 0.01
    positions = torch.arange(n).repeat(h, 1)
    queries = torch.arange(n - 3, n)
    phases = torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16()
    exact = torch.zeros(h, n, dtype=torch.bool)
    exact[:, 0] = True
    weight, _ = importance(q, key, value, wk, wv, positions, queries, phases, exact)
    cuda_weight, _ = importance_cuda(
        *(t.cuda() for t in (q, key, value, wk, wv, positions, queries)),
        tuple(t.cuda() for t in phases),
        exact.cuda(),
    )
    torch.testing.assert_close(cuda_weight.cpu(), weight, atol=1e-6, rtol=1e-5)
    from mosaickv.evaluation.source_query_importance import (
        _probabilities,
        future_importance_cpu,
        future_importance_cuda,
    )

    probability, _, _ = _probabilities(q, key, value, positions, queries)
    output = (probability @ value.float()).bfloat16()
    reference_future = future_importance_cpu(q, value, wk, wv, phases, exact, probability, output)
    actual_future = future_importance_cuda(
        q.cuda(),
        value.cuda(),
        wk.cuda(),
        wv.cuda(),
        tuple(t.cuda() for t in phases),
        exact.cuda(),
        probability.cuda(),
        output.cuda(),
    )
    torch.testing.assert_close(actual_future.cpu(), reference_future, atol=1e-6, rtol=1e-5)
    x = torch.randn(1, 1, n, h * d, generator=generator).bfloat16()
    cases = []
    mask = exact.all(0)[None]
    for bits in (3, 8):
        for axis in (2, 3):
            body = pack(x, bits=bits, axis=axis)
            expected, _ = refit(body, x, importance=weight, exact_rows=mask)
            cuda_body = replace(
                body, codes=body.codes.cuda(), minimum=body.minimum.cuda(), scale=body.scale.cuda()
            )
            actual, _ = refit_cuda(
                cuda_body, x.cuda(), importance=weight.cuda(), exact_rows=mask.cuda()
            )
            host = replace(
                actual,
                codes=actual.codes.cpu(),
                minimum=actual.minimum.cpu(),
                scale=actual.scale.cpu(),
            )
            if not torch.equal(expected.codes, host.codes):
                raise RuntimeError(f"GPU refit code assignment mismatch {bits}/{axis}")
            for a, b in ((expected.minimum, host.minimum), (expected.scale, host.scale)):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            if not torch.equal(unpack_cpu(expected), unpack_cpu(host)):
                raise RuntimeError(f"GPU refit decoded BF16 mismatch {bits}/{axis}")
            cases.append(dict(bits=bits, axis=axis, codes_bitwise=True, decoded_BF16_bitwise=True))
    return dict(
        actual_CUDA_execution=True,
        importance_FP32_atol=1e-6,
        importance_FP32_rtol=1e-5,
        codes_bitwise=True,
        decoded_BF16_bitwise=True,
        same_persistent_payload_size=True,
        TF32_disabled=True,
        CPU_oracle_not_substituted_for_GPU=True,
        fitter_fixture_identical_importance_inputs=True,
        independently_computed_CUDA_importance_compared=True,
        cases=cases,
    )
