"""Conformance screen before considering a fixed-capacity parent-attention graph."""

import torch
import torch.nn.functional as functional

from mosaickv.evaluation.eager_core_operator import reference_scores


def padded(query, key, value, mask, capacity, scaling):
    n = key.shape[-2]
    if (
        type(capacity) is not int
        or n < 1
        or capacity < n
        or key.shape != value.shape
        or query.shape != (*key.shape[:-2], 1, key.shape[-1])
        or (mask is not None and mask.shape != (*query.shape[:-1], n))
    ):
        raise ValueError("aligned original query/K/V/mask and feasible fixed capacity required")
    k, v = (functional.pad(t, (0, 0, 0, capacity - n)) for t in (key, value))
    m = torch.zeros(
        (*query.shape[:-1], capacity),
        device=query.device,
        dtype=(torch.float32 if mask is None else mask.dtype),
    )
    if mask is not None:
        m[..., :n].copy_(mask)
    m[..., n:] = float("-inf")
    p = reference_scores(query, k, m, scaling)
    return p[..., :n], torch.matmul(p, v)


@torch.inference_mode()
def run(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cases = []
    for n in range(603, 732):
        query = torch.randn(1, 32, 1, 128, generator=generator, device="cuda").bfloat16()
        key, value = (
            torch.randn(1, 32, n, 128, generator=generator, device="cuda").bfloat16()
            for _ in range(2)
        )
        q, k, v = query[:, 4:5], key[:, 4:5].clone(), value[:, 4:5].clone()
        for dtype in (None, torch.bfloat16, torch.float32):
            mask = None if dtype is None else torch.zeros(1, 1, 1, n, device="cuda", dtype=dtype)
            if mask is not None:
                mask[..., -3:] = float("-inf")
            reference = reference_scores(q, k, mask, 128**-0.5)
            output = torch.matmul(reference, v)
            probabilities, result = padded(q, k, v, mask, 731, 128**-0.5)
            cases.append(
                dict(
                    length=n,
                    mask_dtype=str(dtype),
                    probabilities_bitwise=bool(torch.equal(probabilities, reference)),
                    output_bitwise=bool(torch.equal(result, output)),
                    max_probability_difference=float(
                        (probabilities.float() - reference.float()).abs().max()
                    ),
                    max_output_difference=float((result.float() - output.float()).abs().max()),
                )
            )
    return dict(
        measurement_type="synthetic_operator_conformance_not_task_quality",
        cases=cases,
        case_count=len(cases),
        strict_bitwise_gate=all(c["probabilities_bitwise"] and c["output_bitwise"] for c in cases),
        persistent_KV_buffer_bytes_if_implemented=2 * 731 * 128 * 2,
        original_thresholds_changed=False,
        pretrained_weights_loaded=False,
        new_model_path_implemented=False,
        superiority_established=False,
    )
