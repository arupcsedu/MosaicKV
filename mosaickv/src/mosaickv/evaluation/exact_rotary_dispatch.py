"""Default-off BF16 rotary launch coalescing, with explicit intermediate rounding.

Reference: Transformers 4.55.2 Llama apply_rotary_pos_emb/rotate_half
(Apache-2.0). Multiplications round to BF16 BEFORE the sum, exactly as in
the eager reference; FMA contraction is disabled. All methods get the same
request-private implementation. Full GPU word controls precede any use.
"""

from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

import torch
import triton
import triton.language as tl

_scope = ContextVar("evicastkv_exact_rotary_dispatch", default=None)


@triton.jit
def _pair(Q, K, COS, SIN, OQ, OK, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dim = index % 128
    rotated = index + tl.where(dim < 64, 64, -64)
    cosine, sine = tl.load(COS + dim).to(tl.float32), tl.load(SIN + dim).to(tl.float32)
    q, k = tl.load(Q + index), tl.load(K + index)
    # Exact bitwise negate before the multiply, including signed zero.
    sign = tl.where(dim < 64, 0x8000, 0).to(tl.uint16)
    rq = (tl.load(Q + rotated).to(tl.uint16, bitcast=True) ^ sign).to(tl.bfloat16, bitcast=True)
    rk = (tl.load(K + rotated).to(tl.uint16, bitcast=True) ^ sign).to(tl.bfloat16, bitcast=True)
    q_left = (q.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    q_right = (rq.to(tl.float32) * sine).to(tl.bfloat16).to(tl.float32)
    k_left = (k.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    k_right = (rk.to(tl.float32) * sine).to(tl.bfloat16).to(tl.float32)
    tl.store(OQ + index, (q_left + q_right).to(tl.bfloat16))
    tl.store(OK + index, (k_left + k_right).to(tl.bfloat16))


def compatible(q, k, cos, sin, *, position_ids=None, unsqueeze_dim=1):
    if (
        torch.is_grad_enabled()
        or q.shape != (1, 32, 1, 128)
        or k.shape != q.shape
        or cos.shape != (1, 1, 128)
        or sin.shape != cos.shape
        or position_ids is not None
        or unsqueeze_dim != 1
        or any(
            t.dtype != torch.bfloat16 or t.device != q.device or not t.is_contiguous()
            for t in (q, k, cos, sin)
        )
        or q.device.type != "cuda"
    ):
        raise NotImplementedError("private original one-token BF16 Llama rotary geometry required")


def apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    compatible(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    kernel = _pair[(32,)](q, k, cos, sin, oq, ok, BLOCK=128, num_warps=4, enable_fp_fusion=False)
    if kernel.n_spills or kernel.metadata.global_scratch_size:
        raise RuntimeError("rotary dispatch introduced hidden spills or scratch")
    state = _scope.get()
    if state is not None:
        state["calls"] += 1
    return oq, ok


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        enabled = kwargs.get("exact_rotary_dispatch", False)
        if type(enabled) is not bool or _scope.get() is not None:
            raise ValueError("explicit nonnested rotary execution boolean required")
        state = dict(enabled=enabled, calls=0)
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            from mosaickv.evaluation.bounded_probe_horizon import expected_updates

            if enabled and state["calls"] != 32 * expected_updates():
                raise RuntimeError("complete registered rotary engagement required")
            details["exact_rotary_dispatch"] = dict(
                state,
                BF16_intermediate_rounding_preserved=True,
                FMA_disabled=True,
                common_to_every_method=True,
                extra_persistent_GPU_bytes=0,
            )
            return values, packets, details
        finally:
            _scope.reset(token)

    return call


def transport():
    state = _scope.get()
    if state is None or not state["enabled"]:
        return nullcontext()
    from transformers.models.llama import modeling_llama as llama

    return patch.object(llama, "apply_rotary_pos_emb", apply)


@torch.inference_mode()
def gpu_controls():
    from transformers.models.llama import modeling_llama as llama

    torch.manual_seed(20261105)
    records = []
    for scale in (0.0, 2.0**-100, 0.01, 1.0, 64.0):
        q = torch.randn((1, 32, 1, 128), device="cuda", dtype=torch.bfloat16) * scale
        k = torch.randn_like(q) * scale
        phase = torch.randn((1, 1, 128), device="cuda", dtype=torch.bfloat16)
        cos, sin = phase.cos(), phase.sin()
        q.view(torch.int16).flatten()[0] = -32768  # Negative zero is not interchangeable.
        wanted, actual = llama.apply_rotary_pos_emb(q, k, cos, sin), apply(q, k, cos, sin)
        differences = [
            int((a.view(torch.int16) != b.view(torch.int16)).sum())
            for a, b in zip(actual, wanted, strict=True)
        ]
        records.append(dict(scale=scale, differing_words=differences))
        if any(differences):
            raise RuntimeError("coalesced rotary changed a reference BF16 word")
    return dict(
        complete=True,
        records=records,
        bitwise_reference=True,
        no_checkpoint_or_benchmark_inputs=True,
        no_persistent_GPU_workspace=True,
    )
