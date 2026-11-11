"""Default-off native B=32 dispatch with one reconstructed head resident.

Both GEMMs retain their complete token reduction and the registered native
BF16/FP32/BF16 boundaries. Expanded batch operands have zero batch stride;
they are views, never repeated/cloned full-head KV allocations. Redundant
batch computation is intentional and must be included in measured latency.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from mosaickv.evaluation.bounded_native_attention import NativeHeadWorkspace

BROADCAST_BATCH = 32
WORKSPACE_SLOTS = 1024


@torch.no_grad()
def attention(
    query: torch.Tensor,
    fill_group: Callable[[int, int, torch.Tensor, torch.Tensor], None],
    workspace: NativeHeadWorkspace,
    *,
    length: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
    num_key_value_heads: int | None = None,
    capture_probabilities: bool = False,
    capture_scores: bool = False,
    observe: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Return HF-layout output using singleton KV owners and native B=32 BMM.

    The filler receives ``[1,1,N,128]`` views for one head and must overwrite
    every prompt/history position. V may serve as K's temporary RoPE scratch,
    provided final V is filled afterward. Scores/probabilities are retained
    only when explicitly requested for an operator diagnostic.
    """
    if (
        query.ndim != 4
        or query.shape[0] != 1
        or query.shape[2:] != (1, 128)
        or query.dtype != torch.bfloat16
        or query.device != workspace.key.device
        or not query.is_contiguous()
        or query.requires_grad
        or not math.isfinite(scaling)
        or scaling <= 0
        or type(length) is not int
        or not 1 <= length <= WORKSPACE_SLOTS
        or workspace.slots != WORKSPACE_SLOTS
        or type(capture_probabilities) is not bool
        or type(capture_scores) is not bool
    ):
        raise ValueError(
            "nongrad contiguous BF16 Hx1x128 query and complete N<=1024 workspace required"
        )
    heads = query.shape[1]
    if heads < 1 or (query.is_cuda and heads != BROADCAST_BATCH):
        raise NotImplementedError("registered CUDA transport requires exactly 32 full-MHA heads")
    if num_key_value_heads is not None and num_key_value_heads != heads:
        raise NotImplementedError("broadcast native transport does not support GQA")
    if attention_mask is not None:
        if (
            attention_mask.device != query.device
            or attention_mask.dtype != query.dtype
            or attention_mask.ndim != 4
            or attention_mask.shape[0] != 1
            or attention_mask.shape[1] not in (1, heads)
            or attention_mask.shape[2] != 1
            or attention_mask.shape[3] < length
        ):
            raise NotImplementedError(
                "only same-dtype all-valid broadcastable eager masks supported"
            )
        valid = (attention_mask[..., :length] == 0).all()
        if query.is_cuda:
            torch._assert_async(valid, "broadcast native transport requires an all-valid mask")
        elif not bool(valid):
            raise NotImplementedError("padded or nonzero attention masks are unsupported")

    output = torch.empty_like(query)
    shape = (1, heads, 1, length)
    captured_p = (
        torch.empty(shape, dtype=query.dtype, device=query.device)
        if capture_probabilities
        else None
    )
    captured_s = (
        torch.empty(shape, dtype=query.dtype, device=query.device) if capture_scores else None
    )
    retained = tuple(t for t in (output, captured_p, captured_s) if t is not None)
    key, value = tuple(t[:length].view(1, 1, length, 128) for t in workspace.owned_tensors())

    def observation(*active: torch.Tensor) -> None:
        if observe is not None:
            observe((*workspace.owned_tensors(), *retained, *active))

    observation()
    for head in range(heads):
        fill_group(head, head + 1, key, value)
        q = query[:, head].view(1, 1, 128).expand(BROADCAST_BATCH, -1, -1)
        k = key.view(1, length, 128).expand(BROADCAST_BATCH, -1, -1)
        v = value.view(1, length, 128).expand(BROADCAST_BATCH, -1, -1)
        raw_scores = torch.bmm(q, k.transpose(1, 2))
        scores = raw_scores * scaling
        observation(raw_scores, scores)
        del raw_scores
        if attention_mask is not None:
            index = 0 if attention_mask.shape[1] == 1 else head
            mask = attention_mask[:, index, :, :length]
            masked = scores + mask
            observation(scores, masked)
            scores = masked
            del masked
        if captured_s is not None:
            captured_s[:, head].copy_(scores[:1])
        fp32_probabilities = torch.softmax(scores, -1, dtype=torch.float32)
        probabilities = fp32_probabilities.to(query.dtype)
        observation(scores, fp32_probabilities, probabilities)
        del scores, fp32_probabilities
        p = probabilities[:1].expand(BROADCAST_BATCH, -1, -1)
        group_output = torch.bmm(p, v)
        observation(probabilities, group_output)
        output[:, head].copy_(group_output[:1])
        if captured_p is not None:
            captured_p[:, head].copy_(probabilities[:1])
        del probabilities, p, group_output
    observation()
    report: dict[str, Any] = dict(
        policy="native_B32_zero_stride_single_head_v1",
        full_token_axis=length,
        token_reduction_is_unsplit=True,
        physical_heads_per_fill=1,
        native_bmm_batch=BROADCAST_BATCH,
        expanded_batch_stride=0,
        KV_repeat_or_clone=False,
        redundant_batch_computation_included=True,
        workspace_slots_per_component=WORKSPACE_SLOTS,
        workspace_backing_bytes=workspace.backing_bytes,
        probabilities_retained=capture_probabilities,
        scores_retained=capture_scores,
        native_operator_parity_required=True,
    )
    if captured_s is not None:
        report["scores"] = captured_s
    return output.transpose(1, 2).contiguous(), captured_p, report


def selftest(
    *,
    device: Any = "cuda",
    lengths: Sequence[int] = (98, 375, 600, 604, 727, 731),
    seed: int = 20261111,
) -> dict[str, Any]:
    """Collect all stage differences from the unchanged native BF16 B=32 path."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA operator selftest requires an allocated GPU")
    generator = torch.Generator(device=device).manual_seed(seed)
    workspace = NativeHeadWorkspace(device=device, slots=WORKSPACE_SLOTS)
    rows = []
    with torch.no_grad():
        for length in lengths:
            query = torch.randn((1, 32, 1, 128), generator=generator, device=device).bfloat16()
            key, value = tuple(
                torch.randn((1, 32, length, 128), generator=generator, device=device).bfloat16()
                for _ in range(2)
            )
            reference_s = (
                torch.bmm(query.view(32, 1, 128), key.view(32, length, 128).transpose(1, 2))
                * 128**-0.5
            )
            reference_p = torch.softmax(reference_s, -1, dtype=torch.float32).bfloat16()
            reference_y = (
                torch.bmm(reference_p, value.view(32, length, 128))
                .view(1, 32, 1, 128)
                .transpose(1, 2)
                .contiguous()
            )

            def fill(first, stop, k, v, key=key, value=value):
                k.copy_(key[:, first:stop])
                v.copy_(value[:, first:stop])

            actual_y, actual_p, record = attention(
                query,
                fill,
                workspace,
                length=length,
                scaling=128**-0.5,
                capture_probabilities=True,
                capture_scores=True,
            )
            actual_s = record.pop("scores")
            row: dict[str, Any] = dict(length=length, **record)
            for name, expected, actual in (
                ("scores", reference_s.view(1, 32, 1, length), actual_s),
                ("probabilities", reference_p.view(1, 32, 1, length), actual_p),
                ("output", reference_y, actual_y),
            ):
                row[name] = dict(
                    bitwise_equal=bool(
                        torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                    ),
                    unequal_elements=int(
                        (expected.view(torch.int16) != actual.view(torch.int16)).sum()
                    ),
                    maximum_absolute_difference=float(
                        (expected.float() - actual.float()).abs().max()
                    ),
                )
            row["bitwise_passed"] = all(
                row[name]["bitwise_equal"] for name in ("scores", "probabilities", "output")
            )
            rows.append(row)
    return dict(
        measurement_type=f"synthetic_{device.type}_operator_diagnostic",
        seed=seed,
        torch_version=torch.__version__,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        workspace_backing_bytes=workspace.backing_bytes,
        rows=rows,
        all_bitwise_passed=all(row["bitwise_passed"] for row in rows),
    )
