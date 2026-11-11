"""Default-off native eager arithmetic with a bounded, head-local KV lease.

The complete token axis is retained in both native BMMs. Only independent
heads are partitioned; no online-softmax or split-token reduction replaces the
reference arithmetic. Head grouping can still affect native kernel dispatch,
so CUDA operator and pretrained-model parity remain mandatory empirical gates.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch


def plan_groups(heads: int, length: int, slots: int = 2048) -> tuple[tuple[int, int], ...]:
    """Partition heads into feasible BMM batches, explicitly excluding B=1."""
    if any(type(value) is not int or value <= 0 for value in (heads, length, slots)):
        raise ValueError("positive integer head count, full length and workspace slots required")
    capacity = min(heads, slots // length)
    if heads < 2 or capacity < 2 or (capacity == 2 and heads % 2):
        raise NotImplementedError("full token axis cannot fit non-singleton head batches")
    result, start = [], 0
    while start < heads:
        count = min(capacity, heads - start)
        if heads - start - count == 1:
            count -= 1
        if count < 2:
            raise NotImplementedError("a singleton native GEMM is not an audited BMM batch")
        result.append((start, start + count))
        start += count
    return tuple(result)


class NativeHeadWorkspace:
    """Two fixed BF16 allocations shared by all head groups and decoder layers."""

    def __init__(self, *, device: Any, head_dim: int = 128, slots: int = 2048) -> None:
        if type(head_dim) is not int or head_dim != 128 or type(slots) is not int or slots < 2:
            raise ValueError("audited head_dim=128 and at least two fixed workspace slots required")
        self.head_dim, self.slots = head_dim, slots
        self.key = torch.empty((slots, head_dim), dtype=torch.bfloat16, device=device)
        self.value = torch.empty_like(self.key)

    def owned_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key, self.value

    @property
    def backing_bytes(self) -> int:
        return sum(t.untyped_storage().nbytes() for t in self.owned_tensors())

    def group_views(self, heads: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            type(heads) is not int
            or type(length) is not int
            or heads < 2
            or length < 1
            or heads * length > self.slots
        ):
            raise ValueError("non-singleton complete-token group must fit the fixed workspace")
        shape = (1, heads, length, self.head_dim)
        return tuple(t[: heads * length].view(shape) for t in self.owned_tensors())


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
    """Fill exact KV groups and return HF-layout attention output and accounting.

    ``fill_group(first, stop, K, V)`` receives contiguous ``[1,G,N,128]``
    workspace views. It must fill every prompt/history position, preserving
    original ordering and values. It may use V as scratch before writing V.
    Attention probabilities/scores are *not* retained unless explicitly asked
    for by the operator diagnostic. No weights or cache owners are modified.
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
        or type(capture_probabilities) is not bool
        or type(capture_scores) is not bool
    ):
        raise ValueError(
            "contiguous nongrad batch-one BF16 Hx1x128 query and finite scaling required"
        )
    heads = query.shape[1]
    if num_key_value_heads is not None and num_key_value_heads != heads:
        raise NotImplementedError("bounded native reader currently supports full MHA, not GQA")
    groups = plan_groups(heads, length, workspace.slots)
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
                "only same-dtype broadcastable all-valid eager masks supported"
            )
        valid = (attention_mask[..., :length] == 0).all()
        if query.is_cuda:
            torch._assert_async(valid, "bounded native attention requires an all-valid mask")
        elif not bool(valid):
            raise NotImplementedError("masked/padded token-axis attention is not supported")
    output = torch.empty_like(query)
    captured_p = (
        torch.empty((1, heads, 1, length), dtype=query.dtype, device=query.device)
        if capture_probabilities
        else None
    )
    captured_s = (
        torch.empty_like(captured_p)
        if capture_scores and captured_p is not None
        else (
            torch.empty((1, heads, 1, length), dtype=query.dtype, device=query.device)
            if capture_scores
            else None
        )
    )
    retained = tuple(t for t in (output, captured_p, captured_s) if t is not None)

    def observation(*active: torch.Tensor) -> None:
        if observe is not None:
            observe((*workspace.owned_tensors(), *retained, *active))

    observation()
    for first, stop in groups:
        key, value = workspace.group_views(stop - first, length)
        fill_group(first, stop, key, value)
        count = stop - first
        q = query[:, first:stop].view(count, 1, 128)
        k = key.view(count, length, 128)
        v = value.view(count, length, 128)
        raw_scores = torch.bmm(q, k.transpose(1, 2))
        scores = raw_scores * scaling
        observation(raw_scores, scores)
        del raw_scores
        if attention_mask is not None:
            mask = (
                attention_mask[:, :1]
                if attention_mask.shape[1] == 1
                else attention_mask[:, first:stop]
            )
            masked = scores + mask[..., :length].reshape(-1, 1, length)
            observation(scores, masked)
            scores = masked
            del masked
        if captured_s is not None:
            captured_s[:, first:stop].copy_(scores.view(1, count, 1, length))
        fp32_probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        probabilities = fp32_probabilities.to(query.dtype)
        observation(scores, fp32_probabilities, probabilities)
        del scores, fp32_probabilities
        group_output = torch.bmm(probabilities, v)
        observation(probabilities, group_output)
        output[:, first:stop].copy_(group_output.view(1, count, 1, 128))
        if captured_p is not None:
            captured_p[:, first:stop].copy_(probabilities.view(1, count, 1, length))
        del probabilities, group_output
    observation()
    report: dict[str, Any] = dict(
        policy="native_full_token_axis_head_bounded_v1",
        groups=[list(group) for group in groups],
        full_token_axis=length,
        token_reduction_is_unsplit=True,
        singleton_batches=False,
        workspace_slots_per_component=workspace.slots,
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
    slots: int = 2048,
) -> dict[str, Any]:
    """Collect all native/reference stage differences; never hide failed lengths.

    Real CUDA execution belongs inside Slurm. This callable loads no weights
    and changes no determinism, TF32 or reduced-precision reduction settings.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA operator selftest requires an allocated GPU")
    generator = torch.Generator(device=device).manual_seed(seed)
    workspace = NativeHeadWorkspace(device=device, slots=slots)
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

            def fill(
                first: int,
                stop: int,
                k: torch.Tensor,
                v: torch.Tensor,
                key: torch.Tensor = key,
                value: torch.Tensor = value,
            ) -> None:
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
