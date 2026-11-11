"""Native B=32 reader with a single reusable full-component projection pool.

K is constructed once, consumed completely, then overwritten by V. No full K
and full V lease coexist. A contiguous one-head scratch preserves the verified
zero-stride native BMM dispatch. Source and raw arms must share this policy and
charge both fixed allocations, including unused capacity.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

NATIVE_BATCH = 32
SLOTS = 1024


class ComponentWorkspace:
    """One N-major BF16 projection component and one contiguous head scratch."""

    def __init__(self, *, device: Any, heads: int = 32, head_dim: int = 128, slots: int = 1024):
        device = torch.device(device)
        if (
            type(heads) is not int
            or heads < 1
            or head_dim != 128
            or slots != SLOTS
            or (device.type == "cuda" and heads != 32)
        ):
            raise ValueError(
                "registered 1024-slot BF16 MHA128 workspace required; CUDA uses 32 heads"
            )
        self.heads, self.head_dim, self.slots = heads, head_dim, slots
        self.component = torch.empty((slots, heads * head_dim), dtype=torch.bfloat16, device=device)
        self.head = torch.empty((slots, head_dim), dtype=torch.bfloat16, device=device)

    def owned_tensors(self):
        return self.component, self.head

    @property
    def backing_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.owned_tensors())

    def component_view(self, length: int):
        if type(length) is not int or not 1 <= length <= self.slots:
            raise ValueError("complete physical sequence must fit the registered 1024 slots")
        return self.component[:length].view(1, length, self.heads, self.head_dim).transpose(1, 2)

    def head_view(self, length: int):
        if type(length) is not int or not 1 <= length <= self.slots:
            raise ValueError("complete physical sequence must fit the registered 1024 slots")
        return self.head[:length].view(1, 1, length, self.head_dim)


@torch.no_grad()
def attention(
    query: torch.Tensor,
    fill_component: Callable[[int, torch.Tensor, torch.Tensor], None],
    workspace: ComponentWorkspace,
    *,
    length: int,
    scaling: float,
    prompt_lengths: tuple[int, ...] | None = None,
    prompt_max_length: int | None = None,
    attention_mask: torch.Tensor | None = None,
    num_key_value_heads: int | None = None,
    capture_probabilities: bool = False,
    capture_scores: bool = False,
    observe: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Read K once, native-softmax all heads, then overwrite the pool with V.

    ``fill_component(component, target, head_scratch)`` receives component 0/1,
    ``target[1,H,N,128]`` with contiguous N-major projection backing, and
    ``head_scratch[1,1,N,128]``. The callback may use the scratch while filling
    K/RoPE. It must fill every position, including deterministic padding.
    """
    if (
        query.shape != (1, workspace.heads, 1, 128)
        or query.dtype != torch.bfloat16
        or query.device != workspace.component.device
        or not query.is_contiguous()
        or query.requires_grad
        or not math.isfinite(scaling)
        or scaling <= 0
        or type(capture_probabilities) is not bool
        or type(capture_scores) is not bool
    ):
        raise ValueError("contiguous nongrad BF16 MHA128 query matching the workspace required")
    if num_key_value_heads is not None and num_key_value_heads != workspace.heads:
        raise NotImplementedError("component reader does not support GQA")
    pool, scratch = workspace.component_view(length), workspace.head_view(length)
    heads = workspace.heads
    if (prompt_lengths is None) != (prompt_max_length is None):
        raise ValueError(
            "per-head prompt lengths and the physical prompt maximum must be supplied together"
        )
    if prompt_lengths is not None and (
        type(prompt_lengths) is not tuple
        or len(prompt_lengths) != heads
        or type(prompt_max_length) is not int
        or not 1 <= prompt_max_length <= length
        or any(type(n) is not int or not 1 <= n <= prompt_max_length for n in prompt_lengths)
    ):
        raise ValueError(
            "complete positive per-head prompt lengths within the physical maximum required"
        )
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
            raise NotImplementedError("only compatible all-zero caller masks supported")
        valid = (attention_mask[..., :length] == 0).all()
        if query.is_cuda:
            torch._assert_async(valid, "component attention caller mask must be all zero")
        elif not bool(valid):
            raise NotImplementedError(
                "nonzero caller masks unsupported; use explicit prompt lengths"
            )

    output = torch.empty_like(query)
    scores = torch.empty((1, heads, 1, length), dtype=query.dtype, device=query.device)

    def observation(*active):
        if observe is not None:
            observe((*workspace.owned_tensors(), output, *active))

    observation(scores)
    fill_component(0, pool, scratch)
    for head in range(heads):
        scratch.copy_(pool[:, head : head + 1])
        q = query[:, head].view(1, 1, 128).expand(NATIVE_BATCH, -1, -1)
        k = scratch.view(1, length, 128).expand(NATIVE_BATCH, -1, -1)
        raw = torch.bmm(q, k.transpose(1, 2))
        scaled = raw * scaling
        observation(scores, raw, scaled)
        scores[:, head].copy_(scaled[:1])
        del raw, scaled
        if attention_mask is not None:
            index = 0 if attention_mask.shape[1] == 1 else head
            scores[:, head].add_(attention_mask[:, index, :, :length])
        if prompt_lengths is not None:
            scores[:, head, :, prompt_lengths[head] : prompt_max_length] = torch.finfo(
                query.dtype
            ).min
    captured_s = scores if capture_scores else None
    fp32_probabilities = torch.softmax(scores, -1, dtype=torch.float32)
    probabilities = fp32_probabilities.to(query.dtype)
    observation(scores, fp32_probabilities, probabilities)
    del scores, fp32_probabilities
    retained = (captured_s,) if captured_s is not None else ()
    observation(*retained, probabilities)
    # This overwrites the only full component; no reference to full K survives.
    fill_component(1, pool, scratch)
    for head in range(heads):
        scratch.copy_(pool[:, head : head + 1])
        p = probabilities[:, head].view(1, 1, length).expand(NATIVE_BATCH, -1, -1)
        v = scratch.view(1, length, 128).expand(NATIVE_BATCH, -1, -1)
        group_output = torch.bmm(p, v)
        observation(*retained, probabilities, group_output)
        output[:, head].copy_(group_output[:1])
        del group_output
    observation(*retained, probabilities)
    report: dict[str, Any] = dict(
        policy="native_B32_one_full_component_v1",
        full_token_axis=length,
        token_reduction_is_unsplit=True,
        component_fills=2,
        simultaneous_full_components=1,
        physical_component_layout="N_major_H_D_transposed_BHND_view",
        native_bmm_batch=NATIVE_BATCH,
        expanded_batch_stride=0,
        KV_repeat_or_clone=False,
        workspace_slots_per_component=SLOTS,
        workspace_backing_bytes=workspace.backing_bytes,
        redundant_batch_computation_included=True,
        prompt_padding_masked_positions=(
            sum(prompt_max_length - n for n in prompt_lengths) if prompt_lengths is not None else 0
        ),
        probabilities_retained=capture_probabilities,
        scores_retained=capture_scores,
        native_operator_parity_required=True,
    )
    if captured_s is not None:
        report["scores"] = captured_s
    return (
        output.transpose(1, 2).contiguous(),
        probabilities if capture_probabilities else None,
        report,
    )


def selftest(
    *,
    device: Any = "cuda",
    lengths: Sequence[int] = (98, 375, 600, 604, 727, 731),
    seed: int = 20261111,
) -> dict[str, Any]:
    """Record every stage versus unchanged native BF16 B=32 reference arithmetic."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA operator selftest requires an allocated GPU")
    generator = torch.Generator(device=device).manual_seed(seed)
    workspace = ComponentWorkspace(device=device)
    rows = []
    with torch.no_grad():
        for length in lengths:
            query = torch.randn((1, 32, 1, 128), generator=generator, device=device).bfloat16()
            key, value = tuple(
                torch.randn((1, 32, length, 128), generator=generator, device=device).bfloat16()
                for _ in range(2)
            )
            ref_s = (
                torch.bmm(query.view(32, 1, 128), key.view(32, length, 128).transpose(1, 2))
                * 128**-0.5
            )
            ref_p = torch.softmax(ref_s, -1, dtype=torch.float32).bfloat16()
            ref_y = (
                torch.bmm(ref_p, value.view(32, length, 128))
                .view(1, 32, 1, 128)
                .transpose(1, 2)
                .contiguous()
            )

            def fill(component, target, _scratch, key=key, value=value):
                target.copy_((key, value)[component])

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
                ("scores", ref_s.view(1, 32, 1, length), actual_s),
                ("probabilities", ref_p.view(1, 32, 1, length), actual_p),
                ("output", ref_y, actual_y),
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
        torch_version=torch.__version__,
        seed=seed,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        workspace_backing_bytes=workspace.backing_bytes,
        rows=rows,
        all_bitwise_passed=all(row["bitwise_passed"] for row in rows),
    )
