"""Default-off common eager attention: original GEMMs, bounded FP32 softmax.

No SDPA substitution, reduced-precision softmax, or shortened attention. Every
row normalizes over the complete original key sequence. Apply identically to
all arms and require operator/model parity before measuring memory effects.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F


def probabilities(scores, *, rows=2048, dtype=None, out=None):
    if type(rows) is not int or rows < 1 or scores.ndim != 4 or not scores.is_contiguous():
        raise ValueError(
            "positive row tile and contiguous four-dimensional attention scores required"
        )
    if out is not None and (
        out is not scores or (dtype or scores.dtype) != scores.dtype or scores.requires_grad
    ):
        raise ValueError("only explicit same-dtype nongrad private-score overwrite is supported")
    result = torch.empty_like(scores, dtype=dtype or scores.dtype) if out is None else out
    source, target = scores.flatten(0, 2), result.flatten(0, 2)
    for begin in range(0, source.shape[0], rows):
        target[begin : begin + rows].copy_(
            F.softmax(source[begin : begin + rows], dim=-1, dtype=torch.float32).to(result.dtype)
        )
    return result


@contextmanager
def use(*, rows=2048, inplace=False):
    from transformers.models.llama import modeling_llama as llama

    if type(rows) is not int or rows < 1 or type(inplace) is not bool:
        raise ValueError("positive registered softmax tile required")
    report = dict(
        rows=rows,
        calls=0,
        maximum_rows=0,
        exact_operator_required=True,
        common_to_all_arms=True,
        original_attention_GEMMs=True,
        private_score_buffer_reused=inplace,
    )

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.training or dropout != 0:
            raise NotImplementedError("deterministic evaluation-only tiled eager attention")
        key = llama.repeat_kv(key, module.num_key_value_groups)
        value = llama.repeat_kv(value, module.num_key_value_groups)
        if inplace:
            if torch.is_grad_enabled() or query.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("private-score reuse requires nongrad BF16/FP16 inference")
            if attention_mask is not None and attention_mask.dtype != query.dtype:
                raise ValueError("private-score reuse cannot change mask promotion arithmetic")
            # The matmul's new output is private, never a model/caller tensor.
            # Retain the exact separate native scale/add operations/rounding.
            scores = torch.matmul(query, key.transpose(2, 3))
            scores.mul_(scaling)
            if attention_mask is not None:
                scores.add_(attention_mask[:, :, :, : key.shape[-2]])
            weights = probabilities(scores, rows=rows, dtype=query.dtype, out=scores)
        else:
            scores = torch.matmul(query, key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, : key.shape[-2]]
            weights = probabilities(scores, rows=rows, dtype=query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        report["calls"] += 1
        report["maximum_rows"] = max(report["maximum_rows"], scores.numel() // scores.shape[-1])
        return output, weights

    with patch.object(llama, "eager_attention_forward", attention):
        yield report
