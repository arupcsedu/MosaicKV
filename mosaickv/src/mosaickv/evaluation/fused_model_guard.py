"""Actual-model numerical substitution diagnostic, NOT a serving backend.

The parent eager operation runs as an explicit oracle. It materializes K/V;
those full clones and lossless encodings are counted diagnostic workspaces.
No latency or memory-efficiency claim follows from this guard. No labels enter.
"""

from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch

import torch

from mosaickv.evaluation.fused_lossless_attention import bits_equal, reference, run
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class GuardFailure(RuntimeError):
    pass


class Guard:
    def __init__(self, adapter, *, mode, layers=(0, 16, 31), horizon=32):
        if mode not in ("clone_repack", "fused") or horizon != 32 or layers != (0, 16, 31):
            raise ValueError("registered clone/fused, layers 0/16/31 and 32 prefixes required")
        modules = adapter._language_layers()
        if len(modules) != 32:
            raise ValueError("pinned 32-layer LLaVA Llama required")
        self.indices = {id(block.self_attn): i for i, block in enumerate(modules)}
        self.mode, self.layers, self.horizon = mode, layers, horizon
        self.steps = {i: 0 for i in layers}
        self.records = []
        self.failed_tensors = None

    def wrap(self, parent):
        @wraps(parent)
        def observed(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            index = self.indices.get(id(module))
            if index not in self.steps:
                return parent(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            step = self.steps[index]
            self.steps[index] += 1
            if step >= self.horizon:
                return parent(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if (
                dropout != 0
                or query.dtype != torch.bfloat16
                or key.data_ptr() != value.data_ptr()
                or query.shape != (1, 32, 1, 128)
            ):
                raise ValueError("private BF16 MHA single-lease eager transaction required")
            if attention_mask is not None and bool(
                (attention_mask[..., : key.shape[-2]] != 0).any()
            ):
                raise ValueError("nonzero/padded attention masks unsupported by fused diagnostic")
            # The parent overwrites K with V in the same lease. Capture K first.
            original_key = key.detach().clone()
            original_output, original_weights = parent(
                module, query, key, value, attention_mask, scaling, dropout, **kwargs
            )
            original_value = value.detach().clone()
            from mosaickv.evaluation.exact_storage import decoded, encode

            pair = tuple(encode(t) for t in (original_key, original_value))
            kv_exact = all(
                bits_equal(decoded(b), t)
                for b, t in zip(pair, (original_key, original_value), strict=True)
            )
            candidate, weights = (reference if self.mode == "clone_repack" else run)(
                query.contiguous(), *pair, scale=float(scaling)
            )
            output = candidate.transpose(1, 2).contiguous()
            exact_output, exact_weights = (
                bits_equal(output, original_output),
                bits_equal(weights, original_weights),
            )
            record = dict(
                layer=index,
                prefix=step,
                mode=self.mode,
                K_V_bits_exact=kv_exact,
                attention_output_bits_exact=exact_output,
                weight_bits_exact=exact_weights,
                maximum_attention_error=float(
                    (output.float() - original_output.float()).abs().max()
                ),
                maximum_weight_error=float(
                    (weights.float() - original_weights.float()).abs().max()
                ),
                oracle_full_KV_clone_bytes=backing_bytes((original_key, original_value)),
                packed_KV_bytes=backing_bytes(tuple(t for b in pair for t in b.buffers)),
                query_source="actual_parent_model_post_RoPE_query",
                generated_history_included=True,
                attention_length=key.shape[-2],
            )
            self.records.append(record)
            if not (kv_exact and exact_output and exact_weights):
                self.failed_tensors = dict(
                    query=query.detach().cpu(),
                    key=original_key.cpu(),
                    value=original_value.cpu(),
                    parent_output=original_output.cpu(),
                    candidate_output=output.cpu(),
                    parent_weights=original_weights.cpu(),
                    candidate_weights=weights.cpu(),
                )
                raise GuardFailure(
                    f"unchanged bitwise attention guard failed at layer={index}, prefix={step}"
                )
            return output, weights

        return observed

    @contextmanager
    def use(self):
        from transformers.models.llama import modeling_llama as llama

        from mosaickv.evaluation.sequential_kv_lease import SequentialAttentionTransport

        original_enter = SequentialAttentionTransport.__enter__
        if getattr(original_enter, "_evicastkv_fused_guard", False):
            raise ValueError("nested fused guards are unsupported")

        def enter(transport):
            try:
                result = original_enter(transport)
                transport.stack.enter_context(
                    patch.object(
                        llama, "eager_attention_forward", self.wrap(llama.eager_attention_forward)
                    )
                )
                return result
            except BaseException:
                transport.stack.close()
                raise

        enter._evicastkv_fused_guard = True
        with patch.object(SequentialAttentionTransport, "__enter__", enter):
            yield self

    def report(self):
        complete = (
            len(self.records) == len(self.layers) * self.horizon
            and all(self.steps[i] >= self.horizon for i in self.layers)
            and all(
                r["K_V_bits_exact"] and r["attention_output_bits_exact"] and r["weight_bits_exact"]
                for r in self.records
            )
        )
        return dict(
            mode=self.mode,
            complete_32_prefix_guard=complete,
            records=self.records,
            independent_quality_N=0,
            no_latency_or_memory_efficiency_claim=True,
            parent_eager_oracle_and_full_KV_clones_are_explicit=True,
            functional_substitution_not_native_or_production_backend=True,
            model_weights_modified=False,
            benchmark_labels_accessed=False,
        )
