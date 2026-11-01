"""Four-token online forecast on a private cache borrowing read-only FullKV.

No second prefill, cloned FullKV payload, benchmark target or final-run draft
emission. Small generated history and layer leases are private and counted.
"""

import gc
import hashlib
import time
from contextlib import ExitStack

import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class Capture:
    def __init__(self, adapter, *, length, steps=4):
        self.adapter, self.length, self.steps = adapter, length, steps
        self.stack = ExitStack()
        self.layers = [
            dict(query=[], probability=[], output=[]) for _ in adapter._language_layers()
        ]
        self.cache = None

    def __enter__(self):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        for i, block in enumerate(self.adapter._language_layers()):

            def before(_module, args, kwargs, *, index=i):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None or hidden.shape[1] != 1:
                    raise ValueError("forecast must reuse prefill and decode one token")
                self.layers[index]["phase"] = kwargs["position_embeddings"]
                cache = kwargs["past_key_value"]
                if self.cache is not None and cache is not self.cache:
                    raise RuntimeError("forecast cache changed")
                self.cache = cache

            def query_hook(_module, _args, output, *, index=i):
                self.layers[index]["q_pre"] = (
                    output.detach().reshape(1, 1, -1, 128).transpose(1, 2).clone()
                )

            def out_before(_module, args, *, index=i):
                self.layers[index]["o"] = (
                    args[0].detach().reshape(1, 1, -1, 128).transpose(1, 2).clone()
                )

            def after(_module, _args, _kwargs, output, *, index=i):
                record = self.layers[index]
                step = len(record["query"])
                p = output[1]
                if p is None or p.shape[-1] != self.length + step + 1 or step >= self.steps:
                    raise RuntimeError(
                        "actual full-cache forecast attention missing or wrong length"
                    )
                q, _ = apply_rotary_pos_emb(
                    record.pop("q_pre"), torch.zeros_like(record["o"]), *record.pop("phase")
                )
                record["query"].append(q)
                record["probability"].append(p[..., : self.length].detach().clone())
                record["output"].append(record.pop("o"))

            for module, kind, hook, kwargs in (
                (block.self_attn, "pre", before, dict(with_kwargs=True)),
                (block.self_attn.q_proj, "post", query_hook, {}),
                (block.self_attn.o_proj, "pre", out_before, {}),
                (block.self_attn, "post", after, dict(with_kwargs=True)),
            ):
                handle = (
                    module.register_forward_pre_hook(hook, **kwargs)
                    if kind == "pre"
                    else module.register_forward_hook(hook, **kwargs)
                )
                self.stack.callback(handle.remove)
        return self

    def finish(self):
        if self.cache is None or any(len(r["query"]) != self.steps for r in self.layers):
            raise RuntimeError("incomplete online forecast")
        result = []
        for record, layer in zip(self.layers, self.cache.layers, strict=True):
            if layer.history_length != self.steps:
                raise RuntimeError("forecast private history incomplete")
            result.append(
                dict(
                    query=torch.cat(record["query"], dim=2),
                    probability=torch.cat(record["probability"], dim=2),
                    output=torch.cat(record["output"], dim=2),
                    history=tuple(
                        t[..., : self.steps, :].detach().contiguous().clone() for t in layer.history
                    ),
                )
            )
        self.cache = None
        self.layers.clear()
        return tuple(result)

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@torch.inference_mode()
def collect(adapter, prefill):
    from mosaickv.evaluation.storage_runs import run

    if adapter.device.type != "cuda":
        raise NotImplementedError("actual CUDA forecast only")
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    n = pairs[0][0].shape[-2]
    if any(k.shape != v.shape or k.shape[-2] != n or k.shape[-1] != 128 for k, v in pairs):
        raise ValueError("uniform full-MHA source required for read-only forecast")
    metadata = (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    before = tensor_hash(tuple(t for pair in pairs for t in pair))
    formats = tuple(
        ExactFormat(
            "kv",
            tuple(k.shape),
            pair=(PackedBF16(tuple(k.shape), True, (k,)), PackedBF16(tuple(v.shape), True, (v,))),
        )
        for k, v in pairs
    )
    torch.cuda.synchronize()
    begun = time.monotonic()
    with Capture(adapter, length=n, steps=4) as captured:
        logits, details = run(
            adapter, prefill, formats, tokens=5, sealed=False, audit=False, sequential_lease=True
        )
        evidence = captured.finish()
    gc.collect()
    torch.cuda.synchronize()
    wall = time.monotonic() - begun
    if metadata != (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    ) or before != tensor_hash(tuple(t for pair in pairs for t in pair)):
        raise RuntimeError("online forecast mutated original full-cache state")
    auxiliaries = tuple(
        t for r in evidence for t in (r["query"], r["probability"], r["output"], *r["history"])
    )
    return (
        evidence,
        dict(
            steps=4,
            second_prefill=False,
            original_cache_unchanged=True,
            private_append_only_history=True,
            read_only_prefix_borrowed_not_cloned=True,
            original_logical_positions_unchanged=True,
            wall_seconds_excludes_integrity_hashes=wall,
            raw_cache_storage=details["storage"],
            borrowed_prefix_bytes=backing_bytes(tuple(t for p in pairs for t in p)),
            auxiliary_forecast_bytes=backing_bytes(auxiliaries),
            no_benchmark_labels_or_reference_artifact=True,
            no_draft_tokens_emitted_in_final_run=True,
            prompt_probabilities_not_renormalized=True,
        ),
        logits,
    )
