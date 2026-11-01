"""Consumed-input diagnostic: exact nonlinear readout of last-layer head changes.

No later transformer layers are approximated. Conditional on a fixed token
prefix, a last-layer output intervention cannot change any cached K/V.
The scout observes one private compressed replay; it never changes that replay.
"""

from contextlib import ExitStack

import torch


@torch.inference_mode()
def tail_logits(block, norm, readout, residual, vector):
    """Run the actual frozen post-attention tail, preserving its BF16 order."""
    hidden = residual + block.self_attn.o_proj(vector)
    hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
    return readout(norm(hidden))[:, -1, :].float()


def statistics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("equal prefix-by-vocabulary logits required")
    a, b = reference.float(), candidate.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite tail logits")
    logp, logq = a.log_softmax(-1), b.log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)
    mismatch = int((a.argmax(-1) != b.argmax(-1)).sum())
    return dict(
        maximum_KL_FP32=float(kl.max()),
        mean_KL_FP32=float(kl.mean()),
        maximum_logit_difference=float((a - b).abs().max()),
        token_mismatches=mismatch,
        token_agreement=1 - mismatch / len(a),
        guard_pass=mismatch == 0 and float(kl.max()) <= 0.02 and float((a - b).abs().max()) <= 1.0,
    )


def order(record):
    s = record["statistics"]
    return (
        not s["guard_pass"],
        s["token_mismatches"],
        s["maximum_KL_FP32"],
        s["maximum_logit_difference"],
        s["mean_KL_FP32"],
        record["head"],
    )


class TailScout:
    """Capture actual compressed residual/query and all-head FullKV alternatives."""

    def __init__(self, adapter, source_pair):
        self.block = adapter._language_layers()[-1]
        self.source = source_pair
        self.stack = ExitStack()
        self.active = {}
        self.records = []
        self.temporary_kv_peak_bytes = 0

    def __enter__(self):
        from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

        attention = self.block.self_attn
        if (
            not isinstance(attention, LlamaAttention)
            or attention.training
            or attention.config._attn_implementation != "eager"
            or attention.num_key_value_groups != 1
            or attention.head_dim != 128
        ):
            raise NotImplementedError("eval-only eager full-MHA Llama tail required")

        def residual(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if self.active or hidden is None or hidden.shape[:2] != (1, 1):
                raise RuntimeError("nonisolated or non-single-query tail capture")
            self.active["residual"] = hidden.detach().clone()

        def phase(_module, _args, kwargs):
            cache = kwargs.get("past_key_value")
            if cache is None or not hasattr(cache, "layers"):
                raise RuntimeError("private cache required")
            self.active["cache"] = cache
            self.active["phase"] = kwargs["position_embeddings"]

        def query(_module, _args, output):
            q = output.detach().view(1, 1, -1, 128).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, *self.active["phase"])
            self.active["query"] = q

        def capture(_module, args):
            layer = self.active["cache"].layers[-1]
            if not layer.leased or layer.history_length != len(self.records) + 1:
                raise RuntimeError("missing consecutive actual compressed K/V history")
            full = tuple(
                torch.cat((p, c[..., layer.prompt_length :, :]), dim=2)
                for p, c in zip(self.source, layer.leased, strict=True)
            )
            q = self.active["query"]
            scores = torch.matmul(q, full[0].transpose(2, 3)) * attention.scaling
            probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            alternative = torch.matmul(probabilities, full[1]).transpose(1, 2).contiguous()
            self.temporary_kv_peak_bytes = max(
                self.temporary_kv_peak_bytes, sum(t.numel() * t.element_size() for t in full)
            )
            self.records.append(
                dict(
                    residual=self.active["residual"],
                    vector=args[0].detach().clone(),
                    full_heads=alternative.reshape_as(args[0]).detach().clone(),
                )
            )

        def done(_module, _args, _output):
            self.active.clear()

        try:
            self.stack.callback(
                self.block.register_forward_pre_hook(residual, with_kwargs=True).remove
            )
            self.stack.callback(attention.register_forward_pre_hook(phase, with_kwargs=True).remove)
            self.stack.callback(attention.q_proj.register_forward_hook(query).remove)
            self.stack.callback(attention.o_proj.register_forward_pre_hook(capture).remove)
            self.stack.callback(self.block.register_forward_hook(done, always_call=True).remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.active.clear()


@torch.inference_mode()
def rerank(scout, norm, readout, source_logits, reference_logits, *, shortlist=4):
    """Batch only the heuristic. Verify a bounded shortlist with native B=1 tail."""
    if len(scout.records) != 32 or shortlist != 4:
        raise ValueError("registered 32-prefix / four-B1-candidate diagnostic required")
    block = scout.block
    h = scout.records[0]["vector"].shape[-1] // 128
    zero = torch.cat(
        [tail_logits(block, norm, readout, r["residual"], r["vector"]) for r in scout.records]
    )
    if not torch.equal(zero.cpu(), source_logits.cpu()):
        raise RuntimeError("B=1 unchanged nonlinear tail is not bitwise identical")
    batched = []
    for r in scout.records:
        vector = r["vector"].expand(h, -1, -1).clone()
        for head in range(h):
            vector[head, :, head * 128 : (head + 1) * 128] = r["full_heads"][
                0, :, head * 128 : (head + 1) * 128
            ]
        logits = tail_logits(block, norm, readout, r["residual"].expand(h, -1, -1), vector)
        batched.append(logits.cpu())
    stack = torch.stack(batched)
    scores = sorted(
        [dict(head=i, statistics=statistics(reference_logits, stack[:, i])) for i in range(h)],
        key=order,
    )
    verified, outputs = [], {}
    for row in scores[:shortlist]:
        head = row["head"]
        values = []
        for r in scout.records:
            vector = r["vector"].clone()
            vector[..., head * 128 : (head + 1) * 128] = r["full_heads"][
                ..., head * 128 : (head + 1) * 128
            ]
            values.append(tail_logits(block, norm, readout, r["residual"], vector).cpu())
        outputs[head] = torch.cat(values)
        verified.append(dict(head=head, statistics=statistics(reference_logits, outputs[head])))
    verified.sort(key=order)
    baseline = statistics(reference_logits, source_logits)
    acceptable = [
        r
        for r in verified
        if r["statistics"]["guard_pass"]
        and r["statistics"]["mean_KL_FP32"] <= baseline["mean_KL_FP32"]
    ]
    chosen = acceptable[0]["head"] if acceptable else None
    return dict(
        zero_tail_bitwise_identical=True,
        batched_ranking_is_heuristic=True,
        native_B1_verified=verified,
        heuristic_all_heads=scores,
        baseline=baseline,
        chosen_head=chosen,
        fail_closed_source_anchor=chosen is None,
        no_label_access=True,
        prefixes=32,
        shortlist=4,
        temporary_full_kv_workspace_bytes=scout.temporary_kv_peak_bytes,
        KV_unchanged_by_last_layer_output_intervention=True,
    ), outputs
