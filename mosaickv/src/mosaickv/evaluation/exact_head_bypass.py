"""Forecast-ranked exact full-head sidecars; isolated eager Llama diagnostic.

The ordinary cache remains unchanged. Only selected head outputs are recomputed
from original post-RoPE prompt K/V and privately appended current history. A
head-local buffer avoids expanding every head's dense prompt capacity.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@torch.inference_mode()
def rank_routes(attention_steps, source, formats, memberships, attentions):
    if len(attention_steps) != 4 or not source:
        raise ValueError("four isolated draft queries and complete source required")
    ranking = []
    for layer, (pair, packet, heads, attention) in enumerate(
        zip(source, formats, memberships, attentions, strict=True)
    ):
        if any(len(step) != len(source) for step in attention_steps):
            raise ValueError("incomplete draft layer evidence")
        k, v = pair
        h, n, d = k.shape[1:]
        if k.shape != v.shape or k.shape[0] != 1 or d != 128 or len(heads) != h:
            raise ValueError("batch-one full-MHA source and original head map required")
        selected_v = packet.materialize(attention)[1][0].float()
        ids = torch.tensor(heads, device=k.device, dtype=torch.long)
        if ids.shape != (h, packet.shape[-2]) or bool(((ids < 0) | (ids >= n)).any()):
            raise ValueError("unpadded original prompt memberships required")
        demands = []
        for step, values in enumerate(attention_steps):
            p = values[layer]
            if p.shape != (1, h, 1, n + step + 1) or not torch.isfinite(p).all() or (p < 0).any():
                raise ValueError("finite full-MHA draft probabilities required")
            p = p[0, :, 0, :n].float()
            demands.append(p / p.sum(-1, keepdim=True).clamp_min(1e-30))
        demand = torch.stack(demands, 1)
        kept = demand.gather(2, ids[:, None, :].expand(-1, 4, -1))
        kept = kept / kept.sum(-1, keepdim=True).clamp_min(1e-30)
        difference = torch.bmm(kept, selected_v) - torch.bmm(demand, v[0].float())
        weight = attention.o_proj.weight.float().reshape(-1, h, d).permute(1, 0, 2)
        projected = torch.einsum("hqd,hod->hqo", difference, weight)
        score = projected.square().mean((1, 2))
        if not torch.isfinite(score).all():
            raise ValueError("nonfinite route output impact")
        ranking.extend(
            dict(
                layer=layer,
                head=head,
                score=float(value),
                prompt_attention_output_error=float(difference[head].square().mean()),
            )
            for head, value in enumerate(score.cpu().tolist())
        )
    return tuple(sorted(ranking, key=lambda r: (-r["score"], r["layer"], r["head"])))


@dataclass(frozen=True)
class HeadBypassFormat:
    base: Any
    heads: tuple[int, ...]
    sidecar: tuple[Any, ...]
    original_prompt_length: int
    enabled: bool
    kind: str = "forecast_ranked_exact_head_bypass_diagnostic"

    def __post_init__(self):
        if (
            type(self.enabled) is not bool
            or not self.heads
            or tuple(sorted(set(self.heads))) != self.heads
            or any(type(h) is not int or not 0 <= h < self.base.shape[1] for h in self.heads)
            or len(self.sidecar) != 2
            or self.original_prompt_length < 1
            or any(
                t.shape != (1, len(self.heads), self.original_prompt_length + 128, 128)
                or t.dtype != torch.bfloat16
                or t.device != self.base.private()[0].device
                for t in self.sidecar
            )
        ):
            raise ValueError(
                "private full-head BF16 prompt/history owners and valid route IDs required"
            )

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), *self.sidecar)

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.base.fill(attention, targets, direct=direct)

    def materialize(self, attention):
        return self.base.materialize(attention)


@dataclass(frozen=True)
class HeadBypassSeed:
    base: Any
    heads: tuple[int, ...]
    prompt: tuple[Any, ...]
    enabled: bool = True

    @property
    def shape(self):
        return self.base.shape

    def fork_for_decode(self, updates):
        if type(updates) is not int or not 1 <= updates <= 128:
            raise ValueError("bounded diagnostic permits at most 128 updates")
        # Same capacity for fixed-prefix and greedy trials. Every decode handle
        # gets fresh owners: seeds never contain mutable generated history.
        shape = list(self.prompt[0].shape)
        n = shape[-2]
        shape[-2] += 128
        buffers = tuple(torch.zeros(shape, device=p.device, dtype=p.dtype) for p in self.prompt)
        for target, source in zip(buffers, self.prompt, strict=True):
            target[..., :n, :].copy_(source)
        return HeadBypassFormat(self.base, self.heads, buffers, n, self.enabled)


@torch.inference_mode()
def construct(formats, source, routes, *, enabled=True):
    if type(enabled) is not bool or not routes or len(set(routes)) != len(routes):
        raise ValueError("unique exact-head routes and explicit enabled flag required")
    result = []
    for layer, (packet, (key, value)) in enumerate(zip(formats, source, strict=True)):
        if hasattr(packet, "valid_prompt_counts"):
            raise NotImplementedError("exact-head source diagnostic needs unpadded base heads")
        heads = tuple(sorted(head for i, head in routes if i == layer))
        if not heads:
            result.append(packet)
            continue
        if (
            any(type(h) is not int or not 0 <= h < key.shape[1] for h in heads)
            or key.shape != value.shape
            or key.shape[0] != 1
            or key.shape[-1] != 128
            or key.dtype != torch.bfloat16
            or value.dtype != key.dtype
        ):
            raise ValueError("original BF16 full-head cache and valid route IDs required")
        ids = torch.tensor(heads, device=key.device)
        prompt = tuple(t.index_select(1, ids).detach().contiguous() for t in (key, value))
        if not all(
            torch.equal(p, t.index_select(1, ids))
            for p, t in zip(prompt, (key, value), strict=True)
        ):
            raise RuntimeError("exact-head source copy changed cache bits")
        result.append(HeadBypassSeed(packet, heads, prompt, enabled))
    if sum(len(p.heads) for p in result if isinstance(p, HeadBypassSeed)) != len(routes):
        raise ValueError("route layer outside original source cache")
    return tuple(result)


class ExactHeadTransport:
    def __init__(self, adapter, cache):
        self.adapter, self.cache = adapter, cache
        self.stack, self.active, self.expected = ExitStack(), {}, {}
        self.recomputed_heads = 0

    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self.stack.close()
            self.active.clear()
            raise

    def _install(self):
        from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

        for index, block in enumerate(self.adapter._language_layers()):
            layer = self.cache.layers[index]
            fmt = layer.representation
            if not isinstance(fmt, HeadBypassFormat):
                continue
            attention = block.self_attn
            if (
                not isinstance(attention, LlamaAttention)
                or attention.training
                or attention.config._attn_implementation != "eager"
                or attention.num_key_value_groups != 1
                or attention.head_dim != 128
            ):
                raise NotImplementedError(
                    "isolated eval-only eager full-MHA Llama head bypass required"
                )
            self.expected[index] = 0

            def before(_module, args, kwargs, *, i=index, target=layer):
                if self.active or kwargs.get("past_key_value") is not self.cache:
                    raise RuntimeError("nested/foreign cache in exact-head transport")
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                phase = kwargs.get("position_embeddings")
                if hidden is None or hidden.shape[:2] != (1, 1) or phase is None:
                    raise NotImplementedError(
                        "one query with actual original rotary phase required"
                    )
                if target.history_length != self.expected[i]:
                    raise RuntimeError("sidecar generation position changed")
                self.active[i] = dict(phase=phase)

            def projected(_module, _args, output, *, i=index, module=attention):
                if i not in self.active:
                    raise RuntimeError("query without active request")
                query = output.detach().view(1, 1, -1, 128).transpose(1, 2)
                cos, sin = self.active[i]["phase"]
                query, _ = apply_rotary_pos_emb(query, query, cos, sin)
                self.active[i]["query"] = query

            def replace_output(
                _module, args, *, i=index, target=layer, packet=fmt, module=attention
            ):
                state = self.active.get(i)
                if state is None or "query" not in state or not target.leased:
                    raise RuntimeError("missing actual query or current K/V lease")
                step = target.history_length
                if step != self.expected[i] + 1 or step > 128:
                    raise RuntimeError("sidecar current decode position is not consecutive")
                vector = args[0]
                full_n = packet.original_prompt_length
                for slot, head in enumerate(packet.heads):
                    for sidecar, current in zip(packet.sidecar, target.leased, strict=True):
                        sidecar[:, slot : slot + 1, full_n + step - 1 : full_n + step, :].copy_(
                            current[
                                :,
                                head : head + 1,
                                target.prompt_length + step - 1 : target.prompt_length + step,
                                :,
                            ]
                        )
                    if packet.enabled:
                        key, value = (
                            t[:, slot : slot + 1, : full_n + step, :] for t in packet.sidecar
                        )
                        query = state["query"][:, head : head + 1]
                        scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
                        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                            query.dtype
                        )
                        output = torch.matmul(probabilities, value).reshape(1, 1, 128)
                        vector[..., head * 128 : (head + 1) * 128].copy_(output)
                        self.recomputed_heads += 1
                self.expected[i] = step
                return args

            def after(_module, _args, _output, *, i=index):
                self.active.pop(i, None)

            self.stack.callback(
                attention.register_forward_pre_hook(before, with_kwargs=True).remove
            )
            self.stack.callback(attention.q_proj.register_forward_hook(projected).remove)
            self.stack.callback(attention.o_proj.register_forward_pre_hook(replace_output).remove)
            self.stack.callback(attention.register_forward_hook(after, always_call=True).remove)
        return self

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.active.clear()
