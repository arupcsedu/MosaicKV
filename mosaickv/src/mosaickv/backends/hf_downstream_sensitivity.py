"""Frozen anchor decoder derivatives, not candidate forwards or a training path."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from mosaickv.backends.hf_prefix_outputs import PrefixOutputRecorder


def normal_clone(value):
    import torch

    # Decoder callbacks run inside inference_mode. Autodiff needs ordinary,
    # isolated tensors, not inference tensors carrying mutable cache views.
    with torch.inference_mode(False):
        return value.detach().clone()


@contextmanager
def frozen_parameters(model):
    parameters = tuple(model.parameters())
    flags = tuple(p.requires_grad for p in parameters)
    versions = tuple(p._version for p in parameters)
    if any(p.grad is not None for p in parameters):
        raise ValueError("diagnostic requires no pre-existing parameter gradients")
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
        if any(p.grad is not None for p in parameters):
            raise RuntimeError("downstream diagnostic accumulated parameter gradients")
        if versions != tuple(p._version for p in parameters):
            raise RuntimeError("downstream diagnostic modified frozen parameter storage")
    finally:
        for p, flag in zip(parameters, flags, strict=True):
            p.requires_grad_(flag)


class ReadOnlyPrefixCache:
    def __init__(self, pairs):
        self.pairs = pairs

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        import torch

        del cache_kwargs
        old_k, old_v = self.pairs[layer_idx]
        return torch.cat((old_k, keys), -2), torch.cat((old_v, values), -2)


class DownstreamCapture:
    """Capture two pre-overwrite anchor prefixes, no candidate activation access."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        if any(module.training for module in adapter.model.modules()):
            raise ValueError("downstream diagnostic requires an evaluation-mode frozen model")
        self.recorder = PrefixOutputRecorder(adapter)
        self.layers = self.recorder.layers
        self.snapshots: dict[int, Any] = {}
        self.capture_seconds = 0.0
        self.progress: dict[str, Any] = dict(status="not_started", prefixes=[])

    @contextmanager
    def capture(self, role, step):
        if role != "anchor" or step not in (1, 4):
            yield
            return
        if step in self.snapshots:
            raise ValueError("duplicate downstream snapshot")
        import torch

        start = time.perf_counter()
        rows = [None] * len(self.layers)
        handles = []

        def hook(index):
            def record(_module, _args, kwargs):
                cache = kwargs["past_key_value"].layers[index]
                if not hasattr(cache, "_updates") or cache._updates != step - 1:
                    raise ValueError("downstream snapshot requires an unwrapped bounded prefix")
                if cache._updates >= cache.generated_capacity:
                    raise ValueError("downstream snapshot cannot append after a ring overwrite")
                mask = kwargs["attention_mask"]
                rows[index] = dict(
                    keys=normal_clone(cache.keys),
                    values=normal_clone(cache.values),
                    mask=None if mask is None else normal_clone(mask),
                    position_embeddings=tuple(
                        normal_clone(t) for t in kwargs["position_embeddings"]
                    ),
                    cache_position=normal_clone(kwargs["cache_position"]),
                )

            return record

        try:
            for index, layer in enumerate(self.layers):
                handles.append(
                    layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True)
                )
            with self.recorder.capture(role, step):
                yield
            if any(row is None for row in rows):
                raise RuntimeError("incomplete downstream cache capture")
            self.snapshots[step] = rows
        finally:
            for handle in handles:
                handle.remove()
            if self.adapter.device.type == "cuda":
                torch.cuda.synchronize(self.adapter.device)
            self.capture_seconds += time.perf_counter() - start

    @property
    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for rows in self.snapshots.values()
            for row in rows
            for t in (row["keys"], row["values"])
        )

    def function(self, step):
        import torch

        root = getattr(self.adapter.model, "model", self.adapter.model)
        language = root.language_model
        norm = getattr(language, "norm", None)
        if norm is None:
            norm = language.model.norm
        head = self.adapter.model.get_output_embeddings()
        rows = self.snapshots[step]
        initial = self.recorder.records["anchor"][step][0]["residual_input"]
        initial = normal_clone(initial).to(self.adapter.device).reshape(1, 1, -1)
        pairs = tuple((row["keys"], row["values"]) for row in rows)

        def forward(delta):
            hidden = initial
            cache = ReadOnlyPrefixCache(pairs)
            for i, (layer, row) in enumerate(zip(self.layers, rows, strict=True)):
                attn, _ = layer.self_attn(
                    hidden_states=layer.input_layernorm(hidden),
                    position_embeddings=row["position_embeddings"],
                    attention_mask=row["mask"],
                    past_key_value=cache,
                    cache_position=row["cache_position"],
                )
                residual = hidden + attn + delta[i]
                hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
            return head(norm(hidden))[:, -1].float()

        return forward, torch.zeros(
            (len(self.layers), *initial.shape), dtype=initial.dtype, device=initial.device
        )

    def gradients(self, references, anchors, *, jvp_direction=None):
        import torch

        def sync():
            if self.adapter.device.type == "cuda":
                torch.cuda.synchronize(self.adapter.device)

        sync()
        start = time.perf_counter()
        results, records = {}, []
        self.progress = dict(status="running", started_monotonic=start, prefixes=records)
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            frozen_parameters(self.adapter.model),
        ):
            for step in (1, 4):
                self.progress.update(prefix=step, phase="anchor_primal")
                function, delta = self.function(step)
                delta.requires_grad_(True)
                z = function(delta)
                expected = anchors[step].to(z.device).float()
                if not torch.equal(z, expected):
                    raise RuntimeError(
                        "isolated downstream primal did not bitwise replay anchor logits"
                    )
                lp = normal_clone(references[step]).to(z.device).float().log_softmax(-1)
                loss = (lp.exp() * (lp - z.log_softmax(-1))).sum()
                self.progress["phase"] = "shared_VJP"
                # Fixed scaling mitigates FP16 input-gradient underflow. It is
                # recorded sideband arithmetic, never an acceptance tolerance.
                gradient = torch.autograd.grad(loss * 1024.0, delta)[0].detach().double() / 1024.0
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("nonfinite downstream derivative")
                results[step] = gradient[:, 0, 0].cpu()
                record = dict(
                    prefix=step,
                    primal_bitwise_replayed=True,
                    KL=float(loss.detach()),
                    loss_scale=1024.0,
                    derivative="shared_anchor_VJP",
                )
                if step == 1 and jvp_direction is not None:
                    self.progress["phase"] = "JVP_crosscheck"
                    sync()
                    jvp_start = time.perf_counter()
                    direction = jvp_direction.to(device=delta.device, dtype=delta.dtype).reshape_as(
                        delta
                    )
                    if not bool(torch.isfinite(direction).all()):
                        raise RuntimeError("nonfinite JVP cross-check direction")
                    norm = direction.float().norm()
                    basis_fallback = bool(norm == 0)
                    if basis_fallback:
                        direction = torch.zeros_like(delta)
                        direction.view(-1)[0] = 1
                    else:
                        direction = (direction.float() / norm).to(delta.dtype)
                    with torch.no_grad():
                        primal, tangent = torch.func.jvp(function, (delta.detach(),), (direction,))
                    if not torch.equal(primal, expected):
                        raise RuntimeError("JVP primal changed anchor logits")
                    q = primal.softmax(-1)
                    direct = ((q - lp.exp()) * tangent).sum().double()
                    reverse = (gradient * direction.double()).sum()
                    # This sideband derivative check is not the registered guard.
                    if not torch.isclose(direct, reverse, rtol=0.05, atol=1e-5):
                        raise RuntimeError(
                            "forward JVP / shared VJP directional cross-check failed"
                        )
                    sync()
                    record.update(
                        jvp_directional_KL=float(direct),
                        vjp_directional_KL=float(reverse),
                        jvp_seconds=time.perf_counter() - jvp_start,
                        jvp_crosscheck_passed=True,
                        jvp_atol=1e-5,
                        jvp_rtol=0.05,
                        jvp_direction_normalized_to_unit_L2=True,
                        jvp_fixed_basis_for_zero_direction=basis_fallback,
                    )
                records.append(record)
                sync()
                if time.perf_counter() - start > 120:
                    raise RuntimeError("bounded downstream derivative exceeded 120 seconds")
        self.progress.update(status="completed", phase="complete")
        return results, dict(
            seconds=time.perf_counter() - start,
            prefixes=records,
            snapshot_KV_bytes=self.storage_bytes,
            parameters_updated=False,
            past_cache_held_fixed=True,
            candidate_forward_required=False,
            full_autoregressive_Jacobian=False,
        )
