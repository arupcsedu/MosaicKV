"""All-layer confirmation of the frozen exact joint-row native configuration.

All original model calls remain in place. Extra decoded tails are diagnostic
workspace, never represented as production memory savings.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from mosaickv.evaluation import joint_kernel_compatibility as kernel
from mosaickv.evaluation import joint_row_feasibility as original

WINDOWS = original.WINDOWS
FROZEN = (31, 15, 1, 0, 0, 0, 18, 4)
current = original.current


def admissible(rows):
    keys = [(r["layer"], r["step"], r["component"]) for r in rows]
    expected = {
        (layer, step, component) for layer in range(32) for step in WINDOWS for component in (0, 1)
    }
    if len(keys) != 192 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("every layer, prefix and projection component is required")
    return all(
        len(r["kernels"]) == 1
        and kernel.fingerprint(r["kernels"][0]["configuration"]) == FROZEN
        and r["kernels"][0]["incoming_row"]["bitwise_equal"]
        and r["kernels"][0]["prompt_rows"]["bitwise_equal"]
        for r in rows
    )


class Probe(kernel.Probe):
    def capture(self, packets):
        from mosaickv.evaluation.head_local_precision import HeadLocalFormat
        from mosaickv.evaluation.lt_source_chunks import decode_slice
        from mosaickv.evaluation.shared_source_format import SharedSourceFormat

        if self.source is not None or self.index != -1 or len(packets) != 32:
            raise RuntimeError("one original 32-layer source capture before decoding required")
        packets = list(packets)
        for i, packet in enumerate(packets):
            while isinstance(packet, HeadLocalFormat):
                packet = packet.base
            if not isinstance(packet, SharedSourceFormat):
                raise NotImplementedError("same pinned fused shared source at every layer required")
            packets[i] = packet
        if (
            len({p.shape for p in packets}) != 1
            or packets[0].shape[1] * packets[0].shape[3] != 4096
        ):
            raise ValueError("identical full-MHA source geometry required")
        self.layout = original.geometry(packets[0].shape[2])
        self.sources = torch.empty(
            (32, 128, 4096), device=packets[0].cos.device, dtype=torch.bfloat16
        )
        self.output = torch.empty_like(self.sources[0])
        for i, packet in enumerate(packets):
            decode_slice(packet, self.sources[i], self.layout["first"], self.layout["prompt_rows"])
        self.source = self.sources[0]
        self.plans = kernel.KernelPlans((self.layout["prompt_rows"], self.layout["joint_rows"]))
        self.plans.candidates = [
            (a, r) for a, r in self.plans.candidates if kernel.fingerprint(r) == FROZEN
        ]
        if len(self.plans.candidates) != 1:
            raise RuntimeError("the frozen configuration is not vendor-admissible")

    def observe_layer(self, layer, component, module, hidden, result):
        if self.index not in WINDOWS or tuple(hidden.shape) != (1, 1, 4096):
            return
        self.source = self.sources[layer]
        super().observe(component, module, hidden, result)
        self.rows[-1]["layer"] = layer

    @contextmanager
    def installed(self):
        if original._probe.get() is not None:
            raise RuntimeError("one request-private all-layer diagnostic required")
        token = original._probe.set(self)
        original_step = self.adapter.decode_one_token

        def step(*args, **kwargs):
            self.index = self.steps
            self.steps += 1
            return original_step(*args, **kwargs)

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.adapter, "decode_one_token", step))
                for layer, block in enumerate(self.adapter._language_layers()):
                    for component, module in enumerate(
                        (block.self_attn.k_proj, block.self_attn.v_proj)
                    ):

                        def hook(module, args, result, *, layer=layer, component=component):
                            self.observe_layer(layer, component, module, args[0], result)

                        handle = module.register_forward_hook(hook)
                        stack.callback(handle.remove)
                yield self
        finally:
            try:
                if self.plans is not None:
                    torch.cuda.synchronize()
                    self.plans.close()
            finally:
                original._probe.reset(token)


def collect(adapter, call, folder):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    with Probe(adapter).installed() as state:
        measured = call()
        if (
            state.steps != 127
            or not measured["logits_bitwise"]
            or not measured["no_FullKV_fallback"]
        ):
            raise RuntimeError("complete unchanged 128-vector parent trajectory required")
        return dict(
            comparisons=state.rows,
            all_layer_exact=admissible(state.rows),
            frozen_configuration=FROZEN,
            observed_components=192,
            source_model_unchanged=True,
            parent_logits_bitwise=True,
            explicit_diagnostic_fixed_workspace_bytes=backing_bytes((state.sources, state.output)),
            original_packets_not_retained_by_probe=True,
            no_FullKV_fallback=True,
            no_runtime_integrated=True,
            no_latency_or_quality_claim=True,
            whole_request_GPU_peak_certified=False,
            independent_quality_N=0,
            native_ABI=state.plans.record(),
        )
