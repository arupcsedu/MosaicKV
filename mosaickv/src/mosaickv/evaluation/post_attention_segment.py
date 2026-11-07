"""Default-off, exact-operator Llama epilogue replay for every cache method.

Original reference: installed Transformers LlamaDecoderLayer/LlamaRMSNorm/
LlamaMLP (Apache-2.0). No fused arithmetic, approximation, weight modification,
attention capture or selection changes. Fixed output buffers make the extra
tensor ownership explicit. Only private, single-token inference is supported.
"""

from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_scope = ContextVar("evicastkv_post_attention_segment", default=None)
_streams = {}  # CUDA streams only: no weights, cache, activations or request owners.


def capture_stream(device):
    key = str(device)
    if key not in _streams:
        _streams[key] = torch.cuda.Stream(device=device)
    return _streams[key]


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        enabled = kwargs.get("post_attention_segment", False)
        if type(enabled) is not bool:
            raise ValueError("explicit post-attention segment boolean required")
        if _scope.get() is not None:
            raise RuntimeError("post-attention request scopes cannot nest")
        state = dict(enabled=enabled, report=None)
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            details["post_attention_segment"] = state["report"] or dict(enabled=False)
            if enabled and state["report"] is None:
                raise RuntimeError("explicit segment path did not engage")
            return values, packets, details
        finally:
            _scope.reset(token)

    return call


class Workspace:
    def __init__(self, layer):
        weight = layer.mlp.gate_proj.weight
        intermediate, hidden = weight.shape
        options = dict(device=weight.device, dtype=weight.dtype)
        self.residual, self.attention, self.hidden, self.normalized, self.output = (
            torch.empty((1, hidden), **options) for _ in range(5)
        )
        self.gate, self.up = (torch.empty((1, intermediate), **options) for _ in range(2))
        self.float_x, self.square = (
            torch.empty((1, hidden), device=weight.device, dtype=torch.float32) for _ in range(2)
        )
        self.variance = torch.empty((1, 1), device=weight.device, dtype=torch.float32)

    def tensors(self):
        return tuple(vars(self).values())

    def fill(self, residual, attention):
        self.residual.copy_(residual.reshape_as(self.residual))
        self.attention.copy_(attention.reshape_as(self.attention))

    def compute(self, layer):
        """Exactly the pinned out-of-place reference's operations and rounding."""
        torch.add(self.residual, self.attention, out=self.hidden)
        self.float_x.copy_(self.hidden)
        torch.pow(self.float_x, 2, out=self.square)
        torch.mean(self.square, -1, keepdim=True, out=self.variance)
        self.variance.add_(layer.post_attention_layernorm.variance_epsilon)
        torch.rsqrt(self.variance, out=self.variance)
        torch.mul(self.float_x, self.variance, out=self.float_x)
        self.normalized.copy_(self.float_x)
        torch.mul(layer.post_attention_layernorm.weight, self.normalized, out=self.normalized)
        torch.mm(self.normalized, layer.mlp.gate_proj.weight.t(), out=self.gate)
        torch.mm(self.normalized, layer.mlp.up_proj.weight.t(), out=self.up)
        functional.silu(self.gate, inplace=True)
        torch.mul(self.gate, self.up, out=self.gate)
        torch.mm(self.gate, layer.mlp.down_proj.weight.t(), out=self.output)
        torch.add(self.hidden, self.output, out=self.output)
        return self.output.view(1, 1, -1)


class Transport:
    def __init__(self, adapter, cache, state):
        self.layers = tuple(adapter._language_layers())
        self.cache, self.state = cache, state
        self.stack, self.entries = ExitStack(), {}
        self.workspace = None
        self.calls = self.captures = self.replays = 0
        self.closed = False
        self.bytes = self.extra_graph_bytes = 0
        self.backend_warm_allocation_bytes = 0

    def __enter__(self):
        from transformers.models.llama import modeling_llama as llama

        if (
            len(self.layers) != 32
            or self.cache.prompt_graph_pool is None
            or any(
                type(layer) is not llama.LlamaDecoderLayer
                or layer.training
                or layer.self_attn.config._attn_implementation != "eager"
                or layer.self_attn.config.hidden_act != "silu"
                or layer.self_attn.config.output_hidden_states
                or layer.mlp.gate_proj.weight.shape != (11008, 4096)
                or layer.mlp.gate_proj.weight.device.type != "cuda"
                or layer.mlp.gate_proj.weight.dtype != torch.bfloat16
                or any(
                    proj.bias is not None
                    for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
                )
                or any(
                    module._forward_hooks or module._forward_pre_hooks
                    for module in (layer, layer.mlp, layer.post_attention_layernorm)
                )
                for layer in self.layers
            )
        ):
            raise NotImplementedError(
                "private pinned BF16 eager Llama layers without activation-retaining hooks required"
            )
        self.workspace = Workspace(self.layers[0])
        self.bytes = backing_bytes(self.workspace.tensors())
        self.cache.observe(self.workspace.tensors())
        self.cache.shared_tensors = (*self.cache.shared_tensors, *self.workspace.tensors())
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.cache.observe()
        try:
            for index, layer in enumerate(self.layers):
                self.stack.enter_context(patch.object(layer, "forward", self.forward(layer, index)))
            return self
        except BaseException:
            self.close()
            raise

    def forward(self, layer, index):
        def call(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            if (
                self.closed
                or hidden_states.shape != (1, 1, 4096)
                or past_key_value is not self.cache
                or torch.is_grad_enabled()
            ):
                raise RuntimeError("isolated one-token segment decode required")
            normalized = layer.input_layernorm(hidden_states)
            attention, _ = layer.self_attn(
                hidden_states=normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return self.run(layer, index, hidden_states, attention)

        return call

    def run(self, layer, index, residual, attention):
        if index != self.calls % len(self.layers):
            raise RuntimeError("strict per-token epilogue ordering required")
        self.calls += 1
        self.workspace.fill(residual, attention)
        graph = self.entries.get(index)
        if graph is None:
            # The first native attention call already registered the prompt
            # graph's two scalar owners. No extra hidden graph tensors allowed.
            if not self.cache.prompt_graph_pool.entries:
                raise RuntimeError(
                    "prompt graph metadata ownership must exist before segment capture"
                )
            current = torch.cuda.current_stream()
            stream = capture_stream(self.workspace.output.device)
            stream.wait_stream(current)
            warm_before = torch.cuda.memory_allocated()
            with torch.cuda.stream(stream):
                self.workspace.compute(layer)
            current.wait_stream(stream)
            torch.cuda.synchronize()
            self.backend_warm_allocation_bytes += max(
                0, torch.cuda.memory_allocated() - warm_before
            )
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        self.workspace.compute(layer)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                torch.cuda.synchronize()
                extra = torch.cuda.memory_allocated() - before
                if extra != 0:
                    self.extra_graph_bytes = extra
                    raise RuntimeError(f"segment graph introduced {extra} unowned tensor bytes")
            except BaseException:
                graph.reset()
                raise
            self.entries[index] = graph
            self.captures += 1
        graph.replay()
        self.replays += 1
        return self.workspace.output.view(1, 1, -1)

    def close(self):
        if self.closed:
            return
        self.stack.close()
        torch.cuda.synchronize()
        for graph in self.entries.values():
            graph.reset()
        self.entries.clear()
        owners = () if self.workspace is None else self.workspace.tensors()
        self.cache.shared_tensors = tuple(
            t for t in self.cache.shared_tensors if all(t is not x for x in owners)
        )
        self.workspace = None
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.cache.observe()
        self.closed = True

    def __exit__(self, kind, value, traceback):
        self.close()
        from mosaickv.evaluation.bounded_probe_horizon import expected_updates

        calls = 32 * expected_updates()
        if kind is None and (self.calls != calls or self.captures != 32 or self.replays != calls):
            raise RuntimeError("complete registered common epilogue trajectory required")
        self.state["report"] = dict(
            enabled=True,
            calls=self.calls,
            captures=self.captures,
            replays=self.replays,
            explicit_workspace_bytes=self.bytes,
            additional_graph_tensor_bytes=self.extra_graph_bytes,
            backend_warm_allocation_bytes=self.backend_warm_allocation_bytes,
            backend_workspace_NOT_certified_as_KV=True,
            retired=self.closed,
            no_attention_or_KV_captured=True,
            no_arithmetic_fusion=True,
            common_to_every_method=True,
            whole_request_GPU_peak_certified=False,
        )


def transport(adapter, cache):
    state = _scope.get()
    return (
        nullcontext() if state is None or not state["enabled"] else Transport(adapter, cache, state)
    )


@torch.inference_mode()
def gpu_controls():
    """Real 4096/11008 BF16 operators and output buffers, before model loading."""
    from transformers.models.llama import modeling_llama as llama

    from mosaickv.evaluation.prompt_graph_reader import allocator_blocks, generator_metadata

    torch.manual_seed(20261105)
    config = llama.LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )
    layer = control_layer(config, "cuda")
    workspace = Workspace(layer)
    current, stream = torch.cuda.current_stream(), capture_stream(workspace.output.device)
    workspace.fill(torch.ones_like(workspace.residual), torch.zeros_like(workspace.attention))
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        workspace.compute(layer)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    before_blocks, before = allocator_blocks(), torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            workspace.compute(layer)
        finally:
            graph.capture_end()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    extra = torch.cuda.memory_allocated() - before
    metadata = generator_metadata(before_blocks, allocator_blocks(), extra) if extra else ()
    observations = []
    versions = [p._version for p in layer.parameters()]
    try:
        for scale in (0.0, 0.01, 1.0, 50.0):
            residual = torch.randn((1, 1, 4096), device="cuda", dtype=torch.bfloat16) * scale
            attention = torch.randn_like(residual)
            hidden = residual + attention
            reference = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            workspace.fill(residual, attention)
            graph.replay()
            torch.cuda.synchronize()
            differences = int(
                (
                    workspace.output.view(torch.int16).reshape(-1)
                    != reference.view(torch.int16).reshape(-1)
                ).sum()
            )
            observations.append(dict(scale=scale, differing_words=differences))
            if differences:
                raise RuntimeError("fixed-buffer epilogue changed original BF16 output words")
        if versions != [p._version for p in layer.parameters()]:
            raise RuntimeError("epilogue modified model weights")
    finally:
        graph.reset()
        del graph
        torch.cuda.synchronize()
    if any(pointer in allocator_blocks() for pointer in metadata):
        raise RuntimeError("operator control retained graph generator owners")
    return dict(
        complete=True,
        observations=observations,
        workspace_bytes=backing_bytes(workspace.tensors()),
        graph_tensor_bytes=extra,
        bitwise_reference=True,
        model_weights_unchanged=True,
        no_checkpoint_or_benchmark_inputs=True,
    )


def control_layer(config, device):
    """Control parameters need version counters even inside inference scope."""
    from transformers.models.llama import modeling_llama as llama

    with torch.inference_mode(False), torch.no_grad(), torch.device(device):
        return llama.LlamaDecoderLayer(config, 0).to(dtype=torch.bfloat16).eval()
