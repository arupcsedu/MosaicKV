"""Default-off common Llama input RMSNorm replay, not fused arithmetic.

Replay exactly the pinned float32 pow/mean/rsqrt/mul, BF16 conversion and
weight multiplication. One request owns every buffer/graph. The first token
uses the original module before prompt graph metadata has been initialized.
"""

from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_scope = ContextVar("evicastkv_common_input_norm", default=None)


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        enabled = kwargs.get("input_norm_segment", False)
        reuse = kwargs.get("reuse_norm_input", False)
        if type(reuse) is not bool or (reuse and not enabled):
            raise ValueError("explicit norm input reuse requires the common norm transport")
        if type(enabled) is not bool:
            raise ValueError("explicit input norm boolean required")
        if enabled and kwargs.get("post_attention_segment", False) is not True:
            raise ValueError("registered common post-attention transport required")
        if _scope.get() is not None:
            raise RuntimeError("input norm request scopes cannot nest")
        state = dict(enabled=enabled, reuse=reuse, report=None)
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            details["common_input_norm"] = state["report"] or dict(enabled=False)
            if enabled and state["report"] is None:
                raise RuntimeError("declared input norm transport did not engage")
            return values, packets, details
        finally:
            _scope.reset(token)

    return call


class Workspace:
    def __init__(self, norm):
        (hidden,) = norm.weight.shape
        options = dict(device=norm.weight.device)
        self.input, self.output = (
            torch.empty((1, hidden), dtype=norm.weight.dtype, **options) for _ in range(2)
        )
        self.float_x, self.square = (
            torch.empty((1, hidden), dtype=torch.float32, **options) for _ in range(2)
        )
        self.variance = torch.empty((1, 1), dtype=torch.float32, **options)

    def tensors(self):
        return tuple(vars(self).values())

    def compute(self, norm, source=None):
        self.float_x.copy_(self.input if source is None else source.reshape_as(self.input))
        torch.pow(self.float_x, 2, out=self.square)
        torch.mean(self.square, -1, keepdim=True, out=self.variance)
        self.variance.add_(norm.variance_epsilon)
        torch.rsqrt(self.variance, out=self.variance)
        torch.mul(self.float_x, self.variance, out=self.float_x)
        self.output.copy_(self.float_x)
        torch.mul(norm.weight, self.output, out=self.output)
        return self.output.view(1, 1, -1)


class Transport:
    def __init__(self, adapter, cache, state):
        self.layers = tuple(adapter._language_layers())
        self.cache, self.state = cache, state
        self.stack, self.entries = ExitStack(), {}
        self.workspace = None
        self.calls = self.cold = self.replays = self.captures = self.extra_graph_bytes = 0
        self.bytes = 0
        self.closed = False
        self.shared_input = None
        self.skipped_copies = self.first_layer_copies = 0

    def bind_segment(self, segment):
        """One request's already-owned epilogue output, never a new buffer."""
        if not self.state.get("reuse", False):
            return
        if (
            self.shared_input is not None
            or segment is None
            or segment.cache is not self.cache
            or segment.closed
            or segment.workspace is None
            or segment.workspace.output.shape != self.workspace.input.shape
            or segment.workspace.output.dtype != self.workspace.input.dtype
            or segment.workspace.output.device != self.workspace.input.device
        ):
            raise RuntimeError("same-request live epilogue output must bind exactly once")
        self.shared_input = segment.workspace.output

    def prepare_input(self, hidden, index):
        if not self.state.get("reuse", False):
            self.workspace.input.copy_(hidden.reshape_as(self.workspace.input))
            return
        if self.shared_input is None:
            raise RuntimeError("input reuse not bound to its epilogue owner")
        if index == 0:
            if hidden.data_ptr() == self.shared_input.data_ptr():
                raise RuntimeError("original token embedding must not alias epilogue output")
            self.shared_input.copy_(hidden.reshape_as(self.shared_input))
            self.first_layer_copies += 1
        else:
            if hidden.data_ptr() != self.shared_input.data_ptr() or not hidden.is_contiguous():
                raise RuntimeError("later layers must consume the exact preceding epilogue output")
            self.skipped_copies += 1

    def __enter__(self):
        from transformers.models.llama import modeling_llama as llama

        if (
            len(self.layers) != 32
            or self.cache.prompt_graph_pool is None
            or any(
                type(layer) is not llama.LlamaDecoderLayer
                or layer.training
                or layer.self_attn.config._attn_implementation != "eager"
                or type(layer.input_layernorm) is not llama.LlamaRMSNorm
                or layer.input_layernorm.weight.shape != (4096,)
                or layer.input_layernorm.weight.dtype != torch.bfloat16
                or layer.input_layernorm.weight.device.type != "cuda"
                or "forward" in vars(layer.input_layernorm)
                or layer.input_layernorm._forward_hooks
                or layer.input_layernorm._forward_pre_hooks
                for layer in self.layers
            )
        ):
            raise NotImplementedError("private pinned BF16 eager Llama RMSNorm required")
        self.workspace = Workspace(self.layers[0].input_layernorm)
        self.bytes = backing_bytes(self.workspace.tensors())
        self.cache.observe(self.workspace.tensors())
        self.cache.shared_tensors = (*self.cache.shared_tensors, *self.workspace.tensors())
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.cache.observe()
        try:
            for index, layer in enumerate(self.layers):
                norm = layer.input_layernorm
                self.stack.enter_context(patch.object(norm, "forward", self.forward(norm, index)))
            return self
        except BaseException:
            self.close()
            raise

    def forward(self, norm, index):
        original = norm.forward

        def call(hidden):
            if (
                self.closed
                or torch.is_grad_enabled()
                or hidden.shape != (1, 1, 4096)
                or hidden.dtype != torch.bfloat16
                or index != self.calls % 32
            ):
                raise RuntimeError("isolated ordered one-token input norm required")
            self.calls += 1
            if self.calls <= 32:
                self.cold += 1
                return original(hidden)
            self.prepare_input(hidden, index)
            graph = self.entries.get(index)
            if graph is None:
                self.capture(norm, index)
                graph = self.entries[index]
            graph.replay()
            self.replays += 1
            return self.workspace.output.view(1, 1, -1)

        return call

    def capture(self, norm, index):
        from mosaickv.evaluation.post_attention_segment import capture_stream

        if not self.cache.prompt_graph_pool.entries:
            raise RuntimeError("prompt graph metadata must exist before norm capture")
        current = torch.cuda.current_stream()
        stream = capture_stream(self.workspace.output.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.workspace.compute(norm, self.shared_input)
        current.wait_stream(stream)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    self.workspace.compute(norm, self.shared_input)
                finally:
                    graph.capture_end()
            current.wait_stream(stream)
            torch.cuda.synchronize()
            extra = torch.cuda.memory_allocated() - before
            if extra:
                self.extra_graph_bytes = extra
                raise RuntimeError(f"input norm graph introduced {extra} unowned bytes")
        except BaseException:
            graph.reset()
            raise
        self.entries[index] = graph
        self.captures += 1

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
            t for t in self.cache.shared_tensors if all(t is not owner for owner in owners)
        )
        self.workspace = None
        self.shared_input = None
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.cache.observe()
        self.closed = True

    def __exit__(self, kind, value, traceback):
        self.close()
        if kind is None and (self.calls, self.cold, self.captures, self.replays) != (
            4064,
            32,
            32,
            4032,
        ):
            raise RuntimeError("complete 127-update common input norm trajectory required")
        if (
            kind is None
            and self.state.get("reuse", False)
            and (self.skipped_copies, self.first_layer_copies) != (3906, 126)
        ):
            raise RuntimeError("all later-layer norm copies must bypass with 126 embedding copies")
        self.state["report"] = dict(
            enabled=True,
            calls=self.calls,
            original_cold_calls=self.cold,
            captures=self.captures,
            replays=self.replays,
            explicit_workspace_bytes=self.bytes,
            additional_graph_tensor_bytes=self.extra_graph_bytes,
            retired=self.closed,
            common_to_every_method=True,
            input_reuse_requested=self.state.get("reuse", False),
            skipped_later_layer_input_copies=self.skipped_copies,
            first_layer_embedding_copies=self.first_layer_copies,
            unused_original_input_buffer_still_owned_and_charged=True,
            original_operator_rounding=True,
            whole_request_GPU_peak_certified=False,
        )


def transport(adapter, cache):
    state = _scope.get()
    return (
        nullcontext() if state is None or not state["enabled"] else Transport(adapter, cache, state)
    )


@torch.inference_mode()
def gpu_controls():
    """No benchmark/model data: exact original versus explicit-buffer norm."""
    from transformers.models.llama import modeling_llama as llama

    from mosaickv.evaluation.post_attention_segment import capture_stream
    from mosaickv.evaluation.prompt_graph_reader import allocator_blocks, generator_metadata

    with torch.inference_mode(False), torch.no_grad(), torch.device("cuda"):
        norm = llama.LlamaRMSNorm(4096).to(dtype=torch.bfloat16).eval()
    work = Workspace(norm)
    current, stream = torch.cuda.current_stream(), capture_stream(work.output.device)
    work.input.zero_()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        work.compute(norm)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    blocks, before = allocator_blocks(), torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            work.compute(norm)
        finally:
            graph.capture_end()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    extra = torch.cuda.memory_allocated() - before
    metadata = generator_metadata(blocks, allocator_blocks(), extra) if extra else ()
    observations, version = [], norm.weight._version
    try:
        for scale in (0.0, 0.01, 1.0, 50.0):
            hidden = torch.randn_like(work.input).mul_(scale).view(1, 1, -1)
            reference = norm(hidden)
            work.input.copy_(hidden.reshape_as(work.input))
            graph.replay()
            torch.cuda.synchronize()
            differences = int(
                (
                    work.output.view(torch.int16).reshape(-1)
                    != reference.view(torch.int16).reshape(-1)
                ).sum()
            )
            observations.append(dict(scale=scale, differing_words=differences))
            if differences:
                raise RuntimeError("input norm graph changed original BF16 words")
        if norm.weight._version != version:
            raise RuntimeError("input norm weight mutated")
    finally:
        graph.reset()
        del graph
        torch.cuda.synchronize()
    if any(pointer in allocator_blocks() for pointer in metadata):
        raise RuntimeError("norm control retained graph metadata")
    return dict(
        complete=True,
        observations=observations,
        workspace_bytes=backing_bytes(work.tensors()),
        no_benchmark_or_checkpoint_data=True,
        bitwise=True,
    )
