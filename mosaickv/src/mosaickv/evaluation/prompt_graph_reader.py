"""Private prompt-reader graphs; never capture attention, history or a model.

The immutable prompt reader alone writes to one charged request-wide lease.
Ordered layer-code replay, generated history and native eager attention stay
outside the graph. The default runtime is unchanged. This is an execution
diagnostic for already consumed inputs, not independent quality evidence.
"""

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
from mosaickv.evaluation.source_history import SourceHistoryCache


class PromptGraphPool:
    def __init__(self, cache):
        self.cache, self.buffer, self.entries = cache, None, {}
        self.closed = False
        self.captures = self.replays = self.warm_fills = 0
        self.maximum_workspace_bytes = 0
        self.checked_private_tensor_bytes = 0

    def acquire(self, layer, shape, source):
        if self.closed or any(x.leased for x in self.cache.layers):
            raise RuntimeError("private nonoverlapping prompt-graph lease required")
        if source.device.type != "cuda" or source.dtype != torch.bfloat16:
            raise NotImplementedError("prompt reader replay requires CUDA BF16")
        # A run emits `updates` predictions and performs `updates - 1` updates.
        # Fail explicitly if any caller tries to exceed that registered horizon.
        capacity = max(x.prompt_length for x in self.cache.layers) + max(
            x.maximum_decode_updates - 1 for x in self.cache.layers
        )
        if shape[:2] != layer.prompt_shape[:2] or not 1 <= shape[-2] <= capacity:
            raise ValueError("registered layer geometry and bounded decode horizon required")
        if self.buffer is None:
            value = torch.empty(
                (*shape[:2], capacity, shape[-1]), dtype=source.dtype, device=source.device
            )
            self.cache.observe((value,))
            self.buffer = value
            self.cache.shared_tensors = (*self.cache.shared_tensors, value)
            self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
            self.maximum_workspace_bytes = backing_bytes((value,))
            self.cache.observe()
        if (
            self.buffer.device != source.device
            or self.buffer.shape[:2] != shape[:2]
            or self.buffer.shape[-1] != shape[-1]
            or shape[-2] > self.buffer.shape[-2]
        ):
            raise ValueError("one fixed-address common MHA prompt workspace required")
        view = self.buffer[..., : shape[-2], :]
        return view, view

    def reconstruct(self, layer, component, destination):
        from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        if (
            self.closed
            or self.buffer is None
            or component not in (0, 1)
            or not layer.leased
            or destination.data_ptr() != self.buffer.data_ptr()
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("only a private fixed-address prompt reader may be replayed")
        packet = layer.representation
        if isinstance(packet, BoundDeltaFormat):
            # This mutates ordered code-replay state and MUST NOT be captured.
            packet = packet.replay.view(packet.index)
            if layer.representation.replay.prepared_views is None:
                raise ValueError("prompt graphs require stable private replay descriptors")
        signature = (id(packet), id(layer.attention), self.buffer.data_ptr(), self.buffer.stride())
        key = id(layer), component
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 2 * len(self.cache.layers):
                raise RuntimeError("bounded prompt-graph route table exhausted")
            # Compile/warm the exact kernels before CUDA capture. Do not include
            # ordered replay or changing history tensors in the graph closure.
            fill_component(packet, layer.attention, component, self.buffer)
            self.warm_fills += 1
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            current, stream = torch.cuda.current_stream(), torch.cuda.Stream()
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        fill_component(packet, layer.attention, component, self.buffer)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                torch.cuda.synchronize()
                additional = torch.cuda.memory_allocated() - before
                # Refuse opaque private tensor allocations rather than hide
                # them in a cache-memory claim. Graph driver metadata is not
                # certified whole-request KV storage and is reported as such.
                if additional != 0:
                    raise RuntimeError(f"prompt graph introduced {additional} unowned tensor bytes")
            except BaseException:
                graph.reset()
                raise
            entry = signature, graph
            self.entries[key] = entry
            self.captures += 1
        elif entry[0] != signature:
            raise RuntimeError("immutable prompt graph bindings changed during request")
        entry[1].replay()
        self.replays += 1

    def close(self):
        if self.closed:
            return
        if any(x.leased for x in self.cache.layers):
            raise RuntimeError("prompt graphs cannot retire an active lease")
        if self.entries:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            for _, graph in self.entries.values():
                graph.reset()
            if torch.cuda.memory_allocated() != before:
                raise RuntimeError("prompt graph retirement changed unowned tensor storage")
        self.entries.clear()
        value = self.buffer
        self.cache.shared_tensors = tuple(t for t in self.cache.shared_tensors if t is not value)
        self.buffer = None
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.closed = True
        self.cache.observe()

    def record(self):
        return dict(
            enabled=True,
            captures=self.captures,
            replays=self.replays,
            warm_fills=self.warm_fills,
            maximum_workspace_bytes=self.maximum_workspace_bytes,
            graph_private_tensor_bytes=self.checked_private_tensor_bytes,
            active_workspace_bytes=backing_bytes((self.buffer,)) if self.buffer is not None else 0,
            retired=self.closed,
            only_immutable_prompt_captured=True,
            history_attention_and_ordered_replay_outside_graph=True,
            capture_and_retirement_in_request_timing=True,
            driver_graph_metadata_not_KV_ownership_certified=True,
            independent_quality_N=0,
        )


class PromptGraphLayer(SequentialSourceHistoryLayer):
    @property
    def lease_bytes(self):
        # The unique fixed backing belongs to cache.shared_tensors, not each
        # layer's alias. All unused capacity is charged, never just view bytes.
        if getattr(self, "prompt_graph_pool", None) is not None and self.leased:
            return 0
        return getattr(self, "_ordinary_lease_bytes", 0)

    @lease_bytes.setter
    def lease_bytes(self, value):
        self._ordinary_lease_bytes = value

    def _new_lease(self, shape, source):
        return self.prompt_graph_pool.acquire(self, shape, source)


class PromptGraphCache(SourceHistoryCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_graph_pool = PromptGraphPool(self)

    def storage_record(self):
        record = super().storage_record()
        record["prompt_graph_reader"] = self.prompt_graph_pool.record()
        return record


@torch.inference_mode()
def gpu_controls():
    """Real graph replay and exact history/sentinel checks, no model weights."""
    from types import SimpleNamespace

    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.storage_runs import cache_from_formats

    generator = torch.Generator(device="cuda").manual_seed(20261104)
    originals = tuple(
        tuple(
            torch.randn((1, 2, n, 128), device="cuda", generator=generator).bfloat16()
            for _ in range(2)
        )
        for n in (5, 9)
    )
    formats = tuple(
        ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(x) for x in pair))
        for pair in originals
    )
    adapter = SimpleNamespace(
        _language_layers=lambda: [
            SimpleNamespace(self_attn=torch.nn.Linear(256, 256, bias=False, device="cuda"))
            for _ in formats
        ]
    )
    caches = [
        cache_from_formats(
            adapter,
            formats,
            updates=8,
            audit=True,
            source_history=True,
            source_history_constructor="direct_batched_checks",
            sealed=True,
            consolidated_history=True,
            sequential_lease=True,
            sequential_source_history=True,
            prompt_graph_reader=True,
        )
        for _ in range(2)
    ]
    try:
        histories = [[[], []] for _ in formats]
        for step in range(7):
            phase = (
                torch.ones(1, 1, 128, device="cuda", dtype=torch.bfloat16),
                torch.zeros(1, 1, 128, device="cuda", dtype=torch.bfloat16),
            )
            position = torch.tensor([[900 + step]], device="cuda")
            for i in range(len(formats)):
                hidden = torch.randn((1, 1, 256), device="cuda", generator=generator).bfloat16()
                pair = tuple(
                    torch.randn((1, 2, 1, 128), device="cuda", generator=generator).bfloat16()
                    for _ in range(2)
                )
                for h, x in zip(histories[i], pair, strict=True):
                    h.append(x.clone())
                for cache in caches:
                    cache.source_input(i, hidden, phase, position)
                    layer = cache.layers[i]
                    leased = layer.update(*pair)
                    for component in (0, 1):
                        layer.materialize_component(component)
                        expected = torch.cat((originals[i][component], *histories[i][component]), 2)
                        if not torch.equal(leased[0].view(torch.int16), expected.view(torch.int16)):
                            raise RuntimeError(
                                "prompt graph changed a K/V, history or logical slot"
                            )
                    layer.release()
        if (
            caches[0].prompt_graph_pool.buffer.data_ptr()
            == caches[1].prompt_graph_pool.buffer.data_ptr()
        ):
            raise RuntimeError("requests share a graph workspace")
        for cache in caches:
            before = cache.storage_record()
            pool = cache.prompt_graph_pool
            if pool.captures != 4 or pool.replays != 42 or pool.maximum_workspace_bytes != 8192:
                raise RuntimeError("both routes, reused graphs and full backing charge required")
            pool.close()
            after = cache.storage_record()
            if (
                before["final_active_kv_storage_bytes"] - after["final_active_kv_storage_bytes"]
                != pool.maximum_workspace_bytes
            ):
                raise RuntimeError("workspace retirement disagrees with independent inventory")
        return dict(
            bitwise_prompt_and_history=True,
            two_requests_disjoint=True,
            captures_per_request=4,
            replays_per_request=42,
            full_workspace_backing_bytes_per_request=8192,
            retired=True,
            independent_quality_N=0,
            synthetic_control_only=True,
        )
    finally:
        for cache in caches:
            for layer in cache.layers:
                layer.release()
            cache.prompt_graph_pool.close()
