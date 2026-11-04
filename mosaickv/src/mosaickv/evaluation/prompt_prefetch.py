"""One-layer-ahead exact prompt-K prefetch, isolated request execution variant.

Only static prompt K is prefetched; generated history is always produced by the
ordinary path. One charged buffer, one producer stream, explicit stream events.
Shared code replay may advance ONLY after current V/PV finished using its codes.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_state = ContextVar("evicastkv_prompt_prefetch", default=None)
MAXIMUM_ADDED_BYTES = 6 * 2**20


class Prefetch:
    def __init__(self, cache, updates, *, cuda=None):
        self.cuda = torch.cuda if cuda is None else cuda
        self.cache, self.updates = cache, updates
        self.indices = {id(layer): i for i, layer in enumerate(cache.layers)}
        if not cache.layers or len(self.indices) != len(cache.layers):
            raise ValueError("complete unique private layer sequence required")
        elements = max(
            layer.representation.shape[1]
            * layer.representation.shape[2]
            * layer.representation.shape[3]
            for layer in cache.layers
        )
        if elements * 2 > MAXIMUM_ADDED_BYTES:
            raise ValueError("prefetch buffer exceeds the registered 6 MiB allowance")
        device = cache.layers[0].representation.private()[0].device
        self.buffer = torch.empty(elements, device=device, dtype=torch.bfloat16)
        self.stream = self.cuda.Stream(device=device)
        self.pending, self.ready = None, None
        self.hits = self.misses = self.launches = 0
        cache.shared_tensors = (*cache.shared_tensors, self.buffer)
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        # owned_tensors() includes shared() per packet, but not this global pool.
        cache.initial_bytes = backing_bytes(
            (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))
        )
        cache.observe()

    def take(self, layer, destination):
        if self.pending is None:
            self.misses += 1
            return False
        if self.pending != self.indices[id(layer)]:
            raise RuntimeError("prefetch request/layer ordering changed")
        self.cuda.current_stream().wait_event(self.ready)
        target = self.buffer[
            : layer.representation.shape[1] * layer.prompt_length * layer.representation.shape[-1]
        ].view(layer.representation.shape)
        destination[..., : layer.prompt_length, :].copy_(target)
        self.pending = None
        self.hits += 1
        return True

    def schedule(self, layer):
        if self.pending is not None:
            raise RuntimeError("unconsumed prompt prefetch would be overwritten")
        index = self.indices[id(layer)]
        if index == len(self.cache.layers) - 1 and layer.history_length >= self.updates - 1:
            return
        next_index = (index + 1) % len(self.cache.layers)
        following = self.cache.layers[next_index]
        # All earlier source-code reads, including V/PV, precede this event.
        dependency = self.cuda.Event()
        dependency.record(self.cuda.current_stream())
        self.stream.wait_event(dependency)
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        with self.cuda.stream(self.stream):
            target = self.buffer[
                : following.representation.shape[1]
                * following.prompt_length
                * following.representation.shape[-1]
            ].view(following.representation.shape)
            fill_component(following.representation, following.attention, 0, target)
            self.ready = self.cuda.Event()
            self.ready.record(self.stream)
        self.pending = next_index
        self.launches += 1

    def close(self, *, require_consumed=True):
        self.stream.synchronize()
        unconsumed = self.pending is not None
        self.cache.shared_tensors = tuple(
            t for t in self.cache.shared_tensors if t is not self.buffer
        )
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.buffer = None
        self.cache = None
        if unconsumed and require_consumed:
            raise RuntimeError("request ended with an unconsumed prefetch")


@contextmanager
def use(enabled):
    if type(enabled) is not bool or _state.get() is not None:
        raise ValueError("explicit nonnested prefetch request required")
    holder = dict(engine=None, enabled=enabled, hits=0, misses=0, launches=0, buffer_bytes=0)
    token = _state.set(holder)
    failed = False
    try:
        yield holder
    except BaseException:
        failed = True
        raise
    finally:
        try:
            engine = holder["engine"]
            if engine is not None:
                holder.update(
                    hits=engine.hits,
                    misses=engine.misses,
                    launches=engine.launches,
                    buffer_bytes=engine.buffer.numel() * engine.buffer.element_size(),
                )
                engine.close(require_consumed=not failed)
                holder["engine"] = None
        finally:
            _state.reset(token)


def attach(cache, updates):
    holder = _state.get()
    if holder is None or not holder["enabled"]:
        return
    if holder["engine"] is not None or not torch.cuda.is_available():
        raise ValueError("one actual CUDA private cache per prefetch request required")
    holder["engine"] = Prefetch(cache, updates)


def take(layer, destination):
    holder = _state.get()
    return bool(
        holder is not None
        and holder["engine"] is not None
        and holder["engine"].take(layer, destination)
    )


def after_attention(layer):
    holder = _state.get()
    if holder is not None and holder["engine"] is not None:
        holder["engine"].schedule(layer)
