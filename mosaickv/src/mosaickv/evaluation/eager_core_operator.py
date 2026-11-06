"""Bounded same-operator attention study, not integrated into model decoding.

Split QK/softmax and PV because the existing reader overwrites K with V in
one charged lease. No second KV allocation, SDPA, FlashAttention or new math.
"""

import gc
import math
import random
import statistics
import time
from collections import Counter
from types import SimpleNamespace

import torch


def opportunities(prompt_lengths, updates):
    if (
        not prompt_lengths
        or any(type(n) is not int or n < 1 for n in prompt_lengths)
        or type(updates) is not int
        or not 1 <= updates <= 127
    ):
        raise ValueError("positive immutable prompt lengths and bounded decode updates required")
    return Counter(n + step for n in prompt_lengths for step in range(1, updates + 1))


def score_into(query, key, mask, scaling, views):
    torch.matmul(query, key.transpose(2, 3), out=views.scores)
    views.scores.mul_(scaling)
    source = views.scores
    if mask is not None:
        source = views.masked if mask.dtype == torch.float32 else views.scores
        torch.add(views.scores, mask, out=source)
    # The SAME native softmax overload/dtype as functional.softmax, with an
    # explicit output owner. Do not replace BF16 input by a different reducer.
    torch.ops.aten.softmax.int_out(source, -1, torch.float32, out=views.probabilities_float)
    views.probabilities.copy_(views.probabilities_float)
    return views.probabilities


def value_into(weights, value, out):
    torch.matmul(weights, value, out=out)
    return out.transpose(1, 2).contiguous()


def reference_scores(query, key, mask, scaling):
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        weights = weights + mask
    return torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)


class Core:
    """One request-private fixed workspace; graphs contain no per-N outputs."""

    def __init__(
        self,
        lease,
        *,
        mask_dtype,
        graph,
        frequencies,
        minimum_reuses=16,
        audit=False,
        reuse_views=False,
    ):
        if (
            lease.ndim != 4
            or lease.shape[0] != 1
            or lease.shape[-1] != 128
            or lease.dtype != torch.bfloat16
            or not lease.is_contiguous()
            or lease.shape[1] < 1
            or lease.shape[1] > 32
            or not 1 <= lease.shape[-2] <= 1024
            or mask_dtype not in (None, torch.bfloat16, torch.float32)
            or type(graph) is not bool
            or type(audit) is not bool
            or type(reuse_views) is not bool
            or type(minimum_reuses) is not int
            or minimum_reuses != 16
        ):
            raise ValueError("bounded BF16 full-MHA lease and explicit fixed graph policy required")
        if graph and lease.device.type != "cuda":
            raise NotImplementedError("real CUDA graph required; no CPU graph imitation")
        if any(
            type(n) is not int
            or not 1 <= n <= lease.shape[-2]
            or type(count) is not int
            or not 1 <= count <= 32 * 127
            for n, count in frequencies.items()
        ):
            raise ValueError("bounded integer layout opportunities required")
        self.lease, self.mask_dtype = lease, mask_dtype
        self.graph, self.frequencies, self.minimum_reuses = graph, dict(frequencies), minimum_reuses
        self.audit = audit
        self.reuse_views, self.view_cache = reuse_views, {}
        self.view_builds = self.view_reuses = 0
        self.pool = torch.cuda.graph_pool_handle() if graph else None
        self.capture_transient_peak = 0
        self.capacity, self.heads = lease.shape[-2], lease.shape[1]
        self.q = torch.empty((1, self.heads, 1, 128), device=lease.device, dtype=lease.dtype)
        size = self.heads * self.capacity
        self.scores = torch.empty(size, device=lease.device, dtype=lease.dtype)
        self.masked = (
            torch.empty(size, device=lease.device, dtype=torch.float32)
            if mask_dtype == torch.float32
            else None
        )
        self.probabilities_float = torch.empty(size, device=lease.device, dtype=torch.float32)
        self.probabilities = torch.empty(size, device=lease.device, dtype=lease.dtype)
        self.out = torch.empty_like(self.q)
        self.mask = (
            None
            if mask_dtype is None
            else torch.empty(self.capacity, device=lease.device, dtype=mask_dtype)
        )
        self.entries, self.pending, self.closed = {}, None, False
        self.captures = self.replays = self.warm_calls = self.direct_calls = 0

    def tensors(self):
        return tuple(
            x
            for x in (
                self.q,
                self.scores,
                self.masked,
                self.probabilities_float,
                self.probabilities,
                self.out,
                self.mask,
            )
            if x is not None
        )

    def views(self, length):
        if self.reuse_views and length in self.view_cache:
            self.view_reuses += 1
            return self.view_cache[length]
        shape = (1, self.heads, 1, length)
        count = self.heads * length
        views = SimpleNamespace(
            scores=self.scores[:count].view(shape),
            masked=None if self.masked is None else self.masked[:count].view(shape),
            probabilities_float=self.probabilities_float[:count].view(shape),
            probabilities=self.probabilities[:count].view(shape),
            mask=None if self.mask is None else self.mask[:length].view(1, 1, 1, length),
            lease=self.lease[..., :length, :],
        )
        self.view_builds += 1
        if self.reuse_views:
            self.view_cache[length] = views
        return views

    def capture(self, function):
        from mosaickv.evaluation.post_attention_segment import capture_stream

        current = torch.cuda.current_stream(self.lease.device)
        # Reuse the EXISTING common epilogue capture stream, not a fresh
        # per-layout cuBLAS stream/workspace. No request tensors live there.
        stream = capture_stream(self.lease.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            function()
            function()
        self.warm_calls += 2
        graph = torch.cuda.CUDAGraph()
        if self.audit:
            current.wait_stream(stream)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        # Match the existing private reader/epilogue's low-level capture.
        # torch.cuda.graph.__enter__ performs a global GC/empty_cache and
        # synchronize; do not repeat that unrelated whole-process work per N.
        with torch.cuda.stream(stream):
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                function()
            finally:
                graph.capture_end()
        current.wait_stream(stream)
        if self.audit:
            torch.cuda.synchronize()
            self.capture_transient_peak = max(
                self.capture_transient_peak, torch.cuda.max_memory_allocated() - before
            )
        self.captures += 1
        return graph

    def scores_for(self, query, mask, length):
        if (
            self.closed
            or self.pending is not None
            or type(length) is not int
            or not 1 <= length <= self.capacity
            or query.shape != self.q.shape
            or query.dtype != self.q.dtype
            or query.device != self.q.device
            or (mask is None) != (self.mask_dtype is None)
        ):
            raise ValueError(
                "one isolated same-device query and complete lease transaction required"
            )
        if mask is not None and (
            mask.dtype != self.mask_dtype
            or mask.shape != (1, 1, 1, length)
            or mask.device != query.device
        ):
            raise ValueError("original mask dtype, geometry and positions required")
        self.q.copy_(query)
        views = self.views(length)
        if mask is not None:
            views.mask.copy_(mask)
        use_graph = self.graph and self.frequencies.get(length, 0) >= self.minimum_reuses
        if use_graph and length not in self.entries:
            qk = self.capture(lambda: score_into(self.q, views.lease, views.mask, 128**-0.5, views))
            pv = self.capture(lambda: value_into(views.probabilities, views.lease, self.out))
            self.entries[length] = qk, pv
        if use_graph:
            self.entries[length][0].replay()
            self.replays += 1
        else:
            score_into(self.q, views.lease, views.mask, 128**-0.5, views)
            self.direct_calls += 1
        self.pending = length, use_graph
        return views.probabilities

    def values_for(self, length):
        if self.closed or self.pending is None or self.pending[0] != length:
            raise RuntimeError("V must follow exactly one matching QK stage on the same lease")
        _, use_graph = self.pending
        views = self.views(length)
        if use_graph:
            self.entries[length][1].replay()
            self.replays += 1
        else:
            value_into(views.probabilities, views.lease, self.out)
            self.direct_calls += 1
        self.pending = None
        return self.out.transpose(1, 2).contiguous()

    def close(self):
        if self.closed:
            return
        if self.pending is not None:
            raise RuntimeError("cannot retire an incomplete K-to-V transaction")
        if self.lease.device.type == "cuda":
            torch.cuda.synchronize()  # Replays must complete BEFORE reset frees graph owners.
        for pair in self.entries.values():
            for graph in pair:
                graph.reset()
        self.entries.clear()
        self.view_cache.clear()
        self.closed = True
        if self.lease.device.type == "cuda":
            torch.cuda.synchronize()
        self.q = self.scores = self.masked = None
        self.probabilities_float = self.probabilities = self.out = self.mask = None
        self.lease = None


def same_words(x, y):
    return x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16))


def warm_backend():
    """Expose common cuBLAS initialization separately, like the existing epilogue.

    These are library allocations, not a hidden per-layout graph workspace.
    Actual model integration would have to reconcile this boundary too.
    """
    from mosaickv.evaluation.post_attention_segment import capture_stream

    before = torch.cuda.memory_allocated()
    current = torch.cuda.current_stream()
    stream = capture_stream(torch.device("cuda", torch.cuda.current_device()))
    for active_stream in (current, stream):
        active_stream.wait_stream(current)
        with torch.cuda.stream(active_stream):
            lease = torch.zeros(1, 32, 731, 128, device="cuda", dtype=torch.bfloat16)
            for dtype in (None, torch.bfloat16, torch.float32):
                core = Core(lease, mask_dtype=dtype, graph=False, frequencies={})
                query = torch.zeros_like(core.q)
                mask = (
                    None if dtype is None else torch.zeros(1, 1, 1, 731, device="cuda", dtype=dtype)
                )
                weights = reference_scores(query, lease, mask, 128**-0.5)
                output = torch.matmul(weights, lease)
                core.scores_for(query, mask, 731)
                core.values_for(731)
                del weights, output, query, mask
                core.close()
            del core, lease
        current.wait_stream(active_stream)
    gc.collect()
    torch.cuda.synchronize()
    return max(0, torch.cuda.memory_allocated() - before)


@torch.inference_mode()
def controls(*, lengths=(601, 605, 665, 669, 727, 731, 601), capacity=731, reuse_views=False):
    if (
        type(capacity) is not int
        or not 1 <= capacity <= 1024
        or not lengths
        or any(type(n) is not int or not 1 <= n <= capacity for n in lengths)
    ):
        raise ValueError("bounded actual control lengths and physical lease capacity required")
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.prompt_graph_reader import (
        allocator_blocks as active_allocations,
    )
    from mosaickv.evaluation.prompt_graph_reader import generator_metadata

    backend_bytes = warm_backend()
    generator = torch.Generator(device="cuda").manual_seed(20261106)
    records = []
    for mask_dtype in (None, torch.bfloat16, torch.float32):
        lease = torch.empty(1, 32, capacity, 128, device="cuda", dtype=torch.bfloat16)
        for graph in (False, True):
            core = Core(
                lease,
                mask_dtype=mask_dtype,
                graph=graph,
                frequencies=dict.fromkeys(lengths, 32),
                audit=True,
                reuse_views=reuse_views,
            )
            old = active_allocations()
            checks = []
            for index, length in enumerate(lengths):
                q = torch.randn(core.q.shape, device="cuda", generator=generator).bfloat16()
                k = torch.randn(1, 32, length, 128, device="cuda", generator=generator).bfloat16()
                v = torch.randn(k.shape, device="cuda", generator=generator).bfloat16()
                mask = (
                    None
                    if mask_dtype is None
                    else torch.zeros(1, 1, 1, length, device="cuda", dtype=mask_dtype)
                )
                if mask is not None:
                    mask[..., ::17] = torch.finfo(mask_dtype).min
                expected_weights = reference_scores(q, k, mask, 128**-0.5)
                expected_output = torch.matmul(expected_weights, v).transpose(1, 2).contiguous()
                lease.fill_(-71)
                lease[..., :length, :].copy_(k)
                got_weights = core.scores_for(q, mask, length)
                if not same_words(got_weights, expected_weights) or not same_words(
                    lease[..., :length, :], k
                ):
                    raise RuntimeError(
                        f"QK/softmax words differ: graph={graph}, mask={mask_dtype}, "
                        f"length={length}, input_index={index}"
                    )
                lease[..., :length, :].copy_(v)
                got_output = core.values_for(length)
                if not same_words(got_output, expected_output) or not same_words(
                    lease[..., :length, :], v
                ):
                    raise RuntimeError(
                        f"PV words differ: graph={graph}, mask={mask_dtype}, "
                        f"length={length}, input_index={index}"
                    )
                if not bool((lease[..., length:, :] == -71).all()):
                    raise RuntimeError("core touched unused physical capacity")
                checks.append(dict(length=length, changed_inputs=index, all_words_identical=True))
                del q, k, v, mask, expected_weights, expected_output, got_weights, got_output
            gc.collect()
            torch.cuda.synchronize()
            after = active_allocations()
            extra = {p: s for p, s in after.items() if p not in old}
            private_bytes = sum(s[1] for s in extra.values())
            if private_bytes:
                generator_metadata(old, after, private_bytes)
            workspace = backing_bytes(core.tensors())
            if workspace + core.capture_transient_peak + private_bytes > 384 * 1024:
                raise RuntimeError("fixed non-KV workspace exceeds registered bound")
            pool_reserved = (
                sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if tuple(segment.get("segment_pool_id", (0, 0))) == core.pool
                )
                if core.pool is not None
                else 0
            )
            counts = dict(captures=core.captures, replays=core.replays, warm_calls=core.warm_calls)
            core.close()
            gc.collect()
            torch.cuda.synchronize()
            if any(p in active_allocations() for p in extra):
                raise RuntimeError("opaque private graph owners did not retire")
            records.append(
                dict(
                    mask_dtype=str(mask_dtype),
                    graph=graph,
                    checks=checks,
                    workspace_bytes=workspace,
                    captured_transient_peak_bytes=core.capture_transient_peak,
                    graph_pool_reserved_bytes_NOT_active_tensor_storage=pool_reserved,
                    generator_private_bytes=private_bytes,
                    retired=True,
                    reuse_views=reuse_views,
                    view_descriptor_builds=core.view_builds,
                    view_descriptor_reuses=core.view_reuses,
                    **counts,
                )
            )
    return dict(cases=records, common_backend_warm_allocation_bytes=backend_bytes)


@torch.inference_mode()
def study():
    checked = controls()  # A word/owner failure prevents any timing claim.
    generator = torch.Generator(device="cuda").manual_seed(20261106)
    rng = random.Random(20261106)
    records = []
    lease = torch.empty(1, 32, 731, 128, device="cuda", dtype=torch.bfloat16)
    for pattern in ("uniform", "adaptive"):
        lengths = (665,) * 32 if pattern == "uniform" else tuple(417 + 8 * i for i in range(32))
        frequencies = opportunities(tuple(n - 64 for n in lengths), 127)
        q = torch.randn(32, 1, 32, 1, 128, device="cuda", generator=generator).bfloat16()
        k = torch.randn(32, 1, 32, 731, 128, device="cuda", generator=generator).bfloat16()
        v = torch.randn(k.shape, device="cuda", generator=generator).bfloat16()
        masks = [torch.zeros(1, 1, 1, n, device="cuda", dtype=torch.bfloat16) for n in lengths]
        trials = []
        for trial in range(-5, 20):
            order = ["original", "out", "graph"]
            rng.shuffle(order)
            for position, policy in enumerate(order):
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                before = time.perf_counter()
                start.record()
                core = (
                    None
                    if policy == "original"
                    else Core(
                        lease,
                        mask_dtype=torch.bfloat16,
                        graph=policy == "graph",
                        frequencies=frequencies,
                    )
                )
                for i, length in enumerate(lengths):
                    lease[..., :length, :].copy_(k[i, ..., :length, :])
                    weights = (
                        reference_scores(q[i], lease[..., :length, :], masks[i], 128**-0.5)
                        if core is None
                        else core.scores_for(q[i], masks[i], length)
                    )
                    lease[..., :length, :].copy_(v[i, ..., :length, :])
                    output = (
                        torch.matmul(weights, lease[..., :length, :]).transpose(1, 2).contiguous()
                        if core is None
                        else core.values_for(length)
                    )
                captures = 0 if core is None else core.captures
                del weights, output
                if core is not None:
                    core.close()
                end.record()
                torch.cuda.synchronize()
                trials.append(
                    dict(
                        trial=trial,
                        warmup=trial < 0,
                        method=policy,
                        order=position,
                        wall_seconds=time.perf_counter() - before,
                        gpu_seconds=start.elapsed_time(end) / 1000,
                        captures=captures,
                        torch_peak_increment_bytes=torch.cuda.max_memory_allocated() - baseline,
                        cold_capture_and_retirement_included=True,
                    )
                )
                del core
        means = {
            name: statistics.mean(
                t["wall_seconds"] for t in trials if t["method"] == name and not t["warmup"]
            )
            for name in ("original", "out", "graph")
        }
        if any(not math.isfinite(v) or v <= 0 for v in means.values()):
            raise RuntimeError("finite positive synchronized operator times required")
        records.append(dict(pattern=pattern, means=means, trials=trials))
        del q, k, v, masks
    return dict(
        complete=True,
        failures=[],
        controls=checked,
        records=records,
        operator_nonregression_all_shapes=all(
            min(r["means"]["out"], r["means"]["graph"]) <= 0.96 * r["means"]["original"]
            for r in records
        ),
        no_model_latency_or_quality_acceptance=True,
        no_runtime_integration=True,
        independent_quality_N=0,
    )
