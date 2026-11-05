"""Default-off next-layer K replay into the SAME idle prompt-graph buffer.

PV must already be enqueued before the producer waits on its event. The main
stream does not use that buffer for output projection/MLP. The next layer waits
before touching it. Never prefetch across a token boundary/history seal.
No tensor buffer, model value, selection, precision, or cache position changes.
CUDA stream/event resources are not certified whole-request GPU-peak evidence.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_enabled = ContextVar("evicastkv_graph_lookahead", default=None)


@contextmanager
def use(enabled):
    if type(enabled) is not bool or _enabled.get() is not None:
        raise ValueError("explicit nonnested graph lookahead policy required")
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


class Lookahead:
    def __init__(self, pool, *, cuda=None):
        self.pool = pool
        self.cuda = torch.cuda if cuda is None else cuda
        self.layers = tuple(pool.cache.layers)
        self.indices = {id(layer): i for i, layer in enumerate(self.layers)}
        if not self.layers or len(self.indices) != len(self.layers):
            raise ValueError("complete unique request-private layers required")
        self.stream = self.dependency = self.ready = None
        self.pending = self.submitting = self.last_scheduled = None
        self.launches = self.hits = self.cold_misses = self.token_boundaries = 0
        self.closed = False

    def authorized(self, layer, epoch, destination):
        return (
            not self.closed
            and self.submitting == (id(layer), epoch)
            and type(epoch) is int
            and epoch == layer.history_length + 1
            and not layer.leased
            and self.pool.buffer is destination
        )

    def check_acquire(self, layer):
        if self.closed or (
            self.pending is not None and self.pending != (id(layer), layer.history_length)
        ):
            raise RuntimeError("lookahead destination/epoch does not match next exclusive lease")

    def schedule(self, current):
        if self.closed or self.pending is not None or self.submitting is not None:
            raise RuntimeError("private lookahead cannot overwrite an unconsumed transaction")
        index = self.indices.get(id(current))
        if index is None or not current.leased or current.current_component != 1:
            raise RuntimeError("only completed current V/PV may schedule the next prompt K")
        if index == len(self.layers) - 1:
            self.token_boundaries += 1
            return  # Sealing/verification may reuse the buffer before next token.
        following = self.layers[index + 1]
        if following.leased or following.history_length + 1 != current.history_length:
            raise RuntimeError("next released layer must belong to this same decode update")
        if (id(following), 0) not in self.pool.entries:
            self.cold_misses += 1
            return  # Do not add captures/warmups or tensor owners to this path.
        transaction = id(current), current.history_length
        if self.last_scheduled == transaction:
            raise RuntimeError("a layer/update may schedule only one lookahead")
        if self.stream is None:
            self.stream = self.cuda.Stream(device=self.pool.buffer.device)
            self.dependency, self.ready = self.cuda.Event(), self.cuda.Event()
        self.dependency.record(self.cuda.current_stream())
        self.stream.wait_event(self.dependency)
        expected = following.history_length + 1
        self.submitting = id(following), expected
        try:
            with self.cuda.stream(self.stream):
                self.pool.reconstruct(following, 0, self.pool.buffer, _lookahead_epoch=expected)
                self.ready.record(self.stream)
        except BaseException:
            self.stream.synchronize()
            raise
        finally:
            self.submitting = None
        self.pending = id(following), expected
        self.last_scheduled = transaction
        self.launches += 1

    def take(self, layer, destination):
        if self.pending is None:
            return False
        self.check_acquire(layer)
        if (
            not layer.leased
            or destination.data_ptr() != self.pool.buffer.data_ptr()
            or destination.stride() != self.pool.buffer.stride()
        ):
            raise RuntimeError("lookahead may only be consumed from its identical fixed backing")
        self.cuda.current_stream().wait_event(self.ready)
        self.pending = None
        self.hits += 1
        return True  # Intentionally no copy; this IS the ordinary K destination.

    def close(self):
        pending = self.pending is not None or self.submitting is not None
        if self.stream is not None:
            self.stream.synchronize()
        self.closed = True
        self.pending = self.submitting = None
        if pending:
            raise RuntimeError("lookahead ended with an unconsumed producer transaction")

    def record(self):
        return dict(
            enabled=True,
            launches=self.launches,
            hits=self.hits,
            first_capture_cold_misses=self.cold_misses,
            skipped_token_boundaries=self.token_boundaries,
            pending=self.pending is not None,
            retired=self.closed,
            extra_tensor_workspace_bytes=0,
            second_prompt_buffer=False,
            prompt_copies=0,
            streams_created=int(self.stream is not None),
            reusable_events_created=2 * int(self.stream is not None),
            whole_request_GPU_peak_certified=False,
        )


def attach(cache):
    if not _enabled.get():
        return
    pool = getattr(cache, "prompt_graph_pool", None)
    if pool is None or pool.lookahead is not None:
        raise ValueError("one private graph pool required for lookahead")
    pool.lookahead = Lookahead(pool)


def after_attention(layer):
    pool = getattr(layer, "prompt_graph_pool", None)
    if pool is not None and pool.lookahead is not None:
        pool.lookahead.schedule(layer)


@torch.inference_mode()
def gpu_controls():
    """A-B-A real stream/graph controls before checkpoint loading, no downloads."""
    from types import SimpleNamespace

    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.graph_accounting import use as accounting
    from mosaickv.evaluation.storage_runs import cache_from_formats

    rng = torch.Generator(device="cuda").manual_seed(20261105)
    originals = tuple(
        tuple(torch.randn(1, 2, n, 128, device="cuda", generator=rng).bfloat16() for _ in (0, 1))
        for n in (5, 7, 9)
    )
    formats = tuple(
        ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
        for pair in originals
    )
    inputs = [
        [
            (
                torch.randn(1, 1, 256, device="cuda", generator=rng).bfloat16(),
                tuple(
                    torch.randn(1, 2, 1, 128, device="cuda", generator=rng).bfloat16()
                    for _ in (0, 1)
                ),
                torch.randn(1, 2, 1, 128, device="cuda", generator=rng).bfloat16(),
            )
            for _ in formats
        ]
        for _ in range(7)
    ]
    adapter = SimpleNamespace(
        _language_layers=lambda: [
            SimpleNamespace(self_attn=torch.nn.Linear(256, 256, bias=False, device="cuda"))
            for _ in formats
        ]
    )
    observations, controls = [], []
    for enabled in (False, True, False):
        with use(enabled), accounting("event_sum"):
            cache = cache_from_formats(
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
        pool = cache.prompt_graph_pool
        histories = [[[], []] for _ in formats]
        outputs = []
        try:
            for step, rows in enumerate(inputs):
                phase = (
                    torch.ones(1, 1, 128, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(1, 1, 128, device="cuda", dtype=torch.bfloat16),
                )
                position = torch.tensor([[900 + step]], device="cuda")
                for i, (hidden, pair, query) in enumerate(rows):
                    for history, tensor in zip(histories[i], pair, strict=True):
                        history.append(tensor)
                    cache.source_input(i, hidden, phase, position)
                    layer = cache.layers[i]
                    leased = layer.update(*pair)
                    expected_K = torch.cat((originals[i][0], *histories[i][0]), 2)
                    if not torch.equal(leased[0].view(torch.int16), expected_K.view(torch.int16)):
                        raise RuntimeError("lookahead changed original K or logical/history slots")
                    weights = torch.matmul(query, leased[0].transpose(-1, -2))
                    weights = weights.softmax(-1, dtype=torch.float32).to(torch.bfloat16)
                    layer.materialize_component(1)
                    expected_V = torch.cat((originals[i][1], *histories[i][1]), 2)
                    if not torch.equal(leased[0].view(torch.int16), expected_V.view(torch.int16)):
                        raise RuntimeError("lookahead changed original V or logical/history slots")
                    outputs.append(torch.matmul(weights, leased[0]).clone())
                    after_attention(layer)  # Must follow PV, precede release/next MLP.
                    layer.release()
            if pool.captures != 6 or pool.replays != 42:
                raise RuntimeError("lookahead added or omitted a prompt graph replay")
            pool.close()
            record = cache.storage_record()
            if enabled and (
                pool.lookahead.launches != 12
                or pool.lookahead.hits != 12
                or pool.lookahead.cold_misses != 2
                or pool.lookahead.token_boundaries != 7
            ):
                raise RuntimeError("exact warm/cold/epoch lookahead engagement required")
            observations.append(torch.stack(outputs))
            controls.append(
                {
                    k: record[k]
                    for k in (
                        "initial_active_kv_storage_bytes",
                        "final_active_kv_storage_bytes",
                        "peak_active_kv_storage_bytes",
                    )
                }
            )
        finally:
            for layer in cache.layers:
                layer.release()
            pool.close()
    if (
        controls[0] != controls[1]
        or controls[0] != controls[2]
        or any(
            not torch.equal(observations[0].view(torch.int16), x.view(torch.int16))
            for x in observations[1:]
        )
    ):
        raise RuntimeError("A-B-A lookahead changed an output word or storage boundary")
    return dict(
        status="completed",
        A_B_A_isolation=True,
        original_KV_words=True,
        output_words_equal=True,
        storage_boundaries_equal=True,
        synthetic_GPU_control_only=True,
        independent_quality_N=0,
        no_second_prompt_buffer=True,
        warm_lookahead_replays=12,
    )
