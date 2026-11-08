"""Private prompt-reader graphs; no attention or model layers are captured.

The immutable prompt reader alone writes to one charged request-wide lease.
Generated history and native eager attention stay outside the graph. Ordered
code replay is captured only by a separately configured private policy.
The explicit joint variant also captures incoming frozen-weight projections;
it must not be labeled immutable-prompt-only.
The default runtime is unchanged. This is an execution
diagnostic for already consumed inputs, not independent quality evidence.
"""

from contextlib import contextmanager
from dataclasses import asdict

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.prompt_graph_policy import PromptGraphPolicy
from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
from mosaickv.evaluation.source_history import SourceHistoryCache


def allocator_blocks():
    return {
        b["address"]: (b["requested_size"], b["size"])
        for s in torch.cuda.memory_snapshot()
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    }


def generator_metadata(before, after, additional):
    """The pinned generator registers two int64 scalars, 512-byte blocks each.

    Never bless arbitrary allocations of the same total size. Existing active
    owners must survive and the only new owners must be exactly these scalars.
    This was verified separately without checkpoint weights before this retry.
    """
    new = {p: size for p, size in after.items() if p not in before}
    if (
        not before.keys() <= after.keys()
        or len(new) != 2
        or any(v != (8, 512) for v in new.values())
        or additional != 1024
    ):
        lost = {p: size for p, size in before.items() if p not in after}
        raise RuntimeError(
            "only the pinned two-scalar CUDA generator metadata is admissible: "
            f"new={new!r}; retired={lost!r}; allocated_delta={additional}; "
            f"new_requested_bytes={sum(v[0] for v in new.values())}; "
            f"new_allocator_bytes={sum(v[1] for v in new.values())}"
        )
    return tuple(new)


class PromptGraphPool:
    _active_owner = None

    def __init__(self, cache, policy=None):
        self.cache, self.buffer, self.entries = cache, None, {}
        self.policy = PromptGraphPolicy.parse(policy)
        self.row4_workspace = None
        self.row4_workspace_bytes = 0
        self.history_borrowed = False
        self.history_borrows = self.history_borrow_misses = self.history_borrow_maximum_bytes = 0
        self.captured_code_replays = self.row4_replays = 0
        self.captured_code_kernel_dispatches = 0
        self.key_epochs = {}
        self.closed = False
        self.captures = self.replays = self.warm_fills = 0
        self.maximum_workspace_bytes = 0
        self.checked_private_tensor_bytes = 0
        self.private_live_bytes = 0
        self.generator_addresses = ()
        self.lt_engine = None
        self.lt_engine_info = None
        self.lt_workspace_bytes = self.lt_replays = self.lt_matmul_dispatches = 0
        self.lookahead = None
        self.joint_projection = None
        self.joint_projection_policy = "none"
        from mosaickv.evaluation.joint_incoming_projection import reuse_graph_plan

        self.prepared_replay = reuse_graph_plan()
        self.entry_metadata = {}
        self.prepared_replay_hits = self.prepared_consume_bypasses = 0

    def acquire(self, layer, shape, source):
        if (
            self.closed
            or self.history_borrowed
            or any(x.leased for x in self.cache.layers)
            or PromptGraphPool._active_owner not in (None, self)
        ):
            raise RuntimeError("private nonoverlapping prompt-graph lease required")
        if source.device.type != "cuda" or source.dtype != torch.bfloat16:
            raise NotImplementedError("prompt reader replay requires CUDA BF16")
        if self.lookahead is not None:
            self.lookahead.check_acquire(layer)
        PromptGraphPool._active_owner = self
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

    def reconstruct(
        self, layer, component, destination, *, _lookahead_epoch=None, _projection_epoch=None
    ):
        from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        producing = _lookahead_epoch is not None or _projection_epoch is not None
        if _lookahead_epoch is not None and _projection_epoch is not None:
            raise RuntimeError("joint projection and lookahead may not share a producer")
        producer = self.joint_projection if _projection_epoch is not None else self.lookahead
        epoch = _projection_epoch if _projection_epoch is not None else _lookahead_epoch
        if producing and (
            component != 0
            or producer is None
            or not producer.authorized(layer, epoch, destination)
            or (id(layer), 0) not in self.entries
            or PromptGraphPool._active_owner not in (None, self)
        ):
            raise RuntimeError("explicit existing-graph next-layer producer required")
        if (
            self.closed
            or self.history_borrowed
            or self.buffer is None
            or component not in (0, 1)
            or (not producing and not layer.leased)
            or destination.data_ptr() != self.buffer.data_ptr()
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("only a private fixed-address prompt reader may be replayed")
        if (
            not producing
            and component == 0
            and self.lookahead is not None
            and self.lookahead.take(layer, destination)
        ):
            return
        if (
            not producing
            and component == 0
            and self.joint_projection is not None
            and self.joint_projection.take(layer, destination)
        ):
            return
        packet = layer.representation
        replay = None
        if isinstance(packet, BoundDeltaFormat):
            bound = packet
            replay = bound.replay if self.policy.captured_code_replay else None
            if replay is not None:
                if component == 0:
                    key_epoch = epoch if producing else layer.history_length
                    if self.key_epochs.get(id(layer)) == key_epoch:
                        raise RuntimeError("captured XOR replay may run only once per layer/token")
                    self.key_epochs[id(layer)] = key_epoch
                packet = replay.advance_for_graph(bound.index)
            else:
                packet = bound.replay.view(bound.index)
            if bound.replay.prepared_views is None:
                raise ValueError("prompt graphs require stable private replay descriptors")
        if self.prepared_replay:
            signature = (
                id(packet),
                id(layer.attention),
                self.buffer.data_ptr(),
                self.buffer.stride(),
            )
            key = id(layer), component
            entry = self.entries.get(key)
            if entry is not None:
                if entry[0] != signature or key not in self.entry_metadata:
                    raise RuntimeError("immutable prepared graph bindings or metadata changed")
                native_rows, row4 = self.entry_metadata[key]
                entry[1].replay()
                self.replays += 1
                self.prepared_replay_hits += 1
                self.captured_code_replays += int(replay is not None and component == 0)
                if replay is not None and component == 0:
                    self.captured_code_kernel_dispatches += len(replay.graph_bindings[bound.index])
                self.row4_replays += int(row4 and component == 0)
                if native_rows:
                    self.lt_replays += 1
                    rows = getattr(self.lt_engine, "chunk_rows", 128)
                    self.lt_matmul_dispatches += (native_rows + rows - 1) // rows
                return
        from mosaickv.evaluation.lt_source_format import source_format as lt_format

        native = lt_format(packet, component)
        if native is not None:
            if self.policy.row_major_codes:
                raise ValueError("native Lt source and row-major code ablation cannot be mixed")
            if self.lt_engine is None:
                from mosaickv.evaluation.joint_incoming_projection import engine_factory, owned

                self.lt_engine = engine_factory(native.base)
                owners = owned(self.lt_engine)
                self.cache.observe(owners)
                self.cache.shared_tensors = (*self.cache.shared_tensors, *owners)
                self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
                self.lt_workspace_bytes = backing_bytes(owners)
                self.cache.observe()
        body = self._row4_body(packet) if self.policy.row_major_codes else None
        if body is not None and self.row4_workspace is None:
            from mosaickv.evaluation.row_major_source_codes import geometry

            value = torch.empty(
                geometry(*body.shape[2:]), device=body.codes.device, dtype=torch.uint8
            )
            self.cache.observe((value,))
            self.row4_workspace = value
            self.row4_workspace_bytes = backing_bytes((value,))
            self.cache.shared_tensors = (*self.cache.shared_tensors, value)
            self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
            self.cache.observe()

        def fill(*, decode):
            from contextlib import nullcontext

            from mosaickv.evaluation.row_major_source_codes import pack_into, use

            if decode and replay is not None and component == 0:
                replay.decode_for_graph(bound.index)
            if body is not None and component == 0:
                pack_into(body, self.row4_workspace)
            from mosaickv.evaluation.lt_source_format import use as use_lt

            with (
                use(body, self.row4_workspace) if body is not None else nullcontext(),
                use_lt(self.lt_engine) if native is not None else nullcontext(),
            ):
                fill_component(packet, layer.attention, component, self.buffer)

        signature = (id(packet), id(layer.attention), self.buffer.data_ptr(), self.buffer.stride())
        key = id(layer), component
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 2 * len(self.cache.layers):
                raise RuntimeError("bounded prompt-graph route table exhausted")
            # Compile/warm the exact kernels before CUDA capture. Do not include
            # ordered replay or changing history tensors in the graph closure.
            # Warming may read predecessor codes into UNUSED output. It must
            # not run an in-place XOR decoder twice. Capture records decode
            # but executes nothing; the first replay then decodes exactly once.
            fill(decode=False)
            self.warm_fills += 1
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            blocks = allocator_blocks() if not self.entries else None
            graph = torch.cuda.CUDAGraph()
            current, stream = torch.cuda.current_stream(), torch.cuda.Stream()
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        fill(decode=True)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                torch.cuda.synchronize()
                additional = torch.cuda.memory_allocated() - before
                if not self.entries:
                    self.generator_addresses = generator_metadata(
                        blocks, allocator_blocks(), additional
                    )
                    self.private_live_bytes = self.checked_private_tensor_bytes = additional
                    self.cache.observe()
                elif additional != 0:
                    raise RuntimeError(f"prompt graph introduced {additional} unowned tensor bytes")
            except BaseException:
                graph.reset()
                raise
            entry = signature, graph
            self.entries[key] = entry
            if self.prepared_replay:
                self.entry_metadata[key] = (
                    0 if native is None else native.shape[2],
                    body is not None,
                )
            self.captures += 1
        elif entry[0] != signature:
            raise RuntimeError("immutable prompt graph bindings changed during request")
        entry[1].replay()
        self.replays += 1
        self.captured_code_replays += int(replay is not None and component == 0)
        if replay is not None and component == 0:
            self.captured_code_kernel_dispatches += len(replay.graph_bindings[bound.index])
        self.row4_replays += int(body is not None and component == 0)
        if native is not None:
            self.lt_replays += 1
            rows = getattr(self.lt_engine, "chunk_rows", 128)
            self.lt_matmul_dispatches += (native.shape[2] + rows - 1) // rows

    @staticmethod
    def _row4_body(packet):
        from mosaickv.evaluation.head_local_precision import HeadLocalFormat
        from mosaickv.evaluation.hybrid_reader import HybridFormat
        from mosaickv.evaluation.shared_source_format import SharedSourceFormat
        from mosaickv.evaluation.text_source_storage import TextSourceFormat

        while isinstance(packet, (HeadLocalFormat, TextSourceFormat, HybridFormat)):
            packet = packet.base
        if isinstance(packet, SharedSourceFormat) and packet.projection_backend == "fused":
            return packet.source
        return None

    @contextmanager
    def borrow_history(self, shape):
        if not self.policy.history_arena:
            yield None
            return
        if (
            self.closed
            or self.history_borrowed
            or any(x.leased for x in self.cache.layers)
            or (self.lookahead is not None and self.lookahead.pending is not None)
            or (
                self.joint_projection is not None
                and (
                    self.joint_projection.ledger.pending is not None
                    or getattr(
                        self.joint_projection.ledger,
                        "value_commits",
                        self.joint_projection.ledger.calls,
                    )
                    != self.joint_projection.ledger.calls
                )
            )
        ):
            raise RuntimeError("history scratch requires a released exclusive graph workspace")
        from mosaickv.evaluation.history_verification_workspace import partition

        scratch = None if self.buffer is None else partition(self.buffer, shape)
        if scratch is None:
            self.history_borrow_misses += 1
            yield None
            return
        self.history_borrowed = True
        self.history_borrows += 1
        self.history_borrow_maximum_bytes = max(
            self.history_borrow_maximum_bytes,
            sum(t.numel() * t.element_size() for pair in scratch for t in pair),
        )
        try:
            yield scratch
        finally:
            self.history_borrowed = False

    def close(self):
        if self.closed:
            return
        if self.history_borrowed or any(x.leased for x in self.cache.layers):
            raise RuntimeError("prompt graphs cannot retire an active lease")
        if self.joint_projection is not None and self.joint_projection.ledger.pending is not None:
            raise RuntimeError("prompt graphs cannot retire an unconsumed joint key")
        if self.joint_projection is not None and any(
            getattr(layer, "_joint_pending_value", None) is not None for layer in self.cache.layers
        ):
            raise RuntimeError("prompt graphs cannot retire an uncommitted joint value")
        if self.lookahead is not None:
            self.lookahead.close()
        from mosaickv.evaluation.native_history_reader import current as native_state

        native = native_state()
        if native is not None and native.graphs is not None:
            native.retire(self)
        if self.entries:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            for _, graph in self.entries.values():
                graph.reset()
            self.entries.clear()
            self.entry_metadata.clear()
            # reset() destroys kernels but the C++ destructor unregisters RNG
            # state. Do not retain the final loop-local graph during this check.
            del graph
            torch.cuda.synchronize()
            if before - torch.cuda.memory_allocated() != self.private_live_bytes:
                raise RuntimeError("graph metadata retirement differs from its charged owners")
            if any(p in allocator_blocks() for p in self.generator_addresses):
                raise RuntimeError("retired graph still owns a generator metadata allocation")
            self.private_live_bytes = 0
        lt_owners = ()
        if self.lt_engine is not None:
            self.lt_engine_info = self.lt_engine.record()
            from mosaickv.evaluation.joint_incoming_projection import owned

            lt_owners = owned(self.lt_engine)
            self.lt_engine.close()
        owners = (self.buffer, self.row4_workspace, *lt_owners)
        self.cache.shared_tensors = tuple(
            t for t in self.cache.shared_tensors if all(t is not x for x in owners)
        )
        self.buffer = self.row4_workspace = None
        self.lt_engine = None
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.closed = True
        if PromptGraphPool._active_owner is self:
            PromptGraphPool._active_owner = None
        self.cache.observe()

    def record(self):
        from mosaickv.evaluation.joint_graph_scope import graph_scope
        from mosaickv.evaluation.joint_incoming_projection import owned
        from mosaickv.evaluation.native_history_reader import current as native_state

        native = native_state()
        history_graph = native is not None and native.graph_enabled

        return dict(
            enabled=True,
            captures=self.captures,
            replays=self.replays,
            warm_fills=self.warm_fills,
            maximum_workspace_bytes=self.maximum_workspace_bytes,
            graph_private_tensor_bytes=self.checked_private_tensor_bytes,
            graph_private_live_bytes=self.private_live_bytes,
            unowned_tensor_bytes=0,
            generator_scalar_allocations=len(self.generator_addresses),
            concurrent_graph_requests_supported=False,
            active_workspace_bytes=backing_bytes(
                tuple(
                    x
                    for x in (
                        self.buffer,
                        self.row4_workspace,
                        *(owned(self.lt_engine) if self.lt_engine else ()),
                    )
                    if x is not None
                )
            ),
            row4_workspace_peak_bytes=self.row4_workspace_bytes,
            maximum_total_reader_workspace_bytes=self.maximum_workspace_bytes
            + self.row4_workspace_bytes
            + self.lt_workspace_bytes,
            lt_workspace_peak_bytes=self.lt_workspace_bytes,
            lt_replays=self.lt_replays,
            lt_matmul_dispatches=self.lt_matmul_dispatches,
            lt_engine=self.lt_engine.record()
            if self.lt_engine is not None
            else self.lt_engine_info,
            vendor_internal_peak_certified=False,
            retired=self.closed,
            **graph_scope(self.joint_projection_policy),
            history_attention_and_ordered_replay_outside_graph=not (
                self.policy.captured_code_replay or history_graph
            ),
            history_and_attention_outside_graph=not history_graph,
            attention_outside_graph=True,
            capture_and_retirement_in_request_timing=True,
            driver_graph_metadata_not_KV_ownership_certified=True,
            independent_quality_N=0,
            optimized_policy=asdict(self.policy),
            prepared_replay_plan=dict(
                enabled=self.prepared_replay,
                hits=self.prepared_replay_hits,
                joint_key_consume_bypasses=self.prepared_consume_bypasses,
                CPU_only_metadata=True,
                additional_GPU_tensor_bytes=0,
            ),
            history_borrows=self.history_borrows,
            history_borrow_misses=self.history_borrow_misses,
            history_borrow_maximum_bytes=self.history_borrow_maximum_bytes,
            captured_code_replays=self.captured_code_replays,
            captured_code_kernel_dispatches=self.captured_code_kernel_dispatches,
            row4_replays=self.row4_replays,
            lookahead=dict(enabled=False) if self.lookahead is None else self.lookahead.record(),
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


class TrackedPromptGraphLayer(PromptGraphLayer):
    """Only event_sum uses these setters; old controls keep the old class."""

    @property
    def resident_bytes(self):
        return getattr(self, "_resident_bytes", 0)

    @resident_bytes.setter
    def resident_bytes(self, value):
        if type(value) is not int or value < 0:
            raise RuntimeError("nonnegative actual resident bytes required")
        owner = getattr(self, "_accounting_owner", None)
        old = getattr(self, "_resident_bytes", 0)
        if owner is not None:
            owner.accounting_delta(value - old)
        self._resident_bytes = value

    @PromptGraphLayer.lease_bytes.setter
    def lease_bytes(self, value):
        if type(value) is not int or value < 0:
            raise RuntimeError("nonnegative actual lease bytes required")
        self._ordinary_lease_bytes = value
        owner = getattr(self, "_accounting_owner", None)
        if owner is not None:
            charge = self.lease_bytes
            owner.accounting_delta(charge - self._tracked_lease_charge)
            self._tracked_lease_charge = charge


class PromptGraphCache(SourceHistoryCache):
    def __init__(self, *, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_graph_pool = PromptGraphPool(self, policy)
        from mosaickv.evaluation.graph_accounting import current

        self.accounting_policy = current()
        self.accounting_total = self.accounting_bound_layers = 0
        self.accounting_notifications = self.accounting_audits = 0
        self.accounting_ids = ()

    def accounting_delta(self, delta):
        if type(delta) is not int or self.accounting_total + delta < 0:
            raise RuntimeError("nonnegative exact integer live-storage total required")
        self.accounting_total += delta
        self.accounting_notifications += 1

    def bind_accounting(self):
        if len(self.layers) < self.accounting_bound_layers:
            raise RuntimeError("private fixed graph-cache layer list cannot shrink")
        if len(self.layers) == self.accounting_bound_layers:
            return
        for layer in self.layers[self.accounting_bound_layers :]:
            if (
                not isinstance(layer, TrackedPromptGraphLayer)
                or getattr(layer, "_accounting_owner", None) is not None
                or layer.prompt_graph_pool is not self.prompt_graph_pool
            ):
                raise RuntimeError("unique private tracked graph layer required")
            cost = layer.resident_bytes + layer.lease_bytes
            if type(cost) is not int or cost < 0:
                raise RuntimeError("nonnegative actual layer bytes required")
            self.accounting_total += cost
            layer._tracked_lease_charge = layer.lease_bytes
            layer._accounting_owner = self
        self.accounting_bound_layers = len(self.layers)
        self.accounting_ids = tuple(map(id, self.layers))

    def observe(self, extras=()):
        policy = getattr(self, "accounting_policy", "original")
        if policy != "original":
            if policy == "event_sum":
                self.bind_accounting()
                layer_bytes = self.accounting_total
                if self.audit_workspace:
                    actual = sum(x.resident_bytes + x.lease_bytes for x in self.layers)
                    if actual != layer_bytes or self.accounting_ids != tuple(map(id, self.layers)):
                        raise RuntimeError("event accounting differs from independent layer sum")
                    self.accounting_audits += 1
            else:
                layer_bytes = sum(x.resident_bytes + x.lease_bytes for x in self.layers)
            pool = getattr(self, "prompt_graph_pool", None)
            private = 0 if pool is None else pool.private_live_bytes
            if type(private) is not int or private < 0:
                raise RuntimeError("nonnegative charged graph-private bytes required")
            self.peak_bytes = max(
                self.peak_bytes, self.shared_bytes + layer_bytes + backing_bytes(extras) + private
            )
            self.observations += 1
            return
        super().observe(extras)
        pool = getattr(self, "prompt_graph_pool", None)
        private = 0 if pool is None else pool.private_live_bytes
        current = self.shared_bytes + sum(x.resident_bytes + x.lease_bytes for x in self.layers)
        self.peak_bytes = max(self.peak_bytes, current + backing_bytes(extras) + private)

    def storage_record(self):
        if self.accounting_policy == "event_sum":
            self.bind_accounting()
            if self.accounting_ids != tuple(map(id, self.layers)) or self.accounting_total != sum(
                x.resident_bytes + x.lease_bytes for x in self.layers
            ):
                raise RuntimeError("final independent graph accounting differs")
        record = super().storage_record()
        record["graph_cache_accounting"] = dict(
            policy=self.accounting_policy,
            notifications=self.accounting_notifications,
            audited_independent_comparisons=self.accounting_audits,
            final_independent_tensor_enumeration_preserved=True,
        )
        record["prompt_graph_reader"] = self.prompt_graph_pool.record()
        record["final_active_kv_storage_bytes"] += self.prompt_graph_pool.private_live_bytes
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
        for cache in caches:
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
            before = cache.storage_record()
            pool = cache.prompt_graph_pool
            if pool.captures != 4 or pool.replays != 42 or pool.maximum_workspace_bytes != 8192:
                raise RuntimeError("both routes, reused graphs and full backing charge required")
            pool.close()
            after = cache.storage_record()
            if (
                before["final_active_kv_storage_bytes"] - after["final_active_kv_storage_bytes"]
                != pool.maximum_workspace_bytes + 1024
            ):
                raise RuntimeError("workspace retirement disagrees with independent inventory")
        return dict(
            bitwise_prompt_and_history=True,
            two_requests_disjoint=True,
            captures_per_request=4,
            replays_per_request=42,
            full_workspace_backing_bytes_per_request=8192,
            separately_charged_generator_metadata_bytes_per_request=1024,
            serial_requests_only=True,
            retired=True,
            independent_quality_N=0,
            synthetic_control_only=True,
        )
    finally:
        for cache in caches:
            for layer in cache.layers:
                layer.release()
            cache.prompt_graph_pool.close()
