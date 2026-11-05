"""Default-off, request-private joint incoming-key/prompt transaction.

K/V values, attention, RoPE, selection and history math never change. This is
an execution variant. Warm keys reuse an already captured prompt graph before
cache.update; the ordinary K materialization consumes that transaction once.
V remains the original model projection. No placeholder values are introduced.
"""

from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from mosaickv.evaluation.joint_all_layer_compatibility import FROZEN
from mosaickv.evaluation.joint_kernel_compatibility import KernelPlans, fingerprint
from mosaickv.evaluation.lt_source_chunks import Engine as OriginalEngine
from mosaickv.evaluation.shared_source_kernel import _read_coherent

_scope = ContextVar("evicastkv_joint_incoming_projection", default=None)


def validate(policy):
    if type(policy) is not str or policy not in ("none", "keys", "kv"):
        raise ValueError("explicit registered none/keys/kv incoming-projection policy required")


def managed(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        policy = kwargs.get("joint_incoming_projection", "none")
        validate(policy)
        graph_plan = kwargs.get("reuse_graph_replay_plan", False)
        input_copy = kwargs.get("reuse_source_input_copy", False)
        if type(input_copy) is not bool:
            raise ValueError("explicit once-only source-copy boolean required")
        if type(graph_plan) is not bool:
            raise ValueError("explicit prepared graph replay boolean required")
        if _scope.get() is not None:
            raise RuntimeError("joint projection scopes may not nest or share requests")
        state = dict(
            policy=policy,
            transports=[],
            eligible_layers=0,
            graph_plan=graph_plan,
            input_copy=input_copy,
            input_copy_reports=[],
        )
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            details["joint_incoming_projection"] = dict(
                policy=policy,
                eligible_layers=state["eligible_layers"],
                transactions=state["transports"],
                value_projection_unchanged=policy != "kv" or not state["eligible_layers"],
                original_attention_and_RoPE=True,
                no_placeholder_values=True,
                no_new_selection_or_quality_claim=True,
                source_input_copy=dict(enabled=input_copy, reports=state["input_copy_reports"]),
            )
            return values, packets, details
        finally:
            _scope.reset(token)

    return wrapped


class Ledger:
    """CPU-only strict ordering; no tensor references or benchmark information."""

    def __init__(self, layers):
        if type(layers) is not int or layers < 1:
            raise ValueError("positive private layer count required")
        self.layers, self.calls, self.warm, self.cold = layers, 0, 0, 0
        self.pending = self.issuing = self.last_consumed = None

    def enter(self, index, history, *, ready):
        if self.pending is not None or self.issuing is not None:
            raise RuntimeError("previous joint key was not consumed before the next projection")
        if (
            type(index) is not int
            or type(history) is not int
            or (history, index) != divmod(self.calls, self.layers)
        ):
            raise RuntimeError("same-token ordered layer projection required")
        if type(ready) is not bool or ready != (history > 0):
            raise RuntimeError("only the first update may use uncaptured original projections")
        self.calls += 1
        if ready:
            self.pending = self.issuing = (index, history + 1)
            self.warm += 1
        else:
            self.cold += 1

    def produced(self):
        if self.pending is None or self.issuing != self.pending:
            raise RuntimeError("exactly one authorized joint key graph must complete")
        self.issuing = None

    def consume(self, index, history):
        if self.pending is None:
            if self.last_consumed == (index, history):
                raise RuntimeError("the joint key was already consumed at this layer/update")
            return False
        if self.issuing is not None or self.pending != (index, history):
            raise RuntimeError("joint key consumer differs from its layer/update transaction")
        self.last_consumed, self.pending = self.pending, None
        return True

    def finish(self, updates):
        if (
            self.pending is not None
            or self.issuing is not None
            or (
                self.calls != self.layers * updates
                or self.cold != self.layers
                or self.warm != self.layers * (updates - 1)
            )
        ):
            raise RuntimeError("incomplete or unconsumed bounded joint-key trajectory")
        return dict(
            key_calls=self.calls,
            folded_keys=self.warm,
            original_cold_keys=self.cold,
            pending=False,
            all_value_projections_original=True,
        )


@triton.jit
def _decode_joint(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    INPUT,
    OUT,
    ROWS,
    FIRST: tl.constexpr,
    TAKE: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
):
    local = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    x = _read_coherent(
        CODES,
        MINIMUM,
        SCALE,
        MAP,
        VALUES,
        FIRST + local,
        tl.program_id(1) * 64,
        ROWS,
        WIDTH,
        3,
        HAS_COLUMNS,
    )
    incoming = tl.load(
        INPUT + tl.broadcast_to(col[None, :], (16, 64)), local[:, None] == TAKE, other=0
    )
    value = tl.where(local[:, None] == TAKE, incoming, x)
    tl.store(OUT + local[:, None] * WIDTH + col[None, :], value, local[:, None] <= TAKE)


class Engine(OriginalEngine):
    def __init__(self, packet):
        super().__init__(packet)
        try:
            n = packet.shape[2]
            self.tail = n % 128
            if not 1 <= self.tail < 128:
                raise NotImplementedError("joint key requires a spare original 128-row slot")
            self.plans.close()
            self.plans = KernelPlans((128, self.tail, self.tail + 1))
            chosen = [(a, r) for a, r in self.plans.candidates if fingerprint(r) == FROZEN]
            if len(chosen) != 1:
                raise RuntimeError("the frozen all-layer-verified joint kernel is unavailable")
            a, b, d, _, _ = self.plans.plans[self.tail + 1]
            algorithm, record = chosen[0]
            self.plans.plans[self.tail + 1] = (
                a,
                b,
                d,
                algorithm,
                dict(
                    record, rows=self.tail + 1, workspace_bytes=0, compute_type="CUBLAS_COMPUTE_32F"
                ),
            )
            self.incoming = torch.zeros(4096, device=packet.cos.device, dtype=torch.bfloat16)
            self.extra_tensors = (self.incoming,)
            self.joint_calls = 0
            self.fold_values = _scope.get() is not None and _scope.get()["policy"] == "kv"
        except BaseException:
            self.close()
            raise

    @property
    def workspace_bytes(self):
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        return backing_bytes((self.source, self.output, *getattr(self, "extra_tensors", ())))

    def project(self, packet, attention, component, target, *, inspect_kernel=False):
        from mosaickv.evaluation.lt_source_chunks import decode_slice
        from mosaickv.evaluation.repaired_source_tile import validate as check
        from mosaickv.evaluation.shared_source_kernel import rotate_native
        from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

        if self.closed:
            raise RuntimeError("closed joint native engine")
        check(packet, attention, component, target)
        _, heads, n, dim = packet.shape
        if n % 128 != self.tail:
            raise ValueError("frozen same-geometry joint source required")
        weight = (attention.k_proj, attention.v_proj)[component].weight
        for first in range(0, n, 128):
            take = min(128, n - first)
            if take == self.tail:
                mapping, values, columns = kernel_arguments(packet)
                compiled = _decode_joint[(triton.cdiv(take + 1, 16), 64)](
                    packet.source.codes,
                    packet.source.minimum,
                    packet.source.scale,
                    mapping,
                    values,
                    self.incoming,
                    self.source,
                    n,
                    FIRST=first,
                    TAKE=take,
                    WIDTH=4096,
                    HAS_COLUMNS=columns,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                if compiled.metadata.global_scratch_size != 0 or compiled.n_spills != 0:
                    raise RuntimeError("joint decoder introduced scratch or spills")
                self.plans.execute(self.source, weight, self.output, take + 1)
                self.joint_calls += 1
            else:
                decode_slice(packet, self.source, first, take)
                self.plans.execute(self.source, weight, self.output, take)
            target[0, :, first : first + take, :].copy_(
                self.output[:take].view(take, heads, dim).transpose(0, 1)
            )
            self.matmul_calls += 1
        if component == 0:
            rotate_native(target, packet.cos, packet.sin, 0, n)
        self.project_calls += 1
        if inspect_kernel:
            return dict(
                algorithm_workspace_bytes=0,
                explicit_workspace_bytes=self.workspace_bytes,
                incoming_words_verified_by_separate_all_layer_gate=True,
            )

    def record(self):
        return dict(
            super().record(),
            joint_tail_calls=self.joint_calls,
            frozen_joint_configuration=FROZEN,
            extra_incoming_buffer_bytes=8192,
            input_source_copied_once_per_layer=True,
            value_projection_unchanged=not self.fold_values,
        )


@torch.inference_mode()
def gpu_decoder_controls():
    """Real bounded decoder words/sentinels, before checkpoint weights load."""
    from types import SimpleNamespace

    from mosaickv.evaluation.lt_source_chunks import decode_slice
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    generator = torch.Generator(device="cuda").manual_seed(20261105)
    records = []
    for n in (1, 127, 600, 604):
        raw = torch.randn((1, 1, n, 4096), generator=generator, device="cuda").bfloat16()
        body = pack(raw, bits=3, axis=2)
        packet = SimpleNamespace(source=body)
        take, first = n % 128, n - n % 128
        reference = torch.full((128, 4096), -71.0, device="cuda", dtype=torch.bfloat16)
        actual = torch.full_like(reference, -71.0)
        incoming = torch.randn(4096, generator=generator, device="cuda").bfloat16()
        incoming[:2] = torch.tensor([0.0, -0.0], device="cuda", dtype=torch.bfloat16)
        decode_slice(packet, reference, first, take)
        mapping, values, columns = kernel_arguments(packet)
        compiled = _decode_joint[(triton.cdiv(take + 1, 16), 64)](
            body.codes,
            body.minimum,
            body.scale,
            mapping,
            values,
            incoming,
            actual,
            n,
            FIRST=first,
            TAKE=take,
            WIDTH=4096,
            HAS_COLUMNS=columns,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if (
            compiled.metadata.global_scratch_size != 0
            or compiled.n_spills != 0
            or not torch.equal(actual[:take].view(torch.int16), reference[:take].view(torch.int16))
            or not torch.equal(actual[take].view(torch.int16), incoming.view(torch.int16))
            or not torch.equal(
                actual[take + 1 :].view(torch.int16), reference[take + 1 :].view(torch.int16)
            )
        ):
            raise RuntimeError("joint source decoder changed prompt/input/sentinel words")
        records.append(
            dict(
                source_length=n,
                prompt_words=take * 4096,
                incoming_words=4096,
                source_words_exact=True,
                incoming_signed_zeros_exact=True,
                unused_sentinels_preserved=True,
                spills=0,
                global_scratch=0,
            )
        )
    return dict(
        status="completed",
        real_CUDA_decoder=True,
        records=records,
        checkpoint_weights_not_loaded=True,
        independent_quality_N=0,
    )


def engine_factory(packet):
    state = _scope.get()
    if state is not None and state["policy"] in ("keys", "kv"):
        return Engine(packet)
    from mosaickv.evaluation.lt_source_chunks import Engine as DefaultEngine

    return DefaultEngine(packet)


def owned(engine):
    return engine.source, engine.output, *getattr(engine, "extra_tensors", ())


def reuse_graph_plan():
    state = _scope.get()
    return state is not None and state.get("graph_plan", False)


class Transport:
    def __init__(self, adapter, cache, state):
        self.adapter, self.cache, self.state = adapter, cache, state
        self.pool, self.stack = cache.prompt_graph_pool, ExitStack()
        self.layers = tuple(cache.layers)
        self.indices = {id(layer): i for i, layer in enumerate(self.layers)}
        self.mode = state["policy"]
        if self.mode == "kv":
            from mosaickv.evaluation.joint_value_transaction import ValueLedger

            self.ledger = ValueLedger(len(self.layers))
        else:
            self.ledger = Ledger(len(self.layers))
        self.closed = False
        self.input_copy = None

    def __enter__(self):
        from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
        from mosaickv.evaluation.lt_source_format import source_format

        if self.pool.lookahead is not None or self.pool.lt_engine is not None:
            raise RuntimeError("joint keys require a new pool without lookahead or foreign engine")
        native = source_format(self.layers[0].representation)
        if native is None or any(
            source_format(layer.representation) is None for layer in self.layers
        ):
            raise RuntimeError("all source layers must be compatible with the native reader")
        first = self.layers[0].representation
        if isinstance(first, BoundDeltaFormat):
            if first.replay.prepared_views is None:
                raise RuntimeError("joint engine requires already prepared private replay views")
            # Read only stable geometry/pointers. Do not advance or decode XOR state.
            native = source_format(first.replay.prepared_views[first.index])
        self.pool.lt_engine = engine_factory(native.base)
        owners = owned(self.pool.lt_engine)
        self.cache.observe(owners)
        self.cache.shared_tensors = (*self.cache.shared_tensors, *owners)
        self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
        self.pool.lt_workspace_bytes = backing_bytes(owners)
        self.cache.observe()
        self.pool.joint_projection = self
        self.pool.joint_projection_policy = self.mode
        self.state["eligible_layers"] += len(self.layers)
        try:
            if self.state.get("input_copy", False):
                from torch.nn.modules import module as torch_modules

                from mosaickv.evaluation.joint_input_copy import Copy

                if (
                    hasattr(self.cache, "joint_source_input_copy")
                    or torch_modules._global_forward_hooks
                    or torch_modules._global_forward_pre_hooks
                    or any(
                        type(x.attention.q_proj) is not torch.nn.Linear
                        or "forward" in vars(x.attention.q_proj)
                        or x.attention.q_proj._forward_hooks
                        or x.attention.q_proj._forward_pre_hooks
                        for x in self.layers
                    )
                ):
                    raise NotImplementedError(
                        "unchanged side-effect-free native query projection required"
                    )
                self.input_copy = Copy(self)
                self.cache.joint_source_input_copy = self.input_copy.write
                self.stack.callback(delattr, self.cache, "joint_source_input_copy")
            for index, layer in enumerate(self.layers):
                module, original_forward = layer.attention.k_proj, layer.attention.k_proj.forward

                def forward(hidden, *, index=index, layer=layer, original=original_forward):
                    if (
                        torch.is_grad_enabled()
                        or hidden.shape != (1, 1, 4096)
                        or hidden.dtype != torch.bfloat16
                        or hidden.device.type != "cuda"
                        or not layer.source_pending
                        or layer.leased
                        or self.closed
                    ):
                        raise RuntimeError("isolated source-captured one-token BF16 key required")
                    engine = self.pool.lt_engine
                    if self.input_copy is None:
                        engine.incoming.copy_(hidden.reshape(-1))
                    else:
                        self.input_copy.take(index, layer, hidden)
                    ready = (id(layer), 0) in self.pool.entries
                    self.ledger.enter(index, layer.history_length, ready=ready)
                    if not ready:
                        return original(hidden)
                    self.pool.reconstruct(
                        layer, 0, self.pool.buffer, _projection_epoch=layer.history_length + 1
                    )
                    self.ledger.produced()
                    return engine.output[engine.tail].view(1, 1, 4096)

                self.stack.enter_context(patch.object(module, "forward", forward))
                if self.mode == "kv":
                    from mosaickv.evaluation.joint_value_transaction import attention_forward

                    def no_separate_value(*args, **kwargs):
                        raise RuntimeError(
                            "separate V projection is forbidden in the joint-KV path"
                        )

                    self.stack.enter_context(
                        patch.object(layer.attention.v_proj, "forward", no_separate_value)
                    )
                    self.stack.enter_context(
                        patch.object(layer.attention, "forward", attention_forward(self, layer))
                    )
            return self
        except BaseException:
            self.stack.close()
            raise

    def authorized(self, layer, epoch, destination):
        index = self.indices.get(id(layer))
        return (
            not self.closed
            and self.ledger.issuing == (index, epoch)
            and epoch == layer.history_length + 1
            and not layer.leased
            and destination is self.pool.buffer
        )

    def take(self, layer, destination):
        if self.ledger.pending is None:
            return self.ledger.consume(self.indices[id(layer)], layer.history_length)
        if (
            self.closed
            or self.pool.closed
            or self.pool.history_borrowed
            or torch.cuda.is_current_stream_capturing()
            or not layer.leased
            or destination.data_ptr() != self.pool.buffer.data_ptr()
            or destination.stride() != self.pool.buffer.stride()
        ):
            raise RuntimeError("same-buffer ordinary joint-key consumer required")
        return self.ledger.consume(self.indices[id(layer)], layer.history_length)

    def __exit__(self, kind, value, traceback):
        self.stack.close()
        self.closed = True
        try:
            if kind is None:
                self.state["transports"].append(self.ledger.finish(127))
                if self.input_copy is not None:
                    self.state["input_copy_reports"].append(self.input_copy.record())
        finally:
            if self.input_copy is not None:
                self.input_copy.owner = None
                self.input_copy = None
            self.pool.joint_projection = None


def transport(adapter, cache):
    state = _scope.get()
    if state is None or state["policy"] == "none":
        return nullcontext()
    from mosaickv.evaluation.lt_source_format import source_format

    if not any(source_format(layer.representation) is not None for layer in cache.layers):
        return nullcontext()  # Baseline has no eligible prompt-source projection to share.
    return Transport(adapter, cache, state)
