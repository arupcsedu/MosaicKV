"""Default-off schedule-only source reuse across the private eager QK boundary.

History K is the last source projection in key materialization. After QK and
softmax consume K, reconstruct history V BEFORE prompt V overwrites the source.
Only reconstruction writes are reordered; the full value matmul, attention
order, original projectors/patches, cache positions and storage are unchanged.
No additional GPU tensor or retained tensor owner is introduced.
"""

import weakref
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

import torch

_active = ContextVar("evicastkv_history_value_first", default=None)


def signature(tensor):
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        version,
    )


@contextmanager
def use(enabled=True):
    from mosaickv.evaluation import mixed_history_projection as mixed
    from mosaickv.evaluation import native_history_feasibility as engine
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation import sequential_source_history as sequential

    if type(enabled) is not bool or _active.get() is not None:
        raise ValueError("explicit nonnested history-value schedule required")
    report = dict(
        enabled=enabled,
        eligible_keys=0,
        early_values=0,
        skipped_source_unpacks=0,
        skipped_duplicate_history_fills=0,
        unsupported_or_empty_history_calls=0,
        additional_GPU_storage_bytes=0,
        retained_tensor_owners=0,
        original_projectors_and_numerical_guards_unchanged=True,
        reconstruction_only_NOT_attention_summation_order_changed=True,
    )
    token = _active.set(report)
    pending, phase = [None], [None]
    original_materialize = sequential.SequentialSourceHistoryLayer.materialize_component
    original_fill = sequential.fill_component

    def unpack(original):
        @wraps(original)
        def call(packet, target):
            current = phase[0] if _active.get() is report else None
            if current is None:
                return original(packet, target)
            if packet is not current["packet"] or target is not current["state"].engine.source:
                raise RuntimeError("foreign source in private key/value schedule")
            current["unpacks"] += 1
            if current["unpacks"] != 1:
                raise RuntimeError("one complete consolidated source required")
            if current["kind"] == "key":
                return original(packet, target)
            if signature(target) != current["signature"]:
                raise RuntimeError("history source changed across the read-only score boundary")
            report["skipped_source_unpacks"] += 1

        return call

    def fill(packet, attention, component, target, **kwargs):
        current = phase[0] if _active.get() is report else None
        if current and current["kind"] == "prompt_value" and packet is current["packet"]:
            if component != 1 or signature(target)[1:-1] != current["target_signature"][1:-1]:
                raise RuntimeError("early value must occupy the identical physical history slots")
            current["skips"] += 1
            if current["skips"] != 1:
                raise RuntimeError("duplicate early-history consumption forbidden")
            report["skipped_duplicate_history_fills"] += 1
            return
        return original_fill(packet, attention, component, target, **kwargs)

    def materialize(layer, component):
        if not enabled or _active.get() is not report:
            return original_materialize(layer, component)
        state = native.current()
        packets = layer.source_sealed
        eligible = len(packets) == 1 and isinstance(
            packets[0], (mixed.MixedHistory, native.NativeHistoryFormat)
        )
        if not eligible:
            if pending[0] is not None:
                raise RuntimeError("pending history source lost its private layer")
            report["unsupported_or_empty_history_calls"] += 1
            return original_materialize(layer, component)
        if (
            component not in (0, 1)
            or not layer.leased
            or state is None
            or state.closed
            or state.graphs is not None
            or state.engine is None
            or torch.is_grad_enabled()
        ):
            raise RuntimeError("isolated nongrad eager single-lease history required")
        packet = packets[0]
        source = state.engine.source
        destination = layer.leased[0]
        if (
            source.requires_grad
            or source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
            or layer.leased[1].data_ptr() != destination.data_ptr()
        ):
            raise RuntimeError("history source and single KV lease must not alias")
        if component == 0:
            if pending[0] is not None or phase[0] is not None:
                raise RuntimeError("previous key/value source transaction is incomplete")
            current = dict(kind="key", packet=packet, state=state, unpacks=0)
            phase[0] = current
            try:
                result = original_materialize(layer, 0)
            finally:
                phase[0] = None
            if current["unpacks"] != 1:
                raise RuntimeError("actual key-source reconstruction must execute once")
            pending[0] = dict(
                layer=weakref.ref(layer),
                packet=weakref.ref(packet),
                state=weakref.ref(state),
                source=weakref.ref(source),
                signature=signature(source),
                stream=(
                    torch.cuda.current_stream(source.device).cuda_stream if source.is_cuda else None
                ),
            )
            report["eligible_keys"] += 1
            return result
        ticket = pending[0]
        if (
            ticket is None
            or ticket["layer"]() is not layer
            or ticket["packet"]() is not packet
            or ticket["state"]() is not state
            or ticket["source"]() is not source
            or signature(source) != ticket["signature"]
            or (
                source.is_cuda
                and torch.cuda.current_stream(source.device).cuda_stream != ticket["stream"]
            )
            or layer.current_component != 0
        ):
            raise RuntimeError("value requires the identical preceding private key source")
        start, end = layer.prompt_length, layer.prompt_length + packet.shape[-2]
        if end + layer.tail_used != layer.get_seq_length():
            raise RuntimeError("original logical history membership must remain complete")
        target = destination[..., start:end, :]
        current = dict(
            kind="early_value",
            packet=packet,
            state=state,
            unpacks=0,
            signature=ticket["signature"],
        )
        phase[0] = current
        try:
            original_fill(packet, layer.attention, 1, target)
            if current["unpacks"] != 1:
                raise RuntimeError("early V must reuse exactly one complete source")
            current.update(kind="prompt_value", skips=0, target_signature=signature(target))
            result = original_materialize(layer, 1)
            if current["skips"] != 1:
                raise RuntimeError("early V must replace exactly one original history fill")
            report["early_values"] += 1
            pending[0] = None
            return result
        finally:
            phase[0] = None

    try:
        with ExitStack() as stack:
            if enabled:
                stack.enter_context(
                    patch.object(
                        sequential.SequentialSourceHistoryLayer,
                        "materialize_component",
                        materialize,
                    )
                )
                stack.enter_context(patch.object(sequential, "fill_component", fill))
                stack.enter_context(
                    patch.object(mixed, "unpack_sources", unpack(mixed.unpack_sources))
                )
                stack.enter_context(
                    patch.object(engine, "unpack_source", unpack(engine.unpack_source))
                )
            yield report
            if pending[0] is not None:
                raise RuntimeError("history key/source transaction ended without its value")
    finally:
        pending[0] = phase[0] = None
        _active.reset(token)
