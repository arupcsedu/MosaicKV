"""Default-off exact source reuse within an immediate K/V fill transaction.

Only constructor/verification pairs qualify. Ordinary sequential decode, where
prompt V reconstruction can overwrite the workspace between K and V, NEVER
reuses a source. All original projections and three verification passes remain.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

_pair = ContextVar("evicastkv_immediate_history_pair", default=None)
_active = ContextVar("evicastkv_paired_history_policy", default=False)


@contextmanager
def use():
    from mosaickv.evaluation import mixed_history_projection as mixed
    from mosaickv.evaluation import native_history_feasibility as native_engine
    from mosaickv.evaluation import native_history_reader as native

    if _active.get():
        raise RuntimeError("nonnested paired history source policy required")
    token = _active.set(True)
    report = dict(
        variant="immediate_history_KV_pair_source_reuse",
        original_unpacks=0,
        reused_sources=0,
        paired_transactions=0,
        added_GPU_storage_bytes=0,
        original_projections_and_verification_unchanged=True,
        ordinary_sequential_decode_reuse=False,
    )

    def paired(original):
        @wraps(original)
        def fill(packet, attention, pair, *, direct):
            if _pair.get() is not None:
                raise RuntimeError("nested source-pair transaction forbidden")
            state = native.current()
            if state is None or state.closed or state.graphs is not None:
                # Preserve original rejection/graph behavior; no reuse is claimed.
                return original(packet, attention, pair, direct=direct)
            context = dict(packet=packet, source=None, count=0, pair=pair, engine=state.engine)
            pair_token = _pair.set(context)
            try:
                result = original(packet, attention, pair, direct=direct)
                if context["count"] != 2:
                    raise RuntimeError("incomplete immediate K/V source transaction")
                report["paired_transactions"] += 1
                return result
            finally:
                _pair.reset(pair_token)

        return fill

    def reuse(original):
        @wraps(original)
        def unpack(packet, destination):
            context = _pair.get()
            if context is None:
                report["original_unpacks"] += 1
                return original(packet, destination)
            if context["packet"] is not packet or destination is not context["engine"].source:
                raise RuntimeError("foreign source in immediate K/V transaction")
            if context["count"] == 0:
                owner = destination.untyped_storage().data_ptr()
                if (
                    any(t.untyped_storage().data_ptr() == owner for t in context["pair"])
                    or context["engine"].output.untyped_storage().data_ptr() == owner
                ):
                    raise RuntimeError(
                        "source reuse requires nonaliasing K/V and projection outputs"
                    )
                original(packet, destination)
                context["source"] = destination
                context["count"] = 1
                report["original_unpacks"] += 1
            elif context["count"] == 1 and context["source"] is destination:
                # Only a read-only native GEMM, output copy, output RoPE and
                # original numerical patches executed since the preceding call.
                context["count"] = 2
                report["reused_sources"] += 1
            else:
                raise RuntimeError("source pair may reuse its exact result only once")

        return unpack

    try:
        with (
            patch.object(mixed, "unpack_sources", reuse(mixed.unpack_sources)),
            patch.object(native_engine, "unpack_source", reuse(native_engine.unpack_source)),
            patch.object(mixed.MixedHistory, "fill", paired(mixed.MixedHistory.fill)),
            patch.object(
                native.NativeHistoryFormat, "fill", paired(native.NativeHistoryFormat.fill)
            ),
        ):
            yield report
    finally:
        _active.reset(token)
