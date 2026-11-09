"""Default-off launch fusion of the registered mixed-history source decoder.

One kernel replaces up to five unpack launches. All stored words, arithmetic,
original GEMMs, RoPE and numerical corrections remain unchanged. Plans retain
only weak tensor references and compiled binaries, never extra GPU owners.
"""

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from unittest.mock import patch

import torch

_state = ContextVar("evicastkv_fused_history_unpack", default=None)


@dataclass
class Binding:
    packet: object
    pointers: tuple
    geometry: dict
    grid: tuple
    compiled: object = None
    runner: object = None

    def arguments(self):
        result = tuple(pointer() for pointer in self.pointers)
        if self.packet() is None or any(t is None for t in result):
            raise RuntimeError("expired request-private source binding")
        return result


def bind(packet, destination, *, device_type="cuda"):
    """Validate complete immutable source geometry before compiling a binding.

    CPU device_type is only for fixture tests; actual launch rejects CPU.
    """
    from mosaickv.evaluation.lossless_bf16 import PackedBF16

    recent = packet.recent
    if (
        not 1 <= len(recent) <= 4
        or destination.dtype != torch.bfloat16
        or destination.device.type != device_type
        or not destination.is_contiguous()
        or destination.ndim != 2
        or destination.shape[1] != 4096
        or destination.shape[0] < packet.shape[-2]
    ):
        raise ValueError(
            "one to four exact packets and sufficient contiguous BF16 workspace required"
        )
    arguments, raw = [], []
    for chunk in recent:
        body = chunk.archive.source
        if (
            not isinstance(body, PackedBF16)
            or chunk.shape != (1, 32, 16, 128)
            or body.shape != (1, 1, 16, 4096)
            or body.shared_literal_window
            or type(body.raw) is not bool
        ):
            raise ValueError("complete original 16-token BF16 packets required")
        if body.raw:
            if (
                len(body.buffers) != 1
                or body.buffers[0].dtype != torch.bfloat16
                or body.buffers[0].numel() != 65536
            ):
                raise ValueError("complete raw BF16 source required")
            pointers = (body.buffers[0],) * 5  # Dead packed branch never reads these aliases.
        else:
            if len(body.buffers) != 5:
                raise ValueError("complete BF16 sign/code/base/offset/literal buffers required")
            literals = body.buffers[-1]
            if (
                tuple(t.dtype for t in body.buffers)
                != (torch.uint8, torch.uint8, torch.uint8, torch.int32, torch.uint8)
                or tuple(t.numel() for t in body.buffers[:4]) != (65536, 24576, 64, 65)
                or literals.numel() > 65536
            ):
                raise ValueError("invalid complete 1024-word BF16 block geometry")
            pointers = body.buffers
        arguments.extend(pointers)
        raw.append(body.raw)
    # Unreachable extra branches alias the first source; allocate no dummy buffer.
    while len(raw) < 4:
        arguments.extend(arguments[:5])
        raw.append(raw[0])
    old_rows, group = 0, 32
    old = (arguments[0],) * 3
    if packet.older is not None:
        body = packet.older.body
        old_rows, group = body.shape[-2], body.group
        if (
            old_rows not in (16, 32, 48)
            or packet.older.shape != (1, 32, old_rows, 128)
            or body.shape != (1, 1, old_rows, 4096)
            or body.bits != 8
            or body.axis != 3
            or group not in (32, 128)
            or body.row_map is not None
            or body.codebook is not None
            or body.codes.dtype != torch.uint8
            or body.codes.numel() != old_rows * 4096
            or body.minimum.dtype != torch.float32
            or body.scale.dtype != torch.float32
            or body.minimum.shape != (1, 1, old_rows, 4096 // group)
            or body.scale.shape != body.minimum.shape
        ):
            raise ValueError("audited older INT8 feature-group body required")
        old = body.codes, body.minimum, body.scale
    n = old_rows + 16 * len(recent)
    if packet.shape != (1, 32, n, 128) or n > 112:
        raise ValueError("complete mixed-history positions required")
    arguments = (*old, *arguments, destination)
    if any(t.device != destination.device or not t.is_contiguous() for t in arguments):
        raise ValueError("contiguous buffers on the same private device required")
    return Binding(
        weakref.ref(packet),
        tuple(weakref.ref(t) for t in arguments),
        dict(
            OLD_ROWS=old_rows, GROUP=group, RAW_A=raw[0], RAW_B=raw[1], RAW_C=raw[2], RAW_D=raw[3]
        ),
        (n * 4,),
    )


def replay(entry, arguments):
    entry.runner(*arguments, *entry.geometry.values())


def launch(packet, destination):
    state = _state.get()
    if state is None or destination.device.type != "cuda":
        raise RuntimeError("active request-private CUDA fusion context required")
    key = id(packet)
    entry = state["bindings"].get(key)
    if entry is None:
        from mosaickv.evaluation.fused_history_unpack_kernel import unpack

        entry = bind(packet, destination)
        if len(state["bindings"]) >= 256:
            raise RuntimeError("bounded live mixed-history plans exhausted")
        # Weakref callback retires host plans without keeping source buffers alive.
        entry.packet = weakref.ref(packet, lambda _: state["bindings"].pop(key, None))
        args = entry.arguments()
        entry.compiled = unpack[entry.grid](*args, **entry.geometry, enable_fp_fusion=False)
        if entry.compiled.metadata.global_scratch_size or entry.compiled.n_spills:
            raise RuntimeError("fused source decoder introduced undeclared kernel scratch")
        entry.runner = entry.compiled[(*entry.grid, 1, 1)]
        state["bindings"][key] = entry
        state["prepared"] += 1
        state["maximum_live_plans"] = max(state["maximum_live_plans"], len(state["bindings"]))
    else:
        args = entry.arguments()
        if entry.packet() is not packet or args[-1] is not destination:
            raise RuntimeError("cross-request/source workspace substitution forbidden")
        # Triton's compiled launcher still parses constexpr arguments in the
        # original Python ABI, even though they are absent from CUDA params.
        replay(entry, args)
        state["reused"] += 1


@contextmanager
def use():
    import triton

    from mosaickv.evaluation import mixed_history_projection as mixed

    if _state.get() is not None or triton.__version__ != "3.4.0":
        raise ValueError("nonnested audited Triton 3.4.0 fusion context required")
    state = dict(bindings={}, prepared=0, reused=0, maximum_live_plans=0)
    token = _state.set(state)
    report = dict(
        variant="single_launch_mixed_history_source_unpack",
        additional_GPU_storage_bytes=0,
        precision_policy_unchanged=True,
        projection_RoPE_repairs_unchanged=True,
        source_tensor_owners_retained_by_plans=0,
    )
    try:
        with patch.object(mixed, "unpack_sources", launch):
            yield report
    finally:
        _state.reset(token)
        report.update({k: v for k, v in state.items() if k != "bindings"})
        state["bindings"].clear()


@torch.inference_mode()
def gpu_controls():
    """Actual kernels; arbitrary BF16 words include every NaN/sign/zero pattern."""
    from types import SimpleNamespace

    from mosaickv.evaluation.history_source_precision import pack_source
    from mosaickv.evaluation.lossless_bf16 import pack_cpu
    from mosaickv.evaluation.mixed_history_projection import unpack_sources

    words = (
        torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16).reshape(1, 1, 16, 4096)
    )
    packed = pack_cpu(words, raw_fallback=False).to("cuda")
    raw = pack_cpu(words, raw_fallback=True).to("cuda")
    records = []
    for old_rows, group, count in ((0, 32, 1), (0, 32, 4), (16, 32, 4), (32, 128, 4), (48, 32, 4)):
        recent = tuple(
            SimpleNamespace(shape=(1, 32, 16, 128), archive=SimpleNamespace(source=body))
            for body in (packed, raw, packed, raw)[:count]
        )
        older = None
        if old_rows:
            x = torch.randn(1, 1, old_rows, 4096, device="cuda").bfloat16()
            older = SimpleNamespace(body=pack_source(x, group=group), shape=(1, 32, old_rows, 128))

        # Dataclass objects are weak-referenceable, unlike SimpleNamespace.
        @dataclass
        class Packet:
            shape: tuple
            recent: tuple
            older: object

        n = old_rows + count * 16
        packet = Packet((1, 32, n, 128), recent, older)
        expected = torch.full((128 + 7, 4096), -91, dtype=torch.bfloat16, device="cuda")
        actual = expected.clone()
        unpack_sources(packet, expected)
        with use() as report:
            for _ in range(3):
                launch(packet, actual)
                if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                    raise RuntimeError("fused mixed source changed bits or workspace sentinels")
        if report["prepared"] != 1 or report["reused"] != 2:
            raise RuntimeError("original compiled-kernel reuse did not execute")
        records.append(
            dict(
                older_tokens=old_rows,
                group=group,
                recent_chunks=count,
                source_words_bitwise=True,
                tail_sentinels_intact=True,
                **report,
            )
        )
    return records
