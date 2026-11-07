"""Bounded request-private host plans; no KV/weight/position transformation."""

from contextlib import contextmanager
from contextvars import ContextVar

POLICIES = frozenset(("original", "prepared", "restore", "prepared_restore"))
_state = ContextVar("evicastkv_source_hotpath", default=None)


def validate(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("registered explicit source hotpath policy required")


@contextmanager
def use(policy):
    validate(policy)
    if _state.get() is not None:
        raise RuntimeError("source hotpath state must be private and nonnested")
    if policy != "original":
        import triton

        if triton.__version__ != "3.4.0":
            raise NotImplementedError("audited Triton 3.4.0 required")
    state = (
        None
        if policy == "original"
        else dict(
            policy=policy,
            entries={},
            source_hits=0,
            source_misses=0,
            fused_restore_calls=0,
            fused_nonempty_calls=0,
            skipped_prefix_views=0,
        )
    )
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)
        if state is not None:
            state["entries"].clear()


def restore_enabled():
    state = _state.get()
    return state is not None and state["policy"] in ("restore", "prepared_restore")


def skip_prefix():
    state = _state.get()
    if restore_enabled():
        state["skipped_prefix_views"] += 1
        return True
    return False


def source_launch(kernel, grid, arguments, options, attention):
    """Same compiled binary/options; current destinations and strides every call.

    The original caller still validates every call. Unlike the earlier generic
    cache, a route is a fixed (attention module, component), not an expensive
    full launch-key walk. No tensor or destination view is stored in the plan.
    """
    state = _state.get()
    if state is None or state["policy"] not in ("prepared", "prepared_restore"):
        from mosaickv.evaluation.request_kernel_dispatch import launch

        return launch(kernel, grid, arguments, **options)
    component = options["COMPONENT"]
    if component not in (0, 1):
        # Primitive paired projection is outside the private one-buffer path.
        return kernel[grid](*arguments, **options)
    if (
        len(arguments) != 14
        or not hasattr(arguments[13], "data_ptr")
        or arguments[9] is not arguments[10]
        or any(type(x) is not int or not 0 < x < 2**31 for x in arguments[11:13])
        or arguments[9].data_ptr() % 16
        or arguments[9].device.type != "cuda"
        or str(arguments[9].dtype) != "torch.bfloat16"
        or arguments[9].device != arguments[0].device
    ):
        raise ValueError("aligned private one-buffer BF16 source lease required")
    # Header/weight/RoPE objects remain owned by the original frozen packets.
    # Code views are rebuilt by layer replay, so bind their actual pointer.
    signature = (
        arguments[0].data_ptr(),
        arguments[0].dtype,
        arguments[0].device,
        tuple("codes" if x is arguments[0] else id(x) for x in arguments[1:9]),
        arguments[11],
        arguments[13].data_ptr(),
        tuple(grid),
        tuple(sorted(options.items())),
    )
    key = id(kernel), id(attention), component
    entry = state["entries"].get(key)
    if entry is None:
        if len(state["entries"]) >= 128:
            raise RuntimeError("bounded source route plan exhausted")
        compiled = kernel[grid](*arguments, **options)
        if getattr(compiled.metadata, "global_scratch_size", None) != 0:
            raise RuntimeError("prepared source cannot introduce opaque global scratch")
        ordered_constants = tuple(options[n] for n in kernel.arg_names[len(arguments) :])
        runner = compiled[(*grid, *((1,) * (3 - len(grid))))]
        state["entries"][key] = signature, compiled, runner, ordered_constants
        state["source_misses"] += 1
    else:
        saved, compiled, runner, constants = entry
        if saved != signature:
            raise RuntimeError("immutable source route/ABI changed during the request")
        runner(*arguments, *constants)
        state["source_hits"] += 1
    return compiled


def record_restore(patches):
    if not restore_enabled():
        raise RuntimeError("unregistered sparse restore dispatch")
    _state.get()["fused_restore_calls"] += 1
    _state.get()["fused_nonempty_calls"] += int(patches > 0)


def gpu_controls():
    """Real kernels on synthetic inputs, not benchmark-quality evidence."""
    from types import SimpleNamespace

    import torch

    from mosaickv.evaluation import shared_source_kernel as source
    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.fused_source_restore import restore
    from mosaickv.evaluation.head_local_precision_kernel import scatter
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.shared_source_format import apply_transport_patch

    n, h, d = 17, 4, 128
    x = torch.randn(1, 1, n, h * d, device="cuda").bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    packet = SimpleNamespace(
        shape=(1, h, n, d),
        source=pack(x, bits=3, axis=2),
        cos=torch.ones(1, n, d, device="cuda", dtype=torch.bfloat16),
        sin=torch.zeros(1, n, d, device="cuda", dtype=torch.bfloat16),
    )
    with use("prepared") as state:
        for component in (0, 1):
            for tail in (5, 7):
                expected = torch.full((1, h, n + tail, d), -71, device="cuda", dtype=torch.bfloat16)
                actual = torch.full_like(expected, -71)
                # Explicit original path, no alternate weights or math.
                token = _state.set(None)
                try:
                    source.project(packet, attention, (expected, expected), component=component)
                finally:
                    _state.reset(token)
                source.project(packet, attention, (actual, actual), component=component)
                if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                    raise RuntimeError("prepared source changed projection/sentinel bits")
    if state["source_hits"] != 2 or state["source_misses"] != 2 or state["entries"]:
        raise RuntimeError("source launch hit or private-cleanup control missing")
    cases = []
    words = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    for rows in (13, 512):
        h, n = (3, 17) if rows == 13 else (4, 256)
        value = words[: rows * d].view(torch.bfloat16).reshape(1, 1, rows, d)
        positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 2
        indices = torch.tensor(
            [128, 3 * 128 + 7, (h * n - 1) * 128 + 127], device="cuda", dtype=torch.int32
        )
        patch = SparseBitPatch(
            indices, torch.tensor([31, -44, 7], device="cuda", dtype=torch.bfloat16)
        )
        for raw in (False, True):
            body = (
                PackedBF16(tuple(value.shape), True, (value.clone(),))
                if raw
                else encode(value, raw_fallback=False)
            )
            spec = SimpleNamespace(
                shape=(1, h, n, d),
                exact=(body, body),
                row_positions=positions,
                transport_patches=(patch, patch),
                protected_patches_disjoint=True,
            )
            expected = torch.full((1, h, n + 11, d), -71, device="cuda", dtype=torch.bfloat16)
            actual = torch.full_like(expected, -71)
            scatter(body, positions, expected, prompt_length=n)
            apply_transport_patch(patch, expected, prompt_length=n)
            with use("restore") as fusion:
                restore(spec, 0, actual)
            if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                raise RuntimeError("fused restore changed original bits or history sentinels")
            if fusion["fused_restore_calls"] != 1 or fusion["fused_nonempty_calls"] != 1:
                raise RuntimeError("fused restore branch did not engage")
            cases.append(dict(rows=rows, raw=raw, bitwise=True, history_sentinels_unchanged=True))
    torch.cuda.synchronize()
    return dict(
        source_hits=state["source_hits"],
        source_misses=state["source_misses"],
        fused_cases=cases,
        request_state_released=True,
        independent_quality_N=0,
        synthetic_numerical_control=True,
    )
