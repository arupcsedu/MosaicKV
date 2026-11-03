"""Request-private aliases and identical compiled restore-kernel ABI reuse.

No new GPU buffers, destination caching, model changes or cross-request state.
Every hit verifies immutable inputs and uses current output/head stride.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_state = ContextVar("evicastkv_native_execution_plan", default=None)


@contextmanager
def use(enabled):
    if type(enabled) is not bool or _state.get() is not None:
        raise ValueError("explicit nonnested request-private native execution plan required")
    state = None
    if enabled:
        import triton

        if triton.__version__ != "3.4.0":
            raise NotImplementedError("audited Triton 3.4.0 runner ABI required")
        state = dict(
            weights={},
            restores={},
            weight_hits=0,
            weight_misses=0,
            restore_hits=0,
            restore_misses=0,
        )
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)
        if state is not None:
            state["weights"].clear()
            state["restores"].clear()


def weight_view(weight, heads, dim):
    state = _state.get()
    if state is None:
        return weight.view(heads, dim, heads * dim).transpose(1, 2)
    key = id(weight)
    signature = (
        weight.data_ptr(),
        weight._version,
        weight.dtype,
        weight.device,
        tuple(weight.shape),
        tuple(weight.stride()),
        heads,
        dim,
    )
    entry = state["weights"].get(key)
    if entry is None:
        if len(state["weights"]) >= 64:
            raise RuntimeError("bounded native projection plan exhausted")
        view = weight.view(heads, dim, heads * dim).transpose(1, 2)
        state["weights"][key] = signature, view
        state["weight_misses"] += 1
        return view
    saved, view = entry
    if saved != signature:
        raise RuntimeError("original projection weight ownership/version changed")
    state["weight_hits"] += 1
    return view


@contextmanager
def suspend(disabled):
    """Operator control arms cannot accidentally use the candidate's plan."""
    token = _state.set(None) if disabled else None
    try:
        yield
    finally:
        if token is not None:
            _state.reset(token)


def launch(kernel, grid, arguments, options):
    state = _state.get()
    if state is None:
        return kernel[grid](*arguments, **options)
    if (
        len(arguments) != 13
        or any(type(x) is not int or x < 0 for x in arguments[9:])
        or arguments[12] <= 0
    ):
        raise ValueError("audited restore ABI and positive current head stride required")
    # Output pointer and HEAD_OUT vary every step; never store their views.
    # Input backing and the layout-independent scalar counts must stay fixed.
    signature = (
        id(kernel),
        tuple(grid),
        tuple(sorted(options.items())),
        tuple(
            (t.data_ptr(), t.dtype, t.device, tuple(t.shape), tuple(t.stride()))
            for t in arguments[:8]
        ),
        tuple(arguments[9:12]),
    )
    key = id(kernel), arguments[0].data_ptr(), arguments[6].data_ptr(), arguments[5].data_ptr()
    entry = state["restores"].get(key)
    if entry is None:
        if len(state["restores"]) >= 64:
            raise RuntimeError("bounded native restore plan exhausted")
        compiled = kernel[grid](*arguments, **options)
        if getattr(compiled.metadata, "global_scratch_size", None) != 0:
            raise RuntimeError("prepared restore cannot allocate opaque global scratch")
        runner = compiled[(*grid, *((1,) * (3 - len(grid))))]
        constants = tuple(options[name] for name in kernel.arg_names[len(arguments) :])
        state["restores"][key] = signature, compiled, runner, constants
        state["restore_misses"] += 1
        return compiled
    saved, compiled, runner, constants = entry
    if saved != signature:
        raise RuntimeError("immutable native restoration inputs or kernel ABI changed")
    runner(*arguments, *constants)
    state["restore_hits"] += 1
    return compiled


def restore(packet, component, destination):
    from mosaickv.evaluation.fused_source_restore import restore as original

    return original(
        packet,
        component,
        destination,
        record=False,
        launcher=launch if _state.get() is not None else None,
    )
