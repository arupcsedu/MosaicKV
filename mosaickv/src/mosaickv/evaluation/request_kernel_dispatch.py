"""Opt-in request-private reuse of an already validated Triton projection.

Uses Triton 3.4.0 CompiledKernel.__getitem__, not a custom CUDA launch ABI.
All projection validation still runs on every call. No tensors or extra GPU
buffers are retained; a hit reuses the SAME compiled binary and launch options.
This does not fuse operators, alter arithmetic, or change launch geometry.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_cache = ContextVar("evicastkv_request_kernel_dispatch", default=None)


@contextmanager
def use_dispatch(enabled):
    if type(enabled) is not bool:
        raise ValueError("explicit dispatch reuse boolean required")
    if enabled:
        import triton

        if triton.__version__ != "3.4.0":
            raise NotImplementedError("audited Triton 3.4.0 compiled dispatch only")
    state = dict(handles={}, hits=0, misses=0) if enabled else None
    token = _cache.set(state)
    try:
        yield state
    finally:
        _cache.reset(token)
        if state is not None:
            state["handles"].clear()


def _argument_key(value, *, dynamic=False):
    if type(value) is int:
        kind = "i32" if -(2**31) <= value < 2**31 else "u64" if 2**63 <= value < 2**64 else "i64"
        return (kind,) if dynamic else (kind, value)
    if type(value) in (bool, float) or value is None:
        return type(value).__name__, value
    # Keep pointer identity conservative, including view offset; no tensor
    # references in the cache. Projection validation guards shape/stride.
    if hasattr(value, "data_ptr"):
        return "tensor", value.dtype, value.device, value.data_ptr()
    raise TypeError("audited projection tensor/scalar arguments only")


def launch(kernel, grid, args, **kwargs):
    state = _cache.get()
    if state is None:
        return kernel[grid](*args, **kwargs)
    names = kernel.arg_names
    ordered = (*args, *(kwargs[name] for name in names[len(args) :]))
    dynamic = set(kernel.do_not_specialize)
    key = (
        id(kernel),
        tuple(
            _argument_key(v, dynamic=i in dynamic or names[i] in dynamic)
            for i, v in enumerate(ordered)
        ),
        tuple(sorted((k, v) for k, v in kwargs.items() if k not in names)),
    )
    compiled = state["handles"].get(key)
    if compiled is None:
        compiled = kernel[grid](*args, **kwargs)
        if getattr(compiled.metadata, "global_scratch_size", None) != 0:
            raise RuntimeError("compiled reuse cannot introduce opaque global scratch")
        if len(state["handles"]) >= 32768:
            raise RuntimeError("bounded request dispatch cache exhausted")
        state["handles"][key] = compiled
        state["misses"] += 1
    else:
        # Runtime scalar values and pointers are passed anew, never captured.
        shape = (*grid, *((1,) * (3 - len(grid))))
        compiled[shape](*ordered)
        state["hits"] += 1
    return compiled
