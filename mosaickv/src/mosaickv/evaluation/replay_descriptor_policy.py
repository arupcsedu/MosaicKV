"""Opt-in PRIVATE descriptor reuse. No decoded K/V or extra code buffers."""

from contextlib import contextmanager
from contextvars import ContextVar

_state = ContextVar("evicastkv_private_replay_descriptors", default=None)


def current():
    return _state.get()


@contextmanager
def use(enabled):
    if type(enabled) is not bool:
        raise ValueError("explicit private replay descriptor boolean required")
    state = dict(prepared_formats=0, reused_formats=0) if enabled else None
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


def gpu_controls():
    """Synthetic exact-code/ownership control, never a model-quality sample."""
    from dataclasses import replace

    import torch

    from mosaickv.evaluation import layer_code_delta as runtime
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.precision_tier import PrecisionFormat, QuantizedBody

    packets = []
    for i in range(32):
        codes = torch.zeros(1024, device="cuda", dtype=torch.uint8)
        codes[i] = 1
        body = QuantizedBody(
            (1, 1, 8, 128),
            8,
            3,
            128,
            codes,
            torch.zeros((1, 1, 8, 1), device="cuda"),
            torch.ones((1, 1, 8, 1), device="cuda"),
        )
        packets.append(
            PrecisionFormat(
                body.shape,
                (body, replace(body, codes=codes.clone())),
                torch.empty(0, device="cuda", dtype=torch.int32),
                torch.empty(0, device="cuda", dtype=torch.int32),
                tuple(
                    torch.empty((1, 1, 0, 128), device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                ),
            )
        )
    encoded, report = runtime.convert(packets)
    if not report["enabled"]:
        raise RuntimeError("synthetic code chain must be genuinely compressible")
    with use(False):
        ordinary = runtime.bind(encoded)
    with use(True) as state:
        prepared = runtime.bind(encoded)
    expected = report["net_active_bytes"]
    for group in (ordinary, prepared):
        if backing_bytes(tuple(t for p in group for t in p.tensors())) != expected:
            raise RuntimeError("descriptor reuse changed backing inventory")
    if any(
        ordinary[0].replay.scratch[r].data_ptr() == t.data_ptr()
        for r, t in prepared[0].replay.scratch.items()
    ):
        raise RuntimeError("requests share mutable replay buffers")
    for _ in range(3):
        for i in range(32):
            a, b = ordinary[i].replay.view(i), prepared[i].replay.view(i)
            if not all(
                torch.equal(x.codes, y.codes) and torch.equal(x.codes, z.codes)
                for x, y, z in zip(a.pair, b.pair, packets[i].pair, strict=True)
            ):
                raise RuntimeError("prepared descriptors changed an original code bit")
    return dict(
        synthetic_numerical_control=True,
        descriptor_preparations=state["prepared_formats"],
        descriptor_reuses=state["reused_formats"],
        backing_bytes_identical=True,
        private_scratch_disjoint=True,
        code_bits_identical=True,
        no_model_weights_loaded=True,
        no_comparative_latency_claim=True,
    )
