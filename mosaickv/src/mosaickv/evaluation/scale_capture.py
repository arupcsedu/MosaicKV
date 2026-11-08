"""Read-only CPU archive of actual affine headers; never retain a GPU owner."""

from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def use():
    from mosaickv.evaluation import layer_code_delta as runtime

    original = runtime.convert
    rows = {}

    def convert(packets):
        if rows:
            raise RuntimeError("one complete immutable source capture per request required")
        packets = tuple(packets)
        for layer, packet in enumerate(packets):

            def collect(role, body, *, layer=layer):
                if body.scale.dtype != torch.float32:
                    raise ValueError("original FP32 scales required for exact feasibility")
                rows[(layer, str(role))] = dict(
                    shape=body.shape,
                    bits=body.bits,
                    axis=body.axis,
                    group=body.group,
                    minimum=body.minimum.detach().to(device="cpu", copy=True),
                    scale=body.scale.detach().to(device="cpu", copy=True),
                )
                return body

            runtime.map_bodies(packet, collect)
        return original(packets)

    with patch.object(runtime, "convert", convert):
        yield rows
