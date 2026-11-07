"""Bounded request-local guard records, validated before returning predictions.

GPU acceptance still includes the original per-step finite/KL/max/token tests.
Only CPU record transfer and request rejection are deferred. Do not use CUDA
device assertions: they poison the context and are unsuitable for normal error
handling (see the installed torch._assert_async documentation).
"""

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class DeviceRecords:
    def __init__(self, maximum_updates, device):
        if type(maximum_updates) is not int or not 1 <= maximum_updates <= 127:
            raise ValueError("registered bounded guard horizon required")
        self.data = torch.empty((maximum_updates, 5), dtype=torch.float64, device=device)
        self.bytes = backing_bytes((self.data,))
        self.used = 0
        self.checked = self.failed = False
        self.invalid_steps = []

    def append(self, value):
        if (
            self.data is None
            or self.failed
            or self.used >= self.data.shape[0]
            or value.shape != (5,)
            or value.dtype != self.data.dtype
            or value.device != self.data.device
        ):
            raise ValueError("live bounded aligned FP64 guard record required")
        self.data[self.used].copy_(value)
        self.used += 1
        self.checked = False

    def snapshot(self):
        if self.data is None:
            raise RuntimeError("guard records already retired")
        rows = self.data[: self.used].detach().cpu().clone()
        self.checked = True
        self.invalid_steps = (rows[:, 4] != 1).nonzero().flatten().tolist()
        self.failed = bool(self.invalid_steps)
        if self.failed:
            raise ValueError(
                f"nonfinite logits invalidate request at guard steps {self.invalid_steps}"
            )
        if not torch.isfinite(rows).all():
            self.failed = True
            raise ValueError("nonfinite guard diagnostics invalidate request")
        return rows

    def release(self):
        self.data = None


def retire_before_pool(pool, tail, original_close):
    """Always run the original strict pool teardown, including failure paths."""
    error = None
    if tail is not None and not tail.closed:
        try:
            tail.close()
        except Exception as exc:
            error = exc
    try:
        result = original_close(pool)
    except Exception as exc:
        if error is not None:
            raise exc from error
        raise
    if error is not None:
        raise error
    return result
