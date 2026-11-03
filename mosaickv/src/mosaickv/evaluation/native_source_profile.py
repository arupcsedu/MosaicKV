"""Consumed-input reconstruction study, not a serving-policy promotion.

Raw CUDA event intervals include stream idle gaps. Intermediate events perturb
dispatch; uninstrumented pipelines are measured separately. No model answers or
references enter operator selection. All extra source/output buffers are charged.
"""

import random
import statistics
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from mosaickv.evaluation import full_prompt_source as native
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

LAYERS = (0, 16, 31)
PADDINGS = (0, 64, 127)
PRODUCTS = ("bmm", "wide", "wide_128")
PHASES = (
    "workspace_allocate",
    "source_unpack",
    "product",
    "wide_copy",
    "rope",
    "protected_restore",
    "transport_restore",
    "numerical_repair",
    "workspace_release",
    "project_inclusive",
)


def wide_product(workspace, weights, output, buffer, *, chunk=None, timer=None):
    """One wide GEMM or fixed 128-token tiles, followed by explicit layout copy.

    Buffer is caller-owned and charged, never an implicit unreported output.
    Original head-major projection weights have a zero-copy wide view.
    """
    h, n, d = output.shape
    if (
        workspace.shape != (1, n, h * d)
        or not workspace.is_contiguous()
        or weights.shape != (h, h * d, d)
        or buffer.dtype != workspace.dtype
        or buffer.device != workspace.device
        or not buffer.is_contiguous()
        or buffer.ndim != 2
        or buffer.shape[1] != h * d
        or type(chunk) not in (int, type(None))
        or (chunk is not None and chunk != 128)
        or buffer.shape[0] < min(n, chunk or n)
    ):
        raise ValueError("charged contiguous wide-product buffer and registered tile required")
    full_weight = weights.transpose(1, 2).view(h * d, h * d).t()
    step = chunk or n
    for start in range(0, n, step):
        end = min(n, start + step)
        target = buffer[: end - start]
        if timer is None:
            torch.mm(workspace[0, start:end], full_weight, out=target)
            output[:, start:end].copy_(target.view(end - start, h, d).transpose(0, 1))
        else:
            timer.call("product", torch.mm, workspace[0, start:end], full_weight, out=target)
            timer.call(
                "wide_copy",
                output[:, start:end].copy_,
                target.view(end - start, h, d).transpose(0, 1),
            )


class Events:
    def __init__(self):
        self.pending = []

    def call(self, phase, function, *args, **kwargs):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        host = time.perf_counter()
        begin.record()
        result = function(*args, **kwargs)
        end.record()
        self.pending.append((phase, begin, end, time.perf_counter() - host))
        return result

    def rows(self):
        torch.cuda.synchronize()
        return [
            dict(phase=p, cuda_interval_ms=a.elapsed_time(b), host_seconds=t)
            for p, a, b, t in self.pending
        ]


@contextmanager
def product_mode(mode, *, timer=None):
    """Isolated study only. The production packet format/default stays unchanged."""
    if mode not in (*PRODUCTS, "fused"):
        raise ValueError("registered product required")
    original = native.projection_product
    buffers = []

    def product(workspace, weights, output):
        if mode in ("bmm", "fused"):
            if timer is None:
                return original(workspace, weights, output)
            return timer.call("product", original, workspace, weights, output)
        h, n, d = output.shape
        shape = (min(n, 128) if mode == "wide_128" else n, h * d)
        if not buffers or buffers[0].shape != shape:

            def allocate():
                return torch.empty(shape, device=workspace.device, dtype=workspace.dtype)

            buffer = allocate() if timer is None else timer.call("workspace_allocate", allocate)
            buffers[:] = [buffer]
        return wide_product(
            workspace,
            weights,
            output,
            buffers[0],
            chunk=128 if mode == "wide_128" else None,
            timer=timer,
        )

    with patch.object(native, "projection_product", product):
        yield buffers


@contextmanager
def instrument(timer, candidate):
    from mosaickv.evaluation import fused_source_restore as fused
    from mosaickv.evaluation import head_local_precision_kernel as local
    from mosaickv.evaluation import shared_source_format as shared
    from mosaickv.evaluation import shared_source_kernel as kernel

    with ExitStack() as stack:
        for module, name, phase in (
            (native, "decode_into", "source_unpack"),
            (native, "project_into", "project_inclusive"),
            (fused, "restore", "fused_restore"),
            (kernel, "rotate_native", "rope"),
            (local, "scatter", "protected_restore"),
        ):
            function = getattr(module, name)
            stack.enter_context(
                patch.object(
                    module,
                    name,
                    lambda *a, _f=function, _p=phase, **kw: timer.call(_p, _f, *a, **kw),
                )
            )
        restore = shared.apply_transport_patch
        repairs = {id(p) for p in candidate.repairs}

        def apply(patch_, *args, **kwargs):
            phase = "numerical_repair" if id(patch_) in repairs else "transport_restore"
            return timer.call(phase, restore, patch_, *args, **kwargs)

        stack.enter_context(patch.object(shared, "apply_transport_patch", apply))
        yield


def aggregate(records):
    groups = {}
    for row in records:
        if row["warmup"]:
            continue
        key = row["mode"]
        groups.setdefault(key, []).append(row)
    return {
        mode: dict(
            mean_cuda_ms=statistics.mean(x["cuda_interval_ms"] for x in rows),
            median_cuda_ms=statistics.median(x["cuda_interval_ms"] for x in rows),
            mean_host_seconds=statistics.mean(x["host_seconds"] for x in rows),
            count=len(rows),
        )
        for mode, rows in groups.items()
    }


@torch.inference_mode()
def study(base, attention, layer, padding, *, seed, products=PRODUCTS):
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    shape = (*base.shape[:2], base.shape[-2] + padding, base.shape[-1])
    targets = tuple(
        torch.full(shape, -71, dtype=torch.bfloat16, device=base.cos.device) for _ in range(2)
    )
    reference = tuple(torch.empty_like(t) for t in targets)
    for c in (0, 1):
        fill_component(base, attention, c, reference[c])
    candidates, correctness = {}, {}
    for mode in products:
        with product_mode(mode):
            candidate = native.fit(base, attention, (0, 1))
            if mode == "fused":
                candidate = native.enable_fused_restore(candidate)
            workspace = native.new_workspace(candidate, base.cos.device)
            for c in (0, 1):
                candidate.fill_component(attention, c, targets[c], workspace)
                if not torch.equal(
                    targets[c][..., : base.shape[-2], :].view(torch.int16),
                    reference[c][..., : base.shape[-2], :].view(torch.int16),
                ) or not bool((targets[c][..., base.shape[-2] :, :] == -71).all()):
                    raise RuntimeError("product changed parent bits or history sentinel")
            candidates[mode] = candidate
            source = backing_bytes((workspace,))
            correctness[mode] = dict(
                bitwise=True,
                history_sentinel=True,
                repair_bytes=backing_bytes(tuple(t for p in candidate.repairs for t in p.tensors)),
                source_workspace_bytes=source,
                extra_output_workspace_bytes=0
                if mode in ("bmm", "fused")
                else (min(base.shape[-2], 128) if mode == "wide_128" else base.shape[-2])
                * base.shape[1]
                * base.shape[-1]
                * 2,
            )
            del workspace

    def run(mode, timer=None):
        if mode == "original":
            for c in (0, 1):
                fill_component(base, attention, c, targets[c])
            return
        candidate = candidates[mode]

        def allocate():
            return native.new_workspace(candidate, base.cos.device)

        workspace = allocate() if timer is None else timer.call("workspace_allocate", allocate)
        with product_mode(mode, timer=timer) as buffers:
            with instrument(timer, candidate) if timer is not None else ExitStack():
                for c in (0, 1):
                    candidate.fill_component(attention, c, targets[c], workspace)
            host = time.perf_counter()
            buffers.clear()
            del workspace
            if timer is not None:
                # Host allocator release, not a fabricated GPU kernel time.
                releases.append(
                    dict(phase="workspace_release", host_seconds=time.perf_counter() - host)
                )

    raw, releases = [], []
    rng = random.Random(seed + 1000 * layer + padding)
    for trial in range(-5, 20):
        modes = ["original", *products]
        rng.shuffle(modes)
        for mode in modes:
            torch.cuda.synchronize()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            host = time.perf_counter()
            begin.record()
            run(mode)
            end.record()
            torch.cuda.synchronize()
            raw.append(
                dict(
                    mode=mode,
                    trial=trial,
                    warmup=trial < 0,
                    cuda_interval_ms=begin.elapsed_time(end),
                    host_seconds=time.perf_counter() - host,
                )
            )
    phases = []
    for mode in products:
        for trial in range(5):
            timer = Events()
            run(mode, timer)
            phases.extend(dict(row, mode=mode, trial=trial) for row in timer.rows())
    return dict(
        layer=layer,
        padding=padding,
        correctness=correctness,
        raw=raw,
        phases=phases,
        workspace_release_host=releases,
        aggregate=aggregate(raw),
        intermediate_events_excluded_from_pipeline_comparison=True,
        phase_intervals_include_dispatch_and_stream_idle_not_pure_kernel=True,
        study_workspaces_not_serving_peak_certification=True,
    )
