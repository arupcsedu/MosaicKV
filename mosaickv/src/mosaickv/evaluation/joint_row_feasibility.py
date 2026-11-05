"""Evaluation-only test of folding incoming K/V rows into prompt projection.

No original operator/output is replaced. An unequal word rejects same-parent
integration. Never turn a small numerical error into a claimed exact pass.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch

WINDOWS = (1, 64, 126)
_probe = ContextVar("evicastkv_joint_row_probe", default=None)


def geometry(length, *, width=4096):
    if type(length) is not int or type(width) is not int or length < 1 or width != 4096:
        raise ValueError("positive original length and pinned 4096-wide geometry required")
    tail = length % 128
    if not tail:
        raise NotImplementedError("no spare row in the registered 128-row source workspace")
    return dict(first=length - tail, prompt_rows=tail, joint_rows=tail + 1, width=width)


def comparison(actual, expected):
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.dtype != torch.bfloat16
    ):
        raise ValueError("same-shape original BF16 operator outputs required")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise RuntimeError("nonfinite joint-row operator output")
    changed = int(
        (actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).sum()
    )
    return dict(
        compared_words=actual.numel(),
        changed_words=changed,
        bitwise_equal=changed == 0,
        maximum_absolute_difference=float((actual.float() - expected.float()).abs().max()),
    )


def admissible(rows):
    keys = [(r["step"], r["component"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {(s, c) for s in WINDOWS for c in (0, 1)}:
        raise ValueError("every registered prefix and K/V component required")
    return {
        backend: all(
            r[backend]["incoming_row"]["bitwise_equal"]
            and r[backend]["prompt_rows"]["bitwise_equal"]
            for r in rows
        )
        for backend in ("torch_joint", "Lt_joint")
    }


class Probe:
    def __init__(self, adapter):
        self.adapter, self.index, self.steps = adapter, -1, 0
        self.source = self.output = self.plans = self.layout = None
        self.rows = []
        self.attention = adapter._language_layers()[0].self_attn

    def capture(self, packets):
        from mosaickv.evaluation.head_local_precision import HeadLocalFormat
        from mosaickv.evaluation.lt_source_chunks import decode_slice
        from mosaickv.evaluation.shared_source_format import SharedSourceFormat
        from mosaickv.evaluation.zero_workspace_lt import Plans

        if self.source is not None or self.index != -1:
            raise RuntimeError("one unchanged original prefill and source capture required")
        packet = packets[0]
        while isinstance(packet, HeadLocalFormat):
            packet = packet.base
        if not isinstance(packet, SharedSourceFormat) or packet.shape[1] * packet.shape[3] != 4096:
            raise NotImplementedError("actual pinned first-layer shared-source packet required")
        self.layout = geometry(packet.shape[2])
        self.source = torch.empty(128, 4096, device=packet.cos.device, dtype=torch.bfloat16)
        self.output = torch.empty_like(self.source)
        decode_slice(packet, self.source, self.layout["first"], self.layout["prompt_rows"])
        self.plans = Plans((self.layout["prompt_rows"], self.layout["joint_rows"]), 4096)

    def observe(self, component, module, hidden, original):
        if self.index not in WINDOWS or tuple(hidden.shape) != (1, 1, 4096):
            return  # Ignore prefill/constructor projections and multi-token history fits.
        if (
            self.source is None
            or module.bias is not None
            or hidden.dtype != torch.bfloat16
            or original.shape != hidden.shape
            or hidden.device.type != "cuda"
            or module.weight.shape != (4096, 4096)
        ):
            raise RuntimeError("captured original one-token bias-free BF16 projection required")
        n = self.layout["prompt_rows"]
        self.source[n].copy_(hidden.reshape(-1))
        joined = torch.nn.functional.linear(self.source[: n + 1], module.weight)
        separate = torch.nn.functional.linear(self.source[:n], module.weight)
        torch_report = dict(
            incoming_row=comparison(joined[n], original.reshape(-1)),
            prompt_rows=comparison(joined[:n], separate),
        )
        del joined, separate
        self.plans.execute(self.source, module.weight, self.output, n + 1)
        incoming = comparison(self.output[n], original.reshape(-1))
        prompt = self.output[:n].clone()
        self.plans.execute(self.source, module.weight, self.output, n)
        lt_report = dict(incoming_row=incoming, prompt_rows=comparison(prompt, self.output[:n]))
        self.rows.append(
            dict(step=self.index, component=component, torch_joint=torch_report, Lt_joint=lt_report)
        )

    @contextmanager
    def installed(self):
        if _probe.get() is not None:
            raise RuntimeError("request-private nonnested projection feasibility required")
        token = _probe.set(self)
        original_step = self.adapter.decode_one_token

        def step(*args, **kwargs):
            self.index = self.steps
            self.steps += 1
            return original_step(*args, **kwargs)

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.adapter, "decode_one_token", step))
                for component, module in enumerate((self.attention.k_proj, self.attention.v_proj)):

                    def hook(module, args, result, *, component=component):
                        self.observe(component, module, args[0], result)

                    handle = module.register_forward_hook(hook)
                    stack.callback(handle.remove)
                yield self
        finally:
            try:
                if self.plans is not None:
                    torch.cuda.synchronize()
                    self.plans.close()
            finally:
                _probe.reset(token)


def current():
    return _probe.get()


def collect(adapter, call, folder):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    with Probe(adapter).installed() as state:
        measured = call()
        if (
            state.steps != 127
            or not measured["logits_bitwise"]
            or not measured["no_FullKV_fallback"]
        ):
            raise RuntimeError("unchanged complete 128-vector parent trajectory required")
        eligible = admissible(state.rows)
        return dict(
            comparisons=state.rows,
            layout=state.layout,
            exact_feasibility_by_backend=eligible,
            source_model_unchanged=True,
            parent_logits_bitwise=True,
            no_FullKV_fallback=True,
            explicit_diagnostic_fixed_workspace_bytes=backing_bytes((state.source, state.output)),
            temporary_linear_outputs_NOT_whole_GPU_peak_evidence=True,
            whole_request_GPU_peak_certified=False,
            native_ABI=state.plans.record(),
            torch_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            candidate_forward_not_integrated=True,
            no_quality_or_latency_claim=True,
            stopping_rule="any operator word difference forbids same-parent integration",
            independent_quality_N=0,
        )
