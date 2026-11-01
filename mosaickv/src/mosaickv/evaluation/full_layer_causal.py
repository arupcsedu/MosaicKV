"""Higher-memory full-layer causal controls; NOT cap-eligible compression."""

from contextlib import ExitStack

import torch

from mosaickv.evaluation.exact_storage import ExactFormat


class ProjectionInputCapture:
    """CPU diagnostic capture only. No model/cache mutation or retained GPU copy."""

    def __init__(self, adapter):
        self.adapter, self.stack, self.records = adapter, ExitStack(), {}

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):

                def capture(_module, args, kwargs, *, index=i):
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    if (
                        index in self.records
                        or x is None
                        or x.ndim != 3
                        or x.shape[0] != 1
                        or x.dtype != torch.bfloat16
                    ):
                        raise ValueError("one BF16 prefill projection input per layer required")
                    self.records[index] = x.detach().to(device="cpu", copy=True).contiguous()

                handle = block.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def restore(base, raw, layers):
    if (
        not base
        or len(base) != len(raw)
        or not layers
        or tuple(sorted(set(layers))) != tuple(layers)
        or any(type(i) is not int or not 0 <= i < len(base) for i in layers)
    ):
        raise ValueError("unique sorted full-layer interventions required")
    for old, full in zip(base, raw, strict=True):
        if (
            not isinstance(full, ExactFormat)
            or full.kind != "kv"
            or len(full.pair) != 2
            or any(not body.raw for body in full.pair)
            or old.shape[:2] != full.shape[:2]
            or old.shape[-1] != full.shape[-1]
            or old.shape[-2] > full.shape[-2]
        ):
            raise ValueError("original raw full-cache layer geometry required")
    return tuple(raw[i] if i in layers else packet for i, packet in enumerate(base))


def ordered(records):
    if not records or {r["layer"] for r in records} != set(range(len(records))):
        raise ValueError("complete unique per-layer causal evidence required")
    if any(set(r) != {"layer", "statistics"} for r in records):
        raise ValueError("only model-logit statistics may rank interventions")
    return sorted(
        records,
        key=lambda r: (
            not r["statistics"]["guard_pass"],
            r["statistics"]["token_mismatches"],
            r["statistics"]["maximum_KL_FP32"],
            r["statistics"]["maximum_logit_difference"],
            r["statistics"]["mean_KL_FP32"],
            r["layer"],
        ),
    )


def compositions(records):
    ranking = ordered(records)
    return tuple(
        tuple(sorted(r["layer"] for r in ranking[:n])) for n in (2, 4, 8) if n <= len(ranking)
    )
