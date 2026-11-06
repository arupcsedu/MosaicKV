"""Reuse the native fitter's existing component, never retain a full reference.

Only the two consumed full-MHA shapes are enabled by the calling transaction.
This pure packet builder also admits tiny CPU fixtures. No labels/weights,
precision, membership or logical-position changes are permitted.
"""

from dataclasses import replace

import torch


class Capture:
    def __init__(self, base, text, modes):
        self.base, self.text, self.modes = base, tuple(text), tuple(modes)
        h, n = base.shape[1:3]
        if (
            not self.text
            or self.text != tuple(sorted(set(self.text)))
            or any(type(p) is not int or not 0 <= p < n for p in self.text)
            or len(self.modes) != 2
            or not (
                self.modes == ("source", "source")
                or all(m in ("resident", "encoded") for m in self.modes)
            )
        ):
            raise ValueError("original ordered text and frozen complete component modes required")
        self.flat = (
            None
            if self.modes == ("source", "source")
            else torch.tensor(
                [head * n + p for head in range(h) for p in self.text],
                device=base.cos.device,
                dtype=torch.int32,
            )
        )
        self.calls, self.bodies = [], []

    def capture(self, component, reference):
        if (
            component != len(self.calls)
            or component not in (0, 1)
            or tuple(reference.shape) != self.base.shape
        ):
            raise RuntimeError("one original native K then V component required")
        self.calls.append(component)
        if self.flat is not None:
            value = reference.reshape(1, 1, -1, 128).index_select(2, self.flat.long()).contiguous()
            if (
                value.untyped_storage().data_ptr() == reference.untyped_storage().data_ptr()
                or value.untyped_storage().nbytes() != value.numel() * 2
            ):
                raise RuntimeError("compact text must not keep a full native reference owner")
            self.bodies.append(value)

    def finish(self, candidate):
        from mosaickv.evaluation.cached_text_reader import CachedTextFormat
        from mosaickv.evaluation.exact_storage import encode
        from mosaickv.evaluation.lossless_bf16 import PackedBF16
        from mosaickv.evaluation.selective_text_construction import strip_exact

        if self.calls != [0, 1] or candidate.base is not self.base:
            raise RuntimeError("both actual native references and original packet required")
        if self.flat is None:
            return self.base, candidate
        if len(self.bodies) != 2:
            raise RuntimeError("both compact text components required")
        base = strip_exact(self.base, frozenset(self.text))
        if self.base.row_positions_cpu is None:
            base = replace(base, row_positions_cpu=None)
        bodies = tuple(
            encode(value) if mode == "encoded" else PackedBF16(tuple(value.shape), True, (value,))
            for value, mode in zip(self.bodies, self.modes, strict=True)
        )
        self.bodies.clear()
        return (
            CachedTextFormat(base, self.flat, bodies),
            CachedTextFormat(replace(candidate, base=base), self.flat, bodies),
        )


@torch.inference_mode()
def verify_outer(parent, native, attention, engine):
    from mosaickv.evaluation.lt_source_format import use
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    checks = 0
    with use(engine):
        for tail in (0, 1, 127):
            shape = (*parent.shape[:2], parent.shape[2] + tail, 128)
            for component in (0, 1):
                reference = torch.full(
                    shape,
                    -71.0,
                    device=next(iter(attention.parameters())).device,
                    dtype=torch.bfloat16,
                )
                actual = torch.full_like(reference, -71.0)
                fill_component(parent, attention, component, reference)
                fill_component(native, attention, component, actual)
                if not torch.equal(reference.view(torch.int16), actual.view(torch.int16)):
                    raise RuntimeError("async outer cache differs from parent words or padding")
                checks += 1
    return checks


def publish(cache, index, original, prepared):
    if (
        not 0 <= index < cache.next_layer
        or cache.layers[index].packet is not original
        or tuple(prepared.shape) != tuple(original.shape)
    ):
        raise RuntimeError("ordered same-cache sealed source required before text publication")
    cache.observe((*original.tensors(), *prepared.tensors()))
    cache.layers[index].packet = prepared
    cache.layer_owners_changed(index)
    cache.observe(original.tensors())
