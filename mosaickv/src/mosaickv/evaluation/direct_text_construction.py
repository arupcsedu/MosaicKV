"""Construct the SAME frozen cached-text representation without discarded fits.

This is an explicitly scoped execution variant. Use actual protected original
or merged K/V rows, never labels or new values. Default construction delegates
to the unchanged validated text-source constructor. No process-global state.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from mosaickv.evaluation.cached_text_reader import CachedTextFormat
from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.text_source_storage import strip_text_overlays, text_slots

_state = ContextVar("evicastkv_direct_cached_text", default=None)


@torch.inference_mode()
def build(packet, pair, positions, text, modes):
    if (
        not isinstance(packet, (HeadLocalFormat, SharedSourceFormat))
        or type(modes) is not tuple
        or len(modes) != 2
        or any(m not in ("encoded", "resident") for m in modes)
    ):
        raise ValueError("audited protected parent and two lossless text modes required")
    h, n, d = packet.shape[1:]
    if (
        d != 128
        or len(pair) != 2
        or any(
            tuple(t.shape) != packet.shape
            or t.dtype != torch.bfloat16
            or t.device != pair[0].device
            for t in pair
        )
    ):
        raise ValueError("actual selected/merged BF16 pair with original shape required")
    slots = text_slots(positions, tuple(text))
    flat = torch.tensor(
        [head * n + slot for head, row in enumerate(slots) for slot in row if slot >= 0],
        device=pair[0].device,
        dtype=torch.int32,
    )
    if not flat.numel() or len(positions) != h:
        raise ValueError("nonempty original head-local text memberships required")
    # Borrow raw values only where the parent ALREADY guarantees exact rows.
    # This check prevents an apparent speedup from changing the parent values.
    if isinstance(packet, SharedSourceFormat):
        protected = packet.row_positions
    else:
        common = packet.base.exact_indices.to(torch.int32)
        protected = torch.cat(
            (
                (
                    torch.arange(h, device=flat.device, dtype=torch.int32)[:, None] * n
                    + common[None]
                ).flatten(),
                packet.flat_tokens,
            )
        )
    if not bool(torch.isin(flat, protected).all()):
        raise ValueError("every borrowed text row must already be parent-exact")
    base = strip_text_overlays(packet, pair, positions, frozenset(text))
    bodies = []
    for tensor, mode in zip(pair, modes, strict=True):
        rows = tensor.reshape(1, 1, h * n, d).index_select(2, flat.long()).contiguous()
        bodies.append(
            encode(rows) if mode == "encoded" else PackedBF16(tuple(rows.shape), True, (rows,))
        )
    return CachedTextFormat(base, flat, tuple(bodies))


class Construction:
    def __init__(self, policy):
        if (
            set(policy)
            != {"modes", "maximum_added_bytes", "base_private_bytes", "expected_private_bytes"}
            or not policy["modes"]
            or any(
                type(policy[k]) is not int or policy[k] < 0
                for k in ("maximum_added_bytes", "base_private_bytes", "expected_private_bytes")
            )
            or policy["maximum_added_bytes"] > 6 * 2**20
            or policy["expected_private_bytes"]
            > policy["base_private_bytes"] + policy["maximum_added_bytes"]
        ):
            raise ValueError(
                "complete bounded frozen direct-text policy required; labels forbidden"
            )
        self.modes = tuple(tuple(row) for row in policy["modes"])
        if any(
            len(row) != 2
            or not (row == ("source", "source") or all(m in ("encoded", "resident") for m in row))
            for row in self.modes
        ):
            raise ValueError("one registered source/cached mode per layer required")
        self.policy, self.count, self.cached = dict(policy), 0, []

    def optimize(self, packet, positions, text, pair):
        if self.count >= len(self.modes):
            raise RuntimeError("more construction calls than frozen layers")
        index = self.count
        self.count += 1
        modes = self.modes[index]
        if modes == ("source", "source"):
            return packet, dict(direct_text_noop=True)
        value = build(packet, pair, positions, text, modes)
        self.cached.append(index)
        return value, dict(direct_text_constructed=True)

    def finish(self, packets):
        if self.count != len(self.modes) or len(packets) != self.count:
            raise RuntimeError("exactly one construction call per frozen layer required")
        size = backing_bytes(tuple(t for p in packets for t in p.private()))
        if size != self.policy["expected_private_bytes"]:
            raise RuntimeError(
                "direct packet storage differs from frozen cached-text representation"
            )
        return dict(
            private_bytes=size,
            cached_layers=self.cached,
            bypassed_text_fits=self.count,
            same_frozen_representation_required=True,
            no_labels_or_answers_used=True,
        )


@contextmanager
def use(policy):
    if _state.get() is not None:
        raise RuntimeError("direct text construction state must be isolated and nonnested")
    state = Construction(policy) if policy is not None else None
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


def optimize(packet, record, positions, text, attention, *, reference_pair):
    state = _state.get()
    if state is None:
        from mosaickv.evaluation.text_source_storage import optimize as original

        return original(packet, record, positions, text, attention)
    return state.optimize(packet, positions, text, reference_pair)
