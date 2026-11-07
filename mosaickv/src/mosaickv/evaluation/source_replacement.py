"""Lossless physical replacement of redundant shared source codes.

Only already-zero codes may be omitted. Original quantizer headers, logical
positions and decoded BF16 words remain unchanged. This common codec is NOT
a one-head prototype merger, and is available identically to all source users.
No old code tensor or decoded full source is retained by the returned body.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

_scope = ContextVar("evicastkv_source_code_replacement", default=None)


@contextmanager
def use(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("explicit source replacement flag required")
    report = dict(enabled=enabled, layers=[], benchmark_labels_accessed=False)
    token = _scope.set(report if enabled else None)
    try:
        yield report
    finally:
        _scope.reset(token)


def arguments(body):
    """No allocation: placeholder pointer is never dereferenced in dense mode."""
    mapped = body.row_map is not None
    return (body.row_map if mapped else body.codes), (body.stored_rows if mapped else 0), mapped


def validate(body):
    n, width = body.shape[2:]
    if (
        body.shape[:2] != (1, 1)
        or body.bits != 3
        or body.axis != 2
        or body.group != 128
        or width % 8
        or body.codebook is not None
        or body.codes.dtype != torch.uint8
        or not body.codes.is_contiguous()
    ):
        raise ValueError("three-bit contiguous full-width shared source required")
    if body.row_map is not None and (
        body.row_map.shape != (n,)
        or body.row_map.dtype != torch.int32
        or body.row_map.device != body.codes.device
        or not body.row_map.is_contiguous()
        or not 1 <= body.stored_rows <= n
        or body.codes.numel() != body.stored_rows * width * 3 // 8
    ):
        raise ValueError("charged original-to-physical source map required")


def _integers(codes):
    raw = codes.reshape(-1, 3).long()
    words = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    return ((words[:, None] >> (torch.arange(8) * 3)) & 7).flatten()


def _serialize(values):
    words = (values.reshape(-1, 8).long() << (torch.arange(8) * 3)).sum(-1)
    return torch.stack(
        tuple(((words >> (8 * i)) & 255).to(torch.uint8) for i in range(3)), -1
    ).flatten()


@torch.inference_mode()
def compact(body, omitted):
    """Construct a new smaller owner, or keep the body if metadata outweighs savings.

    An omitted row must have zero integer codes across EVERY input channel.
    Decoded values need not be zero: original per-position headers still apply.
    """
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    validate(body)
    n, width = body.shape[2:]
    if body.row_map is not None or omitted.shape != (n,) or omitted.dtype != torch.bool:
        raise ValueError("one original body and explicit logical-row mask required")
    # Host metadata is produced by the constructor's existing protection policy.
    omit = omitted.cpu()
    keep = (~omit).nonzero().flatten().to(torch.int32)
    physical = max(1, keep.numel())
    old_bytes = backing_bytes(body.tensors())
    expected = physical * width * 3 // 8 + n * 4
    if expected >= body.codes.numel():
        return body, dict(replaced=False, reason="no_net_saving", saved_bytes=0)
    mapping = torch.full((n,), -1, dtype=torch.int32)
    mapping[keep.long()] = torch.arange(keep.numel(), dtype=torch.int32)
    mapping = mapping.to(body.codes.device)
    if body.codes.device.type == "cuda":
        from mosaickv.evaluation.source_replacement_kernel import construct

        codes = construct(body, keep.to(body.codes.device), mapping, physical)
    else:
        values = _integers(body.codes).reshape(width, n)
        if bool(values[:, omit].any()):
            raise ValueError("cannot retire nonzero source codes")
        codes = _serialize(values[:, keep.long()] if keep.numel() else torch.zeros(width, 1))
    result = replace(body, codes=codes, row_map=mapping, stored_rows=physical)
    validate(result)
    if result.codes.untyped_storage().data_ptr() == body.codes.untyped_storage().data_ptr():
        raise RuntimeError("replacement aliases retired source payload")
    saved = old_bytes - backing_bytes(result.tensors())
    if saved <= 0:
        raise RuntimeError("replacement must save backing bytes including its map")
    return result, dict(
        replaced=True,
        omitted_rows=int(omit.sum()),
        original_rows=n,
        stored_rows=physical,
        original_source_bytes=old_bytes,
        replacement_source_bytes=backing_bytes(result.tensors()),
        row_map_bytes=n * 4,
        saved_bytes=saved,
        header_storage_unchanged=True,
        original_logical_positions_unchanged=True,
        original_code_owner_retained=False,
        lossless_zero_code_replacement=True,
    )


def maybe_compact(body, omitted):
    import weakref

    state = _scope.get()
    if state is None:
        return body
    result, record = compact(body, omitted)
    if record["replaced"]:
        state.setdefault("_old_owner_weakrefs", []).append(weakref.ref(body.codes))
    state["layers"].append(record)
    return result


def audit_retirement(report):
    """Call after prefill returns: require dead ORIGINAL tensor objects, not counters."""
    refs = report.pop("_old_owner_weakrefs", [])
    report["old_code_tensor_objects_alive"] = sum(ref() is not None for ref in refs)
    report["replaced_code_owners"] = len(refs)
    if report["old_code_tensor_objects_alive"]:
        raise RuntimeError("old dense source-code owner survived prefill retirement")
    report["saved_gpu_bytes"] = sum(row["saved_bytes"] for row in report["layers"])
    report["whole_request_peak_certified"] = False


def unpack_cpu(body):
    validate(body)
    if body.codes.device.type != "cpu":
        raise ValueError("CPU oracle only")
    n, width = body.shape[2:]
    rows = body.row_map.long()
    if bool(((rows < -1) | (rows >= body.stored_rows)).any()):
        raise ValueError("source map contains invalid physical indices")
    code = _integers(body.codes).reshape(width, body.stored_rows)[:, rows.clamp_min(0)]
    code[:, rows < 0] = 0
    scale = body.scale.reshape(width, -1).repeat_interleave(128, -1)[:, :n]
    minimum = body.minimum.reshape(width, -1).repeat_interleave(128, -1)[:, :n]
    return (code.float() * scale + minimum).transpose(0, 1).reshape(body.shape).to(torch.bfloat16)
