"""CPU-only prototype/residual source-format screen; not a serving backend.

Preserve per-row error relative to the existing three-bit source by charging
actual sparse/whole-row literals. No model weights, labels, or output fitting.
All arrays below own their storage; source input is never retained as a view.
"""

from dataclasses import dataclass

import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@dataclass(frozen=True)
class TableSource:
    shape: tuple[int, int]
    bits: int
    codes: torch.Tensor
    centers: torch.Tensor

    def tensors(self):
        return self.codes, self.centers


def serialize_codes(integer, bits):
    if bits not in (1, 2) or integer.numel() % 8:
        raise ValueError("one/two-bit full bytes required")
    if bool(((integer < 0) | (integer >= (1 << bits))).any()):
        raise ValueError("integer outside bit width")
    word = (integer.reshape(-1, 8).int() << (torch.arange(8) * bits)).sum(-1)
    return torch.stack([((word >> (8 * i)) & 255).byte() for i in range(bits)], -1).flatten()


def decode_table(body, base=None):
    n, width = body.shape
    byte = body.codes.reshape(-1, body.bits).long()
    word = sum(byte[:, i] << (8 * i) for i in range(body.bits))
    integer = (word[:, None] >> (torch.arange(8) * body.bits)) & ((1 << body.bits) - 1)
    integer = integer.reshape(width, n)
    group = torch.arange(n) // 128
    decoded = body.centers[:, group].gather(-1, integer[..., None]).squeeze(-1).T.contiguous()
    if base is not None:
        decoded = (base.float() + decoded.float()).bfloat16()
    return decoded


@torch.inference_mode()
def table_pack(values, valid, *, bits, base=None, original=None):
    if (
        values.ndim != 2
        or values.dtype != torch.bfloat16
        or values.device.type != "cpu"
        or values.numel() % 8
        or not bool(values.isfinite().all())
        or valid.shape != (values.shape[0],)
        or valid.dtype != torch.bool
        or bits not in (1, 2)
        or not bool(valid.any())
    ):
        raise ValueError("finite CPU BF16 source and nonempty original-row mask required")
    if (base is None) != (original is None) or (
        base is not None and (base.shape != values.shape or original.shape != values.shape)
    ):
        raise ValueError("source residual requires matching base and original")
    n, width = values.shape
    groups = (n + 127) // 128
    x = torch.zeros(width, groups * 128)
    x[:, :n] = values.T.float()
    x = x.reshape(width, groups, 128)
    mask = torch.zeros(groups * 128, dtype=torch.bool)
    mask[:n] = valid
    mask = mask.reshape(groups, 128)
    count = mask.sum(-1).clamp_min(1)
    mean = (x * mask).sum(-1) / count
    std = (((x - mean[..., None]).square() * mask).sum(-1) / count).sqrt()
    centers = mean[..., None] + std[..., None] * torch.linspace(-1.5, 1.5, 1 << bits)
    # Frozen four Lloyd steps. Padding and exact/text rows never affect means.
    for _ in range(4):
        code = (x[..., None] - centers[:, :, None]).abs().argmin(-1)
        updated = []
        for k in range(1 << bits):
            member = (code == k) & mask
            total = member.sum(-1)
            average = (x * member).sum(-1) / total.clamp_min(1)
            updated.append(torch.where(total > 0, average, centers[..., k]))
        centers = torch.stack(updated, -1).sort(-1).values
    centers = centers.bfloat16().contiguous()
    # Choose using stored BF16 centers and actual rounded source addition.
    palette = centers[:, torch.arange(n) // 128].float()
    target = values.T.float()
    if base is not None:
        palette = (palette + base.T[..., None].float()).bfloat16().float()
        target = original.T.float()
    code = (target[..., None] - palette).square().argmin(-1)
    code[:, ~valid] = 0
    return TableSource((n, width), bits, serialize_codes(code, bits), centers)


@dataclass(frozen=True)
class SourcePatch:
    row_pointer: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    row_map: torch.Tensor
    full_rows: torch.Tensor

    def tensors(self):
        return self.row_pointer, self.columns, self.values, self.row_map, self.full_rows


def apply_patch(decoded, patch):
    result = decoded.clone()
    if patch.columns.numel():
        rows = torch.repeat_interleave(
            torch.arange(len(result)), patch.row_pointer[1:] - patch.row_pointer[:-1]
        )
        result[rows, patch.columns.long()] = patch.values
    if patch.row_map.numel():
        selected = patch.row_map >= 0
        result[selected] = patch.full_rows[patch.row_map[selected].long()]
    return result


@torch.inference_mode()
def refine(original, decoded, baseline, needed):
    if (
        original.shape != decoded.shape
        or original.shape != baseline.shape
        or original.dtype != torch.bfloat16
        or decoded.dtype != torch.bfloat16
        or baseline.dtype != torch.bfloat16
        or original.device.type != "cpu"
        or original.ndim != 2
        or original.shape[1] > 32767
        or len(original) > 32767
    ):
        raise ValueError("compatible CPU BF16 arrays and int16 literal coordinates required")
    n, width = original.shape
    error = (original.double() - decoded.double()).square()
    limit = (original.double() - baseline.double()).square().sum(-1) + width * 1e-12
    excess = (error.sum(-1) - limit).clamp_min(0)
    excess[~needed] = 0
    order = error.argsort(dim=-1, descending=True, stable=True)
    cumulative = error.gather(1, order).cumsum(-1)
    counts = ((cumulative < excess[:, None]) & (excess[:, None] > 0)).sum(-1)
    counts = torch.where(excess > 0, counts + 1, 0).clamp_max(width)
    # Include a full-row index map if any row needs the cheaper exact-row form.
    whole = counts * 4 > width * 2
    scalar = (torch.arange(width)[None] < counts[:, None]) & ~whole[:, None]
    rows, rank = scalar.nonzero(as_tuple=True)
    columns = order[rows, rank]
    values = original[rows, columns].clone()
    ptr = torch.cat((torch.zeros(1, dtype=torch.int32), scalar.sum(-1).int().cumsum(0).int()))
    if not values.numel():
        ptr = torch.empty(0, dtype=torch.int32)
    mapping = torch.empty(0, dtype=torch.int16)
    full_rows = torch.empty(0, width, dtype=torch.bfloat16)
    if bool(whole.any()):
        mapping = torch.full((n,), -1, dtype=torch.int16)
        mapping[whole] = torch.arange(int(whole.sum()), dtype=torch.int16)
        full_rows = original[whole].clone().contiguous()
    patch = SourcePatch(ptr, columns.short().contiguous(), values, mapping, full_rows)
    reconstructed = apply_patch(decoded, patch)
    row_error = (reconstructed.double() - original.double()).square().mean(-1)
    old_error = (baseline.double() - original.double()).square().mean(-1)
    if bool((row_error[needed] > old_error[needed] + 1e-12).any()):
        raise RuntimeError("literal refinement failed per-row source-error contract")
    return patch, reconstructed


def spatial_assignment(text, *, length, patch_grid, group):
    if patch_grid != (24, 24) or group not in ((2, 2), (2, 4)):
        raise ValueError("audited one-image 24x24 raster and registered spatial group required")
    image = torch.ones(length, dtype=torch.bool)
    image[list(text)] = False
    positions = image.nonzero().flatten()
    if len(positions) != 576:
        raise ValueError("all original 576 image-patch coordinates required")
    raster = torch.arange(576)
    assignment = (raster // 24 // group[0]) * (24 // group[1]) + raster % 24 // group[1]
    mapping = torch.full((length,), -1, dtype=torch.int32)
    mapping[positions] = torch.arange(576, dtype=torch.int32)
    return positions, assignment.short().contiguous(), mapping


@torch.inference_mode()
def screen_refined(values, globally_exact, text, *, patch_grid=(24, 24)):
    if (
        values.device.type != "cpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 3
        or values.shape[0] != 1
        or values.shape[2] % 8
        or globally_exact.shape != (values.shape[1],)
        or globally_exact.dtype != torch.bool
        or not bool(values.isfinite().all())
    ):
        raise ValueError("finite original CPU BF16 source with exact logical-row mask required")
    original = values[0]
    n, _width = original.shape
    if any(type(t) is not int or not 0 <= t < n for t in text):
        raise ValueError("original text coordinates required")
    needed = ~globally_exact.clone()
    needed[list(text)] = False
    baseline = pack_unprotected(values[:, None], globally_exact[None], bits=3, axis=2)
    decoded_baseline = unpack_cpu(baseline)[0, 0]
    baseline_bytes = backing_bytes(baseline.tensors())
    old_error = (decoded_baseline.double() - original.double()).square().mean(-1)
    rows, candidates = [], {}
    for family in ("lut2", "spatial4_binary_residual", "spatial8_binary_residual"):
        extra, codeparts = [], []
        if family == "lut2":
            body = table_pack(original, needed, bits=2)
            decoded = decode_table(body)
        else:
            group = (2, 2) if family == "spatial4_binary_residual" else (2, 4)
            positions, assignment, mapping = spatial_assignment(
                text, length=n, patch_grid=patch_grid, group=group
            )
            source = original[positions].float()
            count = torch.bincount(assignment.long()).float()
            prototype = torch.zeros(len(count), original.shape[1])
            prototype.index_add_(0, assignment.long(), source)
            prototype = (prototype / count[:, None]).bfloat16()
            packed = pack(prototype[None, None], bits=4, axis=2)
            base = torch.zeros_like(original)
            base[positions] = unpack_cpu(packed)[0, 0][assignment.long()]
            residual = (original.float() - base.float()).bfloat16()
            body = table_pack(residual, needed, bits=1, base=base, original=original)
            decoded = decode_table(body, base=base)
            extra += [mapping, assignment, *packed.tensors()[1:]]
            codeparts.append(packed.codes)
        patch, reconstructed = refine(original, decoded, decoded_baseline, needed)
        codeparts.append(body.codes)
        codes = torch.cat(codeparts).contiguous()
        # Concatenation retires the separate code owners in the hypothetical
        # serialized payload. CPU fit temporaries are not a GPU peak estimate.
        family_tag = torch.tensor(
            [0 if family == "lut2" else 1 if family == "spatial4_binary_residual" else 2],
            dtype=torch.uint8,
        )
        owners = (codes, body.centers, family_tag, *extra, *patch.tensors())
        size = backing_bytes(owners)
        error = (reconstructed.double() - original.double()).square().mean(-1)
        passed = bool((error[needed] <= old_error[needed] + 1e-12).all())
        row = dict(
            family=family,
            raw_source_cpu_payload_bytes=baseline_bytes,
            candidate_cpu_payload_bytes=size,
            raw_cpu_payload_reduction=1 - size / baseline_bytes,
            baseline_source_mse=float(old_error[needed].mean()),
            candidate_source_mse=float(error[needed].mean()),
            per_row_source_screen_passed=passed,
            literal_scalar_count=patch.values.numel(),
            exact_literal_row_count=len(patch.full_rows),
            numerical_screen_passed=passed and size < baseline_bytes,
            downstream_quality_established=False,
            gpu_cache_backend_implemented=False,
        )
        rows.append(row)
        if row["numerical_screen_passed"]:
            candidates[family] = dict(codes=codes, other_bytes=size - codes.numel(), numerical=row)
    return dict(
        rows=rows,
        baseline=dict(codes=baseline.codes, other_bytes=baseline_bytes - baseline.codes.numel()),
        candidates=candidates,
    )
