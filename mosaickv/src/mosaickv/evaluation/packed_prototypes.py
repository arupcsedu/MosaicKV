"""Bounded direct packing of the EXISTING source-prototype transaction.

Centroids, residuals and logical maps match the reference constructor exactly.
Only selected members are unpacked. The GPU serializer is a single bounded
Triton pass; the CPU oracle uses fixed-size word tiles, never a full INT32 source.
"""

import weakref
from dataclasses import replace

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_head_prototypes import SharedHeadReplacement, serialize
from mosaickv.evaluation.source_replacement import validate


def selected_codes(codes, rows, width, members):
    indices = torch.tensor(members, device=codes.device, dtype=torch.long)[:, None]
    flat = torch.arange(width, device=codes.device)[None, :] * rows + indices
    word = flat // 8 * 3
    value = codes[word].int() | (codes[word + 1].int() << 8) | (codes[word + 2].int() << 16)
    return (value >> ((flat % 8) * 3)) & 7


def copy_packed(codes, retained, anchors, centers, *, rows, width, tile_words=4096):
    stored = retained.numel()
    output = torch.empty(stored * width * 3 // 8, dtype=torch.uint8, device=codes.device)
    if codes.is_cuda:
        from mosaickv.evaluation.packed_prototype_kernel import copy

        copy[((stored * width // 8 + 127) // 128,)](
            codes,
            retained,
            anchors,
            centers,
            output,
            rows,
            stored,
            width,
            WORDS=128,
        )
        return output
    # Same integer serializer as CUDA, with a hard workspace tile bound.
    for begin in range(0, stored * width // 8, tile_words):
        end = min(begin + tile_words, stored * width // 8)
        flat = torch.arange(begin * 8, end * 8).reshape(-1, 8)
        channel, slot = flat // stored, flat % stored
        original = channel * rows + retained[slot]
        word = original // 8 * 3
        value = codes[word].int() | (codes[word + 1].int() << 8) | (codes[word + 2].int() << 16)
        value = (value >> ((original % 8) * 3)) & 7
        group = anchors[slot]
        value = torch.where(group >= 0, centers[group.clamp_min(0), channel], value)
        word = (value << (torch.arange(8) * 3)).sum(-1, dtype=torch.int32)
        packed = torch.stack(tuple(((word >> (8 * i)) & 255).byte() for i in range(3)), -1)
        output[begin * 3 : end * 3] = packed.flatten()
    return output


@torch.inference_mode()
def construct(body, groups, *, heads, mandatory=(), retention_one=False):
    validate(body)
    if type(heads) is not int or heads < 1 or body.shape[-1] % heads:
        raise ValueError("complete divisible K/V head coverage required")
    if type(retention_one) is not bool:
        raise ValueError("explicit retention-one control required")
    if retention_one:
        if groups:
            raise ValueError("retention one forbids prototype transformations")
        return body
    if body.row_map is not None:
        raise ValueError("one original source owner required; no nested replacements")
    n, width = body.shape[2:]
    groups = tuple(tuple(g) for g in groups)
    members = tuple(i for g in groups for i in g)
    if (
        not groups
        or any(len(g) < 2 or tuple(sorted(set(g))) != g for g in groups)
        or any(type(i) is not int or not 0 <= i < n for i in (*members, *mandatory))
        or len(set(members)) != len(members)
        or set(members) & set(mandatory)
        or any(g[0] // body.group != g[-1] // body.group for g in groups)
    ):
        raise ValueError("disjoint sorted nonmandatory groups in the same quantizer group required")
    values = selected_codes(body.codes, n, width, members)
    centers, offset, changed = [], 0, torch.zeros((), device=values.device, dtype=torch.long)
    for group in groups:
        part = values[offset : offset + len(group)]
        center = part.float().mean(0).round().int()
        centers.append(center)
        changed += (part != center).sum()
        offset += len(group)
    centers = torch.stack(centers)
    removed = {i for g in groups for i in g[1:]}
    kept = tuple(i for i in range(n) if i not in removed)
    location = {p: i for i, p in enumerate(kept)}
    mapping = [location.get(i, 0) for i in range(n)]
    anchors = [-1] * len(kept)
    for index, group in enumerate(groups):
        anchors[location[group[0]]] = index
        for member in group:
            mapping[member] = location[group[0]]
    retained = torch.tensor(kept, device=values.device, dtype=torch.long)
    anchors = torch.tensor(anchors, device=values.device, dtype=torch.long)
    codes = copy_packed(body.codes, retained, anchors, centers, rows=n, width=width)
    residual = serialize(values).to(device="cpu", copy=True).contiguous()
    if body.codes.is_cuda:
        residual = residual.pin_memory()
    result = replace(
        body,
        codes=codes,
        row_map=torch.tensor(mapping, device=codes.device, dtype=torch.int32),
        stored_rows=len(kept),
        row_map_shared=False,
    )
    validate(result)
    return SharedHeadReplacement(
        result,
        residual,
        members,
        groups,
        weakref.ref(body.codes),
        heads,
        backing_bytes(body.tensors()),
        int(changed),
    )
