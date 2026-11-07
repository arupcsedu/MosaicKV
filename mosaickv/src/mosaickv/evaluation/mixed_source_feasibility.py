"""CPU-only feature-group precision allocation; no serving/cache support.

Fixed 1/2/3/4-bit choices for groups of eight original source features. A
fractional LP is only an optimistic bound. Charge and verify the actual rounded
layout separately; source MSE does not establish downstream answer quality.
"""

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import eye, kron

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.refined_source_feasibility import decode_table, table_pack


def decode_mixed_cpu(shape, owners):
    """Decode only serialized owners, not original/quantizer/candidate arrays."""
    n, width = shape
    codes, headers, code_offset, header_offset, bits, tag = owners
    if int(tag.item()) != 3 or width // 8 != len(bits):
        raise ValueError("registered mixed-source CPU payload required")
    output = torch.empty(n, width, dtype=torch.bfloat16)
    groups = (n + 127) // 128
    for group in range(width // 8):
        b = int(bits[group])
        first, last = int(code_offset[group]), int(code_offset[group + 1])
        byte = codes[first:last].reshape(-1, b).long()
        word = sum(byte[:, i] << (8 * i) for i in range(b))
        integer = ((word[:, None] >> (torch.arange(8) * b)) & ((1 << b) - 1)).reshape(8, n)
        header = headers[int(header_offset[group]) : int(header_offset[group + 1])]
        if b <= 2:
            center = header.view(torch.bfloat16).reshape(8, groups, 1 << b)
            decoded = center[:, torch.arange(n) // 128].gather(-1, integer[..., None])[..., 0]
        else:
            minimum, scale = header.view(torch.float32).reshape(2, 8, groups)
            decoded = (
                integer.float() * scale[:, torch.arange(n) // 128]
                + minimum[:, torch.arange(n) // 128]
            ).bfloat16()
        output[:, group * 8 : (group + 1) * 8] = decoded.T
    return output


def bounded_allocate(errors, costs, limit, *, metadata_bytes, time_limit=20):
    """No labels; fixed fractional solve followed by <=128 monotone upgrades.

    Parent is choice index two (three bits); every route can always roll back
    to it. Only integral, per-row-feasible layouts can leave this function as
    candidate choices. LP output alone is never measured cache storage.
    """
    if errors.ndim != 3 or errors.shape[2] != 4 or costs.shape != errors.shape[1:]:
        raise ValueError("original rows x feature groups x four fixed bit choices required")
    rows, groups, _ = errors.shape
    if limit.shape != (rows,) or not np.all(np.isfinite(errors)):
        raise ValueError("finite original source-error bounds required")
    if np.any(errors < 0) or np.any(costs <= 0):
        raise ValueError("nonnegative source errors and positive charged costs required")
    if np.any(errors[:, :, 2].sum(1) > limit + 1e-12):
        raise ValueError("parent layout is not feasible under its own error bound")
    scale = np.maximum(limit, 1e-12)
    solution = linprog(
        costs.ravel() / costs[:, 2].sum(),
        A_ub=(errors / scale[:, None, None]).reshape(rows, -1),
        b_ub=limit / scale,
        A_eq=kron(eye(groups, format="csr"), np.ones((1, 4)), format="csr"),
        b_eq=np.ones(groups),
        bounds=(0, 1),
        method="highs",
        options={"time_limit": time_limit, "threads": 1, "random_seed": 20261107},
    )
    info = dict(
        lp_status=int(solution.status),
        lp_message=solution.message,
        fractional_source_payload_lower_bound=None,
        fractional_layout_is_measured_cache=False,
        monotone_upgrade_count=0,
        integral_feasible=False,
        rolled_back_to_parent=False,
    )
    parent = np.full(groups, 2, dtype=np.int64)
    if not solution.success:
        return parent, info
    info["fractional_source_payload_lower_bound"] = float(
        solution.fun * costs[:, 2].sum() + metadata_bytes
    )
    choice = solution.x.reshape(groups, 4).argmax(1)
    group = np.arange(groups)
    current = errors[:, group, choice].sum(1)
    for _ in range(128):
        if bool(np.all(current <= limit + 1e-12)):
            break
        weight = np.maximum((current - limit) / scale, 0)
        gain = errors[:, group, choice, None] - errors
        delta_cost = costs - costs[group, choice, None]
        score = np.einsum("r,rgc->gc", weight / scale, gain) / np.maximum(delta_cost, 1)
        allowed = np.arange(4)[None] > choice[:, None]
        score[~allowed] = -np.inf
        best = int(score.argmax())
        target_group, target_choice = np.unravel_index(best, score.shape)
        if score[target_group, target_choice] <= 0:
            break
        current += (
            errors[:, target_group, target_choice] - errors[:, target_group, choice[target_group]]
        )
        choice[target_group] = target_choice
        info["monotone_upgrade_count"] += 1
    if np.any(current > limit + 1e-12):
        choice = parent
        info["rolled_back_to_parent"] = True
    info["integral_feasible"] = True
    return choice, info


@torch.inference_mode()
def screen_mixed(values, globally_exact, text):
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
        raise ValueError("finite original CPU BF16 source and exact logical-row mask required")
    original = values[0]
    n, width = original.shape
    if any(type(t) is not int or not 0 <= t < n for t in text):
        raise ValueError("original text positions required")
    needed = ~globally_exact.clone()
    needed[list(text)] = False
    families = []
    for bits in (1, 2):
        body = table_pack(original, needed, bits=bits)
        families.append((body, decode_table(body)))
    for bits in (3, 4):
        body = pack_unprotected(values[:, None], globally_exact[None], bits=bits, axis=2)
        families.append((body, unpack_cpu(body)[0, 0]))
    baseline = families[2][0]
    baseline_bytes = backing_bytes(baseline.tensors())
    blocks, time_groups = width // 8, (n + 127) // 128
    errors = torch.stack(
        [
            (decoded.double() - original.double()).square()[needed].reshape(-1, blocks, 8).sum(-1)
            for _, decoded in families
        ],
        -1,
    ).numpy()
    costs = np.broadcast_to(
        np.array(
            [
                n + time_groups * 32,
                2 * n + time_groups * 64,
                3 * n + time_groups * 64,
                4 * n + time_groups * 64,
            ],
            dtype=np.float64,
        ),
        (blocks, 4),
    ).copy()
    limit = errors[:, :, 2].sum(1) + width * 1e-12
    metadata_bytes = blocks + 2 * (blocks + 1) * 4 + 1
    choice, info = bounded_allocate(errors, costs, limit, metadata_bytes=metadata_bytes)
    reconstructed = torch.empty_like(original)
    codes, headers, code_offsets, header_offsets = [], [], [0], [0]
    for group, index in enumerate(choice):
        first, last = group * 8, (group + 1) * 8
        body, decoded = families[index]
        bits = int(index) + 1
        reconstructed[:, first:last] = decoded[:, first:last]
        codes.append(body.codes[group * n * bits : (group + 1) * n * bits].clone())
        if bits <= 2:
            header = body.centers[first:last].contiguous().view(torch.uint8).flatten()
        else:
            header = torch.cat(
                [
                    t.reshape(width, time_groups)[first:last]
                    .contiguous()
                    .view(torch.uint8)
                    .flatten()
                    for t in (body.minimum, body.scale)
                ]
            )
        headers.append(header)
        code_offsets.append(code_offsets[-1] + len(codes[-1]))
        header_offsets.append(header_offsets[-1] + len(header))
    code = torch.cat(codes)
    padding = (-len(code)) % 128
    if padding:
        code = torch.cat((code, torch.zeros(padding, dtype=torch.uint8)))
    owners = (
        code,
        torch.cat(headers),
        torch.tensor(code_offsets, dtype=torch.int32),
        torch.tensor(header_offsets, dtype=torch.int32),
        torch.tensor(choice + 1, dtype=torch.uint8),
        torch.tensor([3], dtype=torch.uint8),
    )
    size = backing_bytes(owners)
    if not torch.equal(decode_mixed_cpu(original.shape, owners), reconstructed):
        raise RuntimeError(
            "serialized mixed-source CPU decoder changed a reconstructed source word"
        )
    before = (families[2][1].double() - original.double()).square().mean(-1)
    after = (reconstructed.double() - original.double()).square().mean(-1)
    passed = bool((after[needed] <= before[needed] + 1e-12).all())
    if not passed:
        raise RuntimeError("actual integral mixed source failed the per-row numerical contract")
    candidate = passed and size < baseline_bytes and info["lp_status"] == 0
    row = dict(
        family="mixed_group8_bits1234",
        **info,
        bit_group_counts={str(i): int(np.sum(choice + 1 == i)) for i in (1, 2, 3, 4)},
        raw_source_cpu_payload_bytes=baseline_bytes,
        candidate_cpu_payload_bytes=size,
        charged_code_padding_bytes=padding,
        serialized_cpu_roundtrip_bitwise=True,
        baseline_source_mse=float(before[needed].mean()),
        candidate_source_mse=float(after[needed].mean()),
        per_row_source_screen_passed=passed,
        numerical_screen_passed=candidate,
        raw_cpu_payload_reduction=1 - size / baseline_bytes,
        downstream_quality_established=False,
        gpu_cache_backend_implemented=False,
    )
    return dict(
        rows=[row],
        baseline=dict(codes=baseline.codes, other_bytes=baseline_bytes - baseline.codes.numel()),
        candidates={}
        if not candidate
        else {row["family"]: dict(codes=code, other_bytes=size - code.numel(), numerical=row)},
    )
