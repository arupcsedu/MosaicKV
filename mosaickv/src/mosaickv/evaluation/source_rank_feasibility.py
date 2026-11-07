"""CPU-only numerical source-factor feasibility, never a model/cache backend.

Use the best unquantized rank-r source approximation as an optimistic screen,
then test actual BF16 coefficients and three-bit basis storage when it passes.
No weights, gradients, labels, generated answers or future-query oracles.
"""

import numpy as np
import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.lossless_code_delta import encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@torch.inference_mode()
def screen(values, globally_exact, text, ranks=(96, 128)):
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
    n, _width = values.shape[1:]
    needed = ~globally_exact.clone()
    if any(type(i) is not int or not 0 <= i < n for i in text):
        raise ValueError("original source text coordinates required")
    needed[list(text)] = False
    indices = needed.nonzero().flatten()
    if not ranks or any(type(r) is not int or not 1 <= r <= len(indices) for r in ranks):
        raise ValueError("registered positive ranks within non-exact image-row count required")
    # Exactly the existing source quantizer; globally exact codes are zeroed,
    # but their decoded minima are NOT assumed to be zero source values.
    baseline = pack_unprotected(values[:, None], globally_exact[None], bits=3, axis=2)
    decoded = unpack_cpu(baseline)[0, 0]
    original = values[0].index_select(0, indices).double()
    error = original - decoded.index_select(0, indices).double()
    baseline_mse = float(error.square().mean())
    gram = original @ original.T
    eigenvalue, eigenvector = torch.linalg.eigh(gram)
    eigenvalue = eigenvalue.clamp_min(0)
    if not torch.isclose(eigenvalue.sum(), original.square().sum(), rtol=1e-9, atol=1e-9):
        raise RuntimeError("source spectrum does not conserve original energy")
    rows, candidates = [], {}
    for rank in ranks:
        optimistic = float(eigenvalue[:-rank].sum() / original.numel())
        row = dict(
            rank=rank,
            original_rows=n,
            approximated_image_rows=len(indices),
            raw_source_cpu_payload_bytes=backing_bytes(baseline.tensors()),
            baseline_source_mse=baseline_mse,
            optimal_unquantized_rank_mse=optimistic,
            optimistic_mse_ratio=optimistic / max(baseline_mse, 1e-30),
            optimistic_screen_passed=optimistic <= baseline_mse + 1e-12,
            actual_factor_source_mse=None,
            factor_cpu_payload_bytes=None,
            factor_numerical_screen_passed=False,
            downstream_quality_established=False,
        )
        if row["optimistic_screen_passed"]:
            singular = eigenvalue[-rank:].sqrt().clamp_min(1e-30)
            left = eigenvector[:, -rank:]
            pivot = left.abs().argmax(0)
            sign = torch.where(left[pivot, torch.arange(rank)] >= 0, 1.0, -1.0)
            left = left * sign
            coefficient = (left * singular).bfloat16().contiguous()
            basis = ((left.T @ original) / singular[:, None]).bfloat16().contiguous()
            body = pack(basis[None, None], bits=3, axis=2)
            actual = (coefficient.float() @ unpack_cpu(body)[0, 0].float()).bfloat16()
            mse = float((actual.double() - original).square().mean())
            mapping = torch.full((n,), -1, dtype=torch.int32)
            mapping[indices] = torch.arange(len(indices), dtype=torch.int32)
            tensors = (*body.tensors(), coefficient, mapping)
            size = backing_bytes(tensors)
            passed = mse <= baseline_mse + 1e-12 and size < row["raw_source_cpu_payload_bytes"]
            row.update(
                actual_factor_source_mse=mse,
                factor_cpu_payload_bytes=size,
                factor_numerical_screen_passed=passed,
            )
            if passed:
                candidates[rank] = dict(
                    codes=body.codes, other_bytes=size - body.codes.numel(), numerical=row
                )
        rows.append(row)
    return dict(
        rows=rows,
        baseline=dict(
            codes=baseline.codes,
            other_bytes=backing_bytes(baseline.tensors()) - baseline.codes.numel(),
        ),
        candidates=candidates,
    )


def cpu_serialized_storage(bodies):
    """Same byte-stream codec, including real offsets and one replay buffer.

    CPU payload accounting only. Unchanged exact/text/phase metadata, GPU
    reconstruction workspaces, history and whole-request peaks are OUTSIDE.
    """
    previous, original, encoded, scratch = None, 0, 0, 0
    for body in bodies:
        codes = body["codes"].contiguous().numpy()
        packet = encode(previous, codes)
        original += codes.nbytes + body["other_bytes"]
        encoded += packet.serialized_bytes + body["other_bytes"]
        if not packet.raw:
            scratch = max(scratch, len(codes))
        previous = np.array(codes, copy=True)
    return dict(
        original_cpu_payload_bytes=original,
        encoded_cpu_payload_bytes=encoded,
        charged_replay_bytes=scratch,
        net_cpu_payload_bytes=min(original, encoded + scratch),
        serialization_accepted=encoded + scratch < original,
        gpu_kv_storage_measured=False,
        whole_request_peak_certified=False,
    )


def choose_codec_coupled(alternatives):
    """Exact finite CPU format DP, charging inter-layer codec and replay scratch.

    Each layer is a stable list of (name, payload), parent first. Payloads have
    already passed their numerical screen. Delta cost depends on the preceding
    code bytes, so independently minimizing each layer's raw bytes is unsafe.
    Include whole-stream raw fallback. Ties prefer earlier (parent) choices.
    """
    if not alternatives or any(not layer for layer in alternatives):
        raise ValueError("at least one feasible parent choice per layer required")
    states = {(None, 0): (0, ())}
    previous_layer = None
    raw_path, raw_total = [], 0
    for layer in alternatives:
        raw = min(
            range(len(layer)),
            key=lambda i: (layer[i][1]["codes"].numel() + layer[i][1]["other_bytes"], i),
        )
        raw_path.append(raw)
        raw_total += layer[raw][1]["codes"].numel() + layer[raw][1]["other_bytes"]
        next_states, packet_cache = {}, {}
        for (old_index, scratch), (cost, path) in states.items():
            for index, (_name, body) in enumerate(layer):
                key = old_index, index
                if key not in packet_cache:
                    old = (
                        None
                        if previous_layer is None
                        else previous_layer[old_index][1]["codes"].contiguous().numpy()
                    )
                    packet_cache[key] = encode(old, body["codes"].contiguous().numpy())
                packet = packet_cache[key]
                capacity = max(scratch, 0 if packet.raw else packet.length)
                proposal = cost + packet.serialized_bytes + body["other_bytes"], (*path, index)
                state_key = index, capacity
                if state_key not in next_states or proposal < next_states[state_key]:
                    next_states[state_key] = proposal
        previous_layer, states = layer, next_states
    best = min((cost + scratch, path) for (_last, scratch), (cost, path) in states.items())
    total, path = min(best, (raw_total, tuple(raw_path)))
    chosen = [layer[index][1] for layer, index in zip(alternatives, path, strict=True)]
    measured = cpu_serialized_storage(chosen)
    if measured["net_cpu_payload_bytes"] != total:
        raise RuntimeError("codec-coupled dynamic program disagrees with serialized payload")
    return chosen, [layer[index][0] for layer, index in zip(alternatives, path, strict=True)]
