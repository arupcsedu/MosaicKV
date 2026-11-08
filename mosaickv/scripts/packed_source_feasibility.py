"""CPU source-archive cost screen. No labels, model load or GPU allocations."""

import json
from pathlib import Path

import numpy as np
import torch

from mosaickv.evaluation.all_head_source_codec import pack_symbols, symbols
from mosaickv.evaluation.compact_rare_source import encode
from mosaickv.evaluation.fox_campaign import sha, write_once

ARCHIVES = Path("/scratch/djy8hg/runs/evicastkv/complete-owner-operator-20261107-v2")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/shared-prototype-ledger-20261108-v6")


def encoded_cost(codes):
    previous, total, replay = None, 0, 0
    for array in codes:
        packet = encode(previous, array)
        total += packet.serialized_bytes
        if not packet.raw:
            replay = max(replay, len(array))
        previous = array
    return total + replay


def run(output):
    result = dict(
        measurement_type="derived",
        CPU_only=True,
        benchmark_labels_accessed=False,
        independent_quality_established=False,
        larger_campaign_authorized=False,
        cases=[],
        archives={},
        full_candidate_cost_requires_actual_transport_fitting=True,
    )
    for i in (0, 1):
        path = ARCHIVES / f"workers/{i}/source_codes.pt"
        result["archives"][str(path)] = sha(path)
        capture = json.loads((ARCHIVES / f"workers/{i}/capture.json").read_text())
        if capture["code_archive_sha"] != sha(path):
            raise ValueError("immutable consumed source archive changed")
        tensors = torch.load(path, weights_only=True, map_location="cpu")
        if sorted(tensors) != list(range(32)):
            raise ValueError("complete 32-layer source archive required")
        width = 4096
        n = tensors[0].numel() * 8 // (width * 3)
        matrices = [symbols(tensors[k].numpy()).reshape(width, n).T for k in range(32)]
        candidates = [
            tuple(range(start, start + 8))
            for start in range(0, n - 7, 8)
            if start // 128 == (start + 7) // 128
            and all(
                not np.any(np.all(matrix[list(range(start, start + 8))] == 0, axis=1))
                for matrix in matrices
            )
        ]
        # Bytes-only grouping proxy, NOT a prompt/output fidelity predictor.
        risks = [
            sum(
                float(
                    np.square(
                        matrix[list(g)].astype(np.float32) - np.rint(matrix[list(g)].mean(0))
                    ).mean()
                )
                for matrix in matrices
            )
            for g in candidates
        ]
        ordered = [
            candidates[j]
            for j in sorted(range(len(candidates)), key=lambda j: (risks[j], candidates[j]))
        ]
        base_codes = [tensors[k].numpy() for k in range(32)]
        base_cost = encoded_cost(base_codes)
        control = json.loads((PARENT / f"workers/{i}/raw.jsonl").read_text().splitlines()[0])
        fixed_remainder = control["detail"]["initial_encoded_bytes"] - base_cost
        if fixed_remainder < 0:
            raise ValueError("CPU codec cannot reconcile frozen whole-cache remainder")
        for count in (1, 2, 4, 8):
            groups = tuple(ordered[:count])
            remove = {pos for group in groups for pos in group[1:]}
            keep = [p for p in range(n) if p not in remove]
            codes = []
            for matrix in matrices:
                changed = matrix.copy()
                for group in groups:
                    changed[group[0]] = np.rint(
                        matrix[list(group)].astype(np.float32).mean(0)
                    ).astype(np.uint8)
                codes.append(pack_symbols(changed[keep].T.ravel()))
            map_bytes, residual = n * 4, 32 * count * 8 * width * 3 // 8
            candidate = encoded_cost(codes) + map_bytes + fixed_remainder
            gpu_saved = control["detail"]["initial_encoded_bytes"] - candidate
            result["cases"].append(
                dict(
                    input=i,
                    groups=[list(g) for g in groups],
                    group_count=count,
                    source_rows=n,
                    unchanged_measured_noncode_remainder_bytes=fixed_remainder,
                    encoded_GPU_bytes_at_frozen_noncode_inventory=candidate,
                    CPU_residual_bytes=residual,
                    GPU_saving_bytes=gpu_saved,
                    GPU_plus_CPU_saving_at_frozen_inventory_bytes=gpu_saved - residual,
                    independently_measured_candidate_transport_patch_change=False,
                    complete_model_candidate_cost_certified=False,
                    eligible_for_bounded_model_screen=gpu_saved > residual,
                    grouping="shared_across_32_layers_minimum_code_variance_not_a_quality_claim",
                )
            )
    write_once(output, result)
    return result
