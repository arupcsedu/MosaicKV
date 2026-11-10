#!/usr/bin/env python3
"""R07 initial backing-byte feasibility from the two consumed V7 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path(
    "/scratch/djy8hg/audits/fox-review-v7-source-analysis-recovery-20261110/source-prefix"
)
TOPK = "prompt_topk_raw_bf16_ceiling_local"
ATTENTION = "source3_attention_optional"
RANDOM = "source3_random_optional"
DIRECT = "direct_projected_kv3_attention_optional_local"
CHOICES = ("optional-0.00", "optional-0.03", "optional-0.05")
SHARED = {"cos", "sin", "slots", "union"}
MAX_FILE = 8 * 2**20
MAX_SPAN = 32 * 2**20


def integer(value, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid physical byte count or geometry")
    return value


def inventory(record, layer_count):
    """Split actual unique owners; reject unknown cross-layer allocation coupling."""
    local, shared, paths, categories = [0] * layer_count, {}, set(), Counter()
    if (
        record["includes_codes_scales_indices_rotary_exact_overlays_and_numerical_corrections"]
        is not True
        or record["CPU_residual_bytes"] != 0
        or record["generated_history"] != "not_yet_allocated_reported_separately_after_decode"
    ):
        raise ValueError("different representation accounting scope")
    for owner in record["components"]:
        count = integer(owner["bytes"])
        names = owner["paths"]
        if not names or len(set(names)) != len(names) or paths.intersection(names):
            raise ValueError("missing or multiply charged backing-owner path")
        paths.update(names)
        matches = [re.fullmatch(r"layer\[(\d+)\]\.(.+)", name) for name in names]
        if any(match is None for match in matches) or not owner["device"].startswith("cuda:"):
            raise ValueError("unclassified or non-GPU owner")
        layers = {int(match[1]) for match in matches}
        suffixes = {match[2] for match in matches}
        if max(layers) >= layer_count:
            raise ValueError("owner outside registered layer geometry")
        if len(layers) == 1:
            local[next(iter(layers))] += count
        else:
            key = next(iter(suffixes))
            if (
                len(suffixes) != 1
                or key not in SHARED
                or key in shared
                or layers != set(range(layer_count))
                or len(names) != layer_count
            ):
                raise ValueError("unmodeled cross-layer shared owner")
            shared[key] = (
                count,
                owner["dtype"],
                tuple(owner["observed_view_shape"]),
                owner["device"],
            )
        categories[owner["category"]] += count
    if (
        min(local) <= 0
        or sum(local) + sum(value[0] for value in shared.values())
        != record["representation_storage_bytes"]
        or dict(categories) != record["category_bytes"]
    ):
        raise ValueError("component accounting does not reconcile")
    return local, shared


def reachability(costs, fixed, targets):
    """Exact multiple-choice subset sums with one choice per layer."""
    if not costs or any(not row for row in costs):
        raise ValueError("one or more choices per layer required")
    fixed = integer(fixed, minimum=0)
    costs = [tuple(integer(value) for value in row) for row in costs]
    lower = [min(row) for row in costs]
    base = fixed + sum(lower)
    deltas = [tuple(value - low for value in row) for row, low in zip(costs, lower, strict=True)]
    span = sum(max(row) for row in deltas)
    if span > MAX_SPAN or len(costs) > 64:
        raise ValueError("reachability resource bound exceeded")
    states = [1]
    for row in deltas:
        new = 0
        for delta in row:
            new |= states[-1] << delta
        states.append(new)
    # Export a deterministic witness for each attainable registered lattice point.
    witnesses = {}
    for target in sorted(set(targets)):
        offset = integer(target) - base
        if offset < 0 or offset > span or not (states[-1] >> offset) & 1:
            continue
        choices = []
        for previous, row in zip(reversed(states[:-1]), reversed(deltas), strict=True):
            choice = next(
                i
                for i, delta in enumerate(row)
                if offset >= delta and (previous >> (offset - delta)) & 1
            )
            choices.append(choice)
            offset -= row[choice]
        choices.reverse()
        if offset or fixed + sum(row[i] for row, i in zip(costs, choices, strict=True)) != target:
            raise ValueError("reachability witness failed exact accounting")
        witnesses[target] = choices
    return witnesses


def topk_grid(row):
    evidence = row["physical_allocation"]
    selected = evidence["allocation"]
    observed = evidence["inventory"]
    logical = integer(row["original_logical_length"])
    count = integer(selected["count"])
    owners = observed["components"]
    layers = len(owners) // 2
    local, shared = inventory(observed, layers)
    if shared or len(owners) != 2 * layers or not 1 <= layers <= 64 or count > logical:
        raise ValueError("expected dense raw K/V pair in every layer")
    first_shape = owners[0]["observed_view_shape"]
    batch, heads, positions, width = map(integer, first_shape)
    per_layer_slot = 2 * heads * width * 2
    quantum = layers * per_layer_slot
    for layer, charged in enumerate(local):
        if charged != count * per_layer_slot:
            raise ValueError(f"raw Top-K backing storage disagrees at layer {layer}")
    if (
        batch != 1
        or positions != count
        or any(
            o["dtype"] != "torch.bfloat16"
            or o["observed_view_shape"] != first_shape
            or o["bytes"] != count * heads * width * 2
            for o in owners
        )
        or selected["physical_slot_bytes"] != quantum
        or selected["physical_padding_bytes"] != 0
        or selected["largest_uniform_dense_capacity_certified"] is not True
        or selected["actual_bytes"] != observed["representation_storage_bytes"]
    ):
        raise ValueError("Top-K geometry does not reproduce measured backing bytes")
    mandatory = integer(selected["minimum_mandatory_bytes"])
    cap = integer(selected["cap_bytes"])
    if mandatory % quantum or not mandatory <= count * quantum <= cap:
        raise ValueError("invalid mandatory/cap geometry")
    if count != min(logical, cap // quantum):
        raise ValueError("saved Top-K allocation did not fill its uniform geometry")
    # All genuine raw slots, not arbitrary padding; keep the original ceiling.
    return layers, quantum, range(mandatory, min(logical * quantum, cap) + 1, quantum)


def arm_grid(row, layers, targets):
    observed = row["physical_allocation"]["candidate_inventories"]
    if set(observed) != set(CHOICES):
        raise ValueError("unexpected measured protection ladder")
    plans = [inventory(observed[key], layers) for key in CHOICES]
    if any(shared != plans[0][1] for _, shared in plans):
        raise ValueError("shared owner layout changed across variants")
    fixed = sum(value[0] for value in plans[0][1].values())
    totals = {key: observed[key]["representation_storage_bytes"] for key in CHOICES}
    witnesses = reachability(list(zip(*(local for local, _ in plans), strict=True)), fixed, targets)
    return dict(
        measured_uniform_plan_bytes=totals,
        measured_uniform_plan_matches=sorted(set(totals.values()).intersection(targets)),
        shared_storage_bytes=fixed,
        minimum_layer_mix_bytes=fixed
        + sum(min(row) for row in zip(*(p[0] for p in plans), strict=True)),
        conditional_layer_mix_witnesses={
            str(target): [CHOICES[i] for i in choices] for target, choices in witnesses.items()
        },
    )


def read_bound(path, expected):
    if path.stat().st_size > MAX_FILE:
        raise ValueError(f"bounded allocation metadata read refused: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"immutable evidence changed: {path}")
    return raw


def derive(parent, gate_sha):
    gate_path = parent / "analysis/GATE_source_fixed_prefix.json"
    gate = json.loads(read_bound(gate_path, gate_sha))
    if gate["complete"] is not True or gate["completed_rows"] != 14:
        raise ValueError("two complete consumed inputs required")
    inputs, links = [], {str(gate_path): gate_sha}
    for index in range(2):
        path = parent / f"workers/{index}/raw.jsonl"
        expected = gate["bindings"][str(path)]
        links[str(path)] = expected
        manifest = path.with_name("manifest.json")
        manifest_sha = gate["bindings"][str(manifest)]
        info = json.loads(read_bound(manifest, manifest_sha))
        if (
            info["source_sha"] != gate["producer_source_sha"]
            or info["source_dirty"] is not False
            or info["model_precision"] != "bf16"
        ):
            raise ValueError("producer manifest is not the registered clean BF16 execution")
        links[str(manifest)] = manifest_sha
        # Only allocation/geometry are consulted. No targets, scores, token IDs,
        # tensors, or benchmark answer/reference files enter the calculation.
        rows = [json.loads(line) for line in read_bound(path, expected).splitlines()]
        wanted = {TOPK, ATTENTION, RANDOM, DIRECT}
        relevant = [row for row in rows if row["arm"] in wanted]
        if len(rows) != 7 or len(relevant) != 4 or {r["arm"] for r in relevant} != wanted:
            raise ValueError("incomplete or duplicated allocation evidence")
        if any(
            r["status"] != "completed"
            or r["source_sha"] != gate["producer_source_sha"]
            or r["labels_or_reference_outputs_used"] is not False
            for r in relevant
        ):
            raise ValueError("invalid allocation provenance or unavailable arm")
        by_arm = {r["arm"]: r for r in relevant}
        layers, quantum, targets = topk_grid(by_arm[TOPK])
        if len({r["original_logical_length"] for r in relevant}) != 1:
            raise ValueError("inconsistent source prompt geometry")
        inputs.append(
            dict(
                index=index,
                layers=layers,
                producer_manifest=str(manifest),
                uniform_raw_topk_quantum_bytes=quantum,
                original_ceiling_bytes=by_arm[TOPK]["physical_allocation"]["allocation"][
                    "cap_bytes"
                ],
                arms={
                    arm: arm_grid(by_arm[arm], layers, targets)
                    for arm in (ATTENTION, RANDOM, DIRECT)
                },
            )
        )
    common = {}
    for arm in (ATTENTION, RANDOM, DIRECT):
        common[arm] = sorted(
            int(v)
            for v in set.intersection(
                *(set(item["arms"][arm]["conditional_layer_mix_witnesses"]) for item in inputs)
            )
        )
    return dict(
        status="completed_initial_geometry_audit_not_new_measurement",
        result_category="unified_comparison",
        measurement_type="derived_saved_allocation_diagnostic",
        inputs=inputs,
        bindings=links,
        common_conditional_initial_targets_bytes=common,
        all_three_to_five_target_gate_passed=False,
        mixed_layer_constructed=False,
        shared_owner_assumption=(
            "mixed construction must preserve one identical shared owner per header"
        ),
        limitations=[
            "Conditional mixed plans are arithmetic only, not existing execution policies.",
            "No claim about final, live peak, quality, or latency.",
            "No enlargement of the original per-input ceiling or dummy allocations.",
        ],
        independent_quality_N=0,
        fresh_confirmation_authorized=False,
        superiority_established=False,
        new_GPU_inference=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--parent-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = clean_source(SOURCE)
    result = derive(args.parent, args.parent_gate_sha256)
    result.update(analysis_source_sha=revision, analysis_script_sha256=sha(Path(__file__)))
    write_once(args.output, result)
    print(
        json.dumps({k: result[k] for k in ("status", "common_conditional_initial_targets_bytes")})
    )


if __name__ == "__main__":
    main()
