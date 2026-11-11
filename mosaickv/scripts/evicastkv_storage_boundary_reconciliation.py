#!/usr/bin/env python3
"""Bounded CPU reconciliation of eight existing V11 construction observations.

Endpoint differences are arithmetic, not an inferred allocation classification.
The old producer did not archive a per-owner peak-time inventory. Missing
categories therefore stay unknown; this tool cannot certify a unified comparison.
No tensors, benchmark answers, new identities, CUDA or models are loaded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v11/run-20261110-1/bytes")
PRODUCER_SHA = "1aab16c63eee2f063194e6db8c9b3b5af466c3a1"
PARENT_GATE_SHA = "34d6a587fa8a3a7b165633377defd616dea61761ce478f254966f983106cd3ab"
TARGETS = (47710208, 51380224)
ARMS = ("prompt_attention_topk", "mixed_source_protection_v1")
GATE = "GATE_storage_boundary_reconciliation.json"
MAX_JSON_BYTES = 8 * 1024 * 1024
SOURCE_SCOPES = {
    "mosaickv/scripts/evicastkv_construct_shared_byte.py": ("worker", "analyze"),
    "mosaickv/src/mosaickv/evaluation/lossless_prompt_cache.py": (
        "PromptStorageCache.observe",
        "PromptStorageCache.storage_record",
    ),
    "mosaickv/src/mosaickv/evaluation/exact_storage.py": ("ExactCache.storage_record",),
    "mosaickv/src/mosaickv/evaluation/source_history.py": ("SourceHistoryCache.storage_record",),
    "mosaickv/src/mosaickv/evaluation/graphed_storage.py": ("SharedSourceCache.storage_record",),
}
POLICY = dict(
    schema_version=1,
    cases=2,
    targets_initial_bytes=list(TARGETS),
    methods=list(ARMS),
    rows=8,
    pairs=4,
    source="existing_V11_constructed_rows_only",
    no_model_or_tensor_loading=True,
    no_benchmark_labels=True,
    no_new_identity_or_GPU=True,
    endpoint_difference_is_not_component_attribution=True,
    gross_layer_component_sum_is_not_unique_owner_storage=True,
    missing_peak_lifetimes_remain_unknown=True,
)


def read(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("bounded JSON artifact exceeds registered size")
    return json.loads(path.read_text())


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"nonnegative integer required: {name}")
    return value


def pinned_source_proofs(source=SOURCE):
    """Bind exact historical producer operators, never current-file assumptions."""
    proofs = {}
    for path, scopes in SOURCE_SCOPES.items():
        body = subprocess.check_output(["git", "show", f"{PRODUCER_SHA}:{path}"], cwd=source)
        text = body.decode()
        tree, found = ast.parse(text), {}

        def visit(node, parents=(), scopes=scopes, text=text, found=found):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = ".".join((*parents, child.name))
                    if name in scopes:
                        lines = text.splitlines(keepends=True)
                        segment = "".join(lines[child.lineno - 1 : child.end_lineno])
                        found[name] = dict(
                            first_line=child.lineno,
                            last_line=child.end_lineno,
                            sha256=hashlib.sha256(segment.encode()).hexdigest(),
                        )
                    visit(child, (*parents, child.name))
                else:
                    visit(child, parents)

        visit(tree)
        if set(found) != set(scopes):
            raise ValueError(f"required historical source scope missing: {path}")
        proofs[path] = dict(
            git_source_sha=PRODUCER_SHA,
            file_sha256=hashlib.sha256(body).hexdigest(),
            scopes=found,
        )
    return proofs


def component_record(row):
    """Return only actually recorded quantities, not reconstructed payload sizes."""
    storage = row["storage"]
    guard_names = ("registered_guard_32", "cache_dependent_guard_32", "prefix_128")
    if any(
        type(row["diagnostics"][name]["registered_guard_pass"]) is not bool for name in guard_names
    ):
        raise ValueError("saved numerical decisions must be explicit booleans")
    initial, final, peak = (
        nonnegative(row[name], name)
        for name in ("actual_initial_bytes", "actual_final_bytes", "decode_owned_peak_bytes")
    )
    for name, expected in (
        ("initial_active_kv_storage_bytes", initial),
        ("final_active_kv_storage_bytes", final),
        ("peak_active_kv_storage_bytes", peak),
    ):
        if nonnegative(storage[name], name) != expected:
            raise ValueError("row and storage endpoints disagree")
    if peak < max(initial, final):
        raise ValueError("observed peak is below an observed endpoint")
    prompt = storage["prompt_construction"]
    history = storage["generated_history_storage"]
    updates = storage["history_updates_per_layer"]
    if (
        len(prompt) != 32
        or {part["layer"] for part in prompt} != set(range(32))
        or len(history) != 32
        or len(updates) != 32
        or any(type(n) is not int or n != 127 for n in updates)
        or row["final_next_position"] != row["original_logical_length"] + 127
        or storage.get("source_history") is not True
        or storage.get("source_history_constructor") != "direct"
        or storage.get("generated_history_lossless_and_append_only") is not True
        or storage.get("unchanged_original_selected_memberships") is not True
    ):
        raise ValueError("complete pinned layer/history/position records required")
    # No per-layer byte formula is invented from a token count or tensor shape.
    prompt_records = [
        dict(
            layer=part["layer"],
            format=part["format"],
            component_bytes=nonnegative(part["component_bytes"], "prompt component"),
        )
        for part in sorted(prompt, key=lambda part: part["layer"])
    ]
    history_counts = Counter()
    for part in history:
        count = tuple(nonnegative(part[k], k) for k in ("sealed_chunks", "raw_tail_used"))
        history_counts[count] += 1
    normalized = nonnegative(
        storage["source_history_normalized_tail_bytes"], "normalized history tail"
    )
    workspace = nonnegative(storage["active_workspace_bytes"], "final active workspace")
    leases = nonnegative(storage["outstanding_layer_leases"], "outstanding layer leases")
    # These were retired at the final endpoint. Nonzero amounts lack saved owner
    # identities for deduplication and must not be summed as disjoint categories.
    if workspace != 0 or leases != 0 or normalized > final:
        raise ValueError("pinned final endpoint must have retired workspace and leases")
    return dict(
        case=row["case"],
        arm=row["arm"],
        target_initial_bytes=row["target_initial_bytes"],
        boundaries=dict(initial_bytes=initial, final_bytes=final, observed_peak_bytes=peak),
        arithmetic_only=dict(
            final_minus_initial_bytes=final - initial,
            observed_peak_minus_initial_bytes=peak - initial,
            observed_peak_minus_final_bytes=peak - final,
            not_component_or_lifetime_attribution=True,
        ),
        initial_components=dict(
            unique_total_bytes=initial,
            per_layer_gross_records=prompt_records,
            gross_layer_sum_bytes=sum(p["component_bytes"] for p in prompt_records),
            gross_sum_is_not_unique_storage=True,
            payload_metadata_padding_split_bytes=None,
            shared_owner_deduplication_records_available=False,
        ),
        final_components=dict(
            normalized_history_tail_backing_bytes=normalized,
            active_reconstruction_workspace_bytes=workspace,
            outstanding_layer_lease_count=leases,
            active_layer_lease_bytes=0,
            unclassified_final_backing_bytes=final - normalized,
            unclassified_includes="prompt, sealed history, raw tails, phases, maps and headers",
            sealed_history_bytes=None,
            raw_KV_tail_bytes=None,
            padding_bytes=None,
            metadata_bytes=None,
        ),
        peak_components=dict(
            unclassified_observed_peak_bytes=peak,
            category_bytes=None,
            time_aligned_owner_inventory_available=False,
            workspace_allocation_count=nonnegative(
                storage["workspace_allocations"], "workspace allocations"
            ),
            maximum_workspace_capacity_units_unspecified=nonnegative(
                storage["maximum_workspace_capacity"], "workspace capacity"
            ),
            capacity_is_not_bytes_or_peak_lifetime_proof=True,
        ),
        history_records=dict(
            layers=32,
            updates_per_layer=127,
            count_histogram=[
                dict(sealed_chunks=k[0], raw_tail_used=k[1], layer_count=v)
                for k, v in sorted(history_counts.items())
            ],
            normalized_tail_bytes_source="SourceHistoryCache.storage_record/backing_bytes",
        ),
        saved_guard_results={
            name: row["diagnostics"][name]["registered_guard_pass"]
            for name in ("registered_guard_32", "cache_dependent_guard_32", "prefix_128")
        },
        declared_boundary=storage["boundary"],
        declared_includes=storage["includes"],
        declared_excludes=storage["excludes"],
        reconstruction_tensor_results_observed=storage["reconstruction_tensor_results_observed"],
        history_constructor_tensor_results_observed=storage[
            "source_history_constructor_tensor_results_observed"
        ],
        exact_decomposition_complete=False,
        whole_request_KV_peak_certified=False,
    )


def reconcile(rows):
    expected = {(c, t, a) for c in range(2) for t in TARGETS for a in ARMS}
    if len(rows) != 8:
        raise ValueError("exactly eight registered rows required; no intersection analysis")
    if any(type(r["case"]) is not int or type(r["target_initial_bytes"]) is not int for r in rows):
        raise ValueError("integer case and physical target required")
    keys = [(r["case"], r["target_initial_bytes"], r["arm"]) for r in rows]
    if len(set(keys)) != 8 or set(keys) != expected:
        raise ValueError("duplicate, missing or unregistered case/target/method")
    records = []
    for row in rows:
        if (
            row["status"] != "completed"
            or row["used_fullkv_fallback"] is not False
            or row["same_initial_bytes_only"] is not True
            or row["actual_initial_bytes"] != row["target_initial_bytes"]
        ):
            raise ValueError("completed exact initial construction without fallback required")
        records.append(component_record(row))
    indexed = {(r["case"], r["target_initial_bytes"], r["arm"]): r for r in records}
    pairs = []
    for case in range(2):
        for target in TARGETS:
            a, b = (indexed[case, target, arm] for arm in ARMS)
            differences = {
                name: b["boundaries"][name] - a["boundaries"][name]
                for name in ("initial_bytes", "final_bytes", "observed_peak_bytes")
            }
            pairs.append(
                dict(
                    case=case,
                    target_initial_bytes=target,
                    difference_orientation="mixed_source_minus_prompt_topk",
                    differences=differences,
                    matched_initial=differences["initial_bytes"] == 0,
                    matched_final=differences["final_bytes"] == 0,
                    matched_observed_peak=differences["observed_peak_bytes"] == 0,
                    peak_difference_component_attribution=None,
                )
            )
    return dict(
        row_count=8,
        expected_row_count=8,
        pair_count=4,
        expected_pair_count=4,
        complete_row_pairing=True,
        records=sorted(records, key=lambda r: (r["case"], r["target_initial_bytes"], r["arm"])),
        pairs=pairs,
        saved_guard_pass_counts={
            name: sum(r["saved_guard_results"][name] is True for r in records)
            for name in ("registered_guard_32", "cache_dependent_guard_32", "prefix_128")
        },
        guard_recomputed_from_logits=False,
        saved_numerical_failures_preserved=True,
        exact_decomposition_complete=False,
        missing_evidence=[
            "initial per-owner alias/deduplication and payload/header/padding categories",
            "final per-owner sealed-history, raw-tail, phase and metadata backing bytes",
            "time-aligned allocation/retirement/lease/workspace inventory at observed peak",
            "unobserved reconstruction and history-constructor tensor results",
            "whole-request construction/source-overlap and internal GEMM workspace coverage",
        ],
        invalid_unified_comparison_evidence=True,
        valid_unified_comparison_evidence=False,
        final_and_peak_byte_matching_established=False,
        independent_quality_N=0,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        six_mechanism_performance_claim_allowed=False,
        whole_request_KV_peak_certified=False,
    )


def load_parent(parent):
    parent = Path(parent).resolve(strict=True)
    names = (
        "plan.json",
        "GATE_constructed_shared_bytes.json",
        "worker/result.json",
        "worker/manifest.json",
        "worker/raw.jsonl",
        "worker/logits.pt",
    )
    paths = {name: parent / name for name in names}
    if any(p.resolve(strict=True).parent != p.parent for p in paths.values()):
        raise ValueError("canonical parent files cannot be external symlinks")
    bindings = {str(path): sha(path) for path in paths.values()}
    if (
        parent == PARENT.resolve()
        and sha(paths["GATE_constructed_shared_bytes.json"]) != PARENT_GATE_SHA
    ):
        raise ValueError("reviewed V11 construction gate hash changed")
    plan, gate, result, manifest = (read(paths[n]) for n in names[:4])
    if (
        plan["source_sha"] != PRODUCER_SHA
        or result["source_sha"] != PRODUCER_SHA
        or manifest["source_sha"] != PRODUCER_SHA
        or manifest["source_dirty"] is not False
        or plan["expected_rows"] != 8
        or result["rows"] != 8
        or result["status"] != "completed"
        or result["errors"]
        or result["plan_sha256"] != sha(paths["plan.json"])
        or manifest["plan_sha256"] != sha(paths["plan.json"])
        or gate["complete"] is not True
        or gate["status"] != "passed_constructed_initial_bytes"
        or gate["failures"]
        or gate["final_and_peak_byte_match_claim"] is not False
        or gate["superiority_established"] is not False
    ):
        raise ValueError("pinned completed negative V11 construction evidence required")
    expected_artifacts = {paths[n].resolve() for n in names if n.startswith("worker/")}
    expected_artifacts.remove(paths["worker/result.json"].resolve())
    actual = {Path(n).resolve(strict=True): h for n, h in result["bindings"].items()}
    if set(actual) != expected_artifacts or len(result["bindings"]) != 3:
        raise ValueError("producer must bind exactly its raw, logits and manifest artifacts")
    if any(sha(p) != h for p, h in actual.items()):
        raise ValueError("producer-bound artifact changed")
    raw = paths["worker/raw.jsonl"]
    if raw.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("bounded eight-row raw artifact exceeds registered size")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if digest(rows) != digest(gate["rows"]):
        raise ValueError("raw rows differ from the preserved initial-construction gate")
    return (
        rows,
        bindings,
        dict(
            original_guard_thresholds=gate["original_guard_thresholds"],
            final_and_peak_byte_match_claim=gate["final_and_peak_byte_match_claim"],
            initial_gate_status=gate["status"],
            source_sha=PRODUCER_SHA,
            result_category=manifest["result_category"],
            category_is_historical_label_not_current_admission=True,
        ),
    )


def run(root, *, cpu_gate_root, parent=PARENT):
    root = Path(root)
    if root.resolve().is_relative_to(Path(parent).resolve()):
        raise ValueError("derived report must not be written inside immutable parent")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    rows, bindings, preserved = load_parent(parent)
    result = reconcile(rows)
    result.update(
        status="completed_CPU_reconciliation_missing_component_evidence",
        source_sha=revision,
        source_root=str(SOURCE),
        parent_root=str(Path(parent).resolve()),
        policy=POLICY,
        policy_sha256=digest(POLICY),
        cpu_gate=cpu,
        cpu_gate_root=str(cpu_gate_root),
        input_bindings=bindings,
        historical_source_proofs=pinned_source_proofs(),
        preserved_negative_evidence=preserved,
        only_existing_eight_rows_read=True,
        original_artifacts_unchanged=True,
    )
    # Rehash before publication: an independently changing producer is not a
    # reproducible input. No parent artifact is ever written by this tool.
    if any(sha(Path(name)) != expected for name, expected in bindings.items()):
        raise ValueError("bound parent changed during CPU reconciliation")
    write_once(root / GATE, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    result = run(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
    print(json.dumps({"status": result["status"], "gate": str(args.root / GATE)}))


if __name__ == "__main__":
    main()
