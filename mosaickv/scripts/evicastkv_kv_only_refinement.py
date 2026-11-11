#!/usr/bin/env python3
"""CPU source-reviewed refinement of two declared cache-only boundaries.

No rules are generated automatically. Historical KV+attention accounting is
preserved; no quality, numerical, freshness or device-memory gate is promoted.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

import evicastkv_source_trace_adjudication as audit
from evicastkv_review_followups import cpu_gate
from evicastkv_source_trace_codec import semantic_digest

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.measurements.cache_only_refinement import CATEGORIES, POLICY, reduce_generations

SOURCE = Path(__file__).resolve().parents[2]
SCRATCH = Path("/scratch/djy8hg")
STACK_KEYS = (audit.RULE_KEYS - {"role"}) | {"categories"}
MODEL_WITNESS = "pinned immutable resident model parameters/buffers"


def statement(rule):
    path = Path(rule["filename"])
    text = path.read_text()
    if sha(path) != rule["source_sha256"]:
        raise ValueError("source operator file changed")
    lines = text.splitlines(keepends=True)
    line = rule["line"]
    if (
        type(line) is not int
        or not 1 <= line <= len(lines)
        or hashlib.sha256(lines[line - 1].encode()).hexdigest() != rule["source_line_sha256"]
        or audit._scope(text, rule["function"], line) != rule["qualified_function"]
    ):
        raise ValueError("exact operator line/qualified-scope source proof differs")
    nodes = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
        and node.lineno <= line <= node.end_lineno
    ]
    if not nodes:
        raise ValueError("one concrete allocation-producing expression required")
    node = min(nodes, key=lambda value: value.end_lineno - value.lineno)
    if node.end_lineno - node.lineno > 12:
        raise ValueError("broad source-function ownership exemptions are forbidden")
    return node


def checked_assignments(trace, declared, ledger, *, source_bindings):
    if (
        ledger.get("schema_version") != 1
        or ledger.get("boundary_policy_sha256") != digest(POLICY)
        or ledger.get("review_status") != "explicit_source_inspection_complete"
        or not isinstance(ledger.get("reviewed_by"), str)
        or not ledger["reviewed_by"].strip()
    ):
        raise ValueError("explicit independently reviewed cache-boundary ledger required")
    rules = ledger.get("stack_rules")
    witnesses = ledger.get("model_identity_witnesses")
    if not isinstance(rules, list) or not isinstance(witnesses, list):
        raise ValueError("finite explicit source rules/witnesses required")
    converted = []
    for rule in rules:
        uses = rule.get("categories")
        if (
            set(rule) != STACK_KEYS
            or not isinstance(uses, list)
            or len(uses) != len(set(uses))
            or not set(uses) <= CATEGORIES - {"model_weights"}
        ):
            raise ValueError("exact stack/source/category schema required; no generic fallback")
        node = statement(rule)
        qualifiers = rule["qualified_function"].split(".")[:-1]
        if uses and (
            rule["function"] in {"generate", "prefill", "decode_one_token"}
            or any(
                name.endswith(("Model", "DecoderLayer", "ForCausalLM", "ForConditionalGeneration"))
                for name in qualifiers
            )
        ):
            raise ValueError("broad model-forward ownership exemptions are forbidden")
        if uses and not set(uses).intersection(
            POLICY["persistent_classes"] + POLICY["workspace_classes"]
        ):
            text = ast.unparse(node)
            if "Attention" in ".".join(qualifiers) and not any(
                term in text
                for term in (
                    "q_proj",
                    "query_states",
                    "q_embed",
                    "attn_weights",
                    "softmax",
                    "dropout",
                    "attn_output",
                    "o_proj",
                )
            ):
                raise ValueError("specific Q/softmax/context operator proof required")
            targets = getattr(node, "targets", ())
            if isinstance(node, ast.Return):
                targets = (node.value,)
            if any(
                name in {"key_states", "value_states", "k_embed", "key", "value", "kv_states"}
                for target in targets
                for name in (item.id for item in ast.walk(target) if isinstance(item, ast.Name))
            ):
                raise ValueError("a joint Q/K/V-producing operator cannot be excluded as non-cache")
        converted.append(
            {
                **{key: value for key, value in rule.items() if key != "categories"},
                "role": "kv_workspace",
            }
        )
    # Reuse strict exact-full-stack/file/hash/AST validation, NOT old role semantics.
    validated = audit.checked_rules(
        trace,
        dict(
            schema_version=1,
            boundary=audit.BOUNDARY,
            review_status=ledger["review_status"],
            reviewed_by=ledger["reviewed_by"],
            rules=converted,
        ),
    )
    categories = {rule["stack_id"]: set(rule["categories"]) for rule in rules}
    assignments = {}
    rows = {(row["address"], row["generation"]): row for row in trace["allocator_records"]}
    for key, row in rows.items():
        if row.get("stack_id") in validated and categories[row["stack_id"]]:
            assignments[key] = categories[row["stack_id"]].copy()
    for witness in witnesses:
        if set(witness) != {
            "address",
            "generation",
            "evidence_sha256",
            "source",
            "reason",
            "reviewed_by",
        }:
            raise ValueError("exact model-identity witness schema required")
        key = witness["address"], witness["generation"]
        row = rows.get(key)
        proof = witness["source"]
        if (
            row is None
            or row["baseline"] is not True
            or MODEL_WITNESS not in row["evidence"]
            or digest(row["evidence"]) != witness["evidence_sha256"]
            or not witness["reviewed_by"]
            or len(witness["reason"].strip()) < 20
            or source_bindings.get(proof["filename"]) != proof["source_sha256"]
        ):
            raise ValueError(
                "actual resident-model identity and hash-bound source witness required"
            )
        node = statement(proof)
        literals = {
            item.value
            for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        text = ast.unparse(node)
        if MODEL_WITNESS not in literals or not all(
            term in text for term in (".parameters()", ".buffers()", ".mark(")
        ):
            raise ValueError("model witness must prove actual parameter/buffer identity marking")
        assignments.setdefault(key, set()).add("model_weights")
    generation_rules = ledger.get("generation_rules", [])
    if not isinstance(generation_rules, list) or len(generation_rules) > 100000:
        raise ValueError("bounded explicit per-generation owner rules required")
    by_address = {}
    for row in trace["allocator_records"]:
        by_address.setdefault(row["address"], []).append(row)
    observations = {digest(item): item for item in declared["observations"]}
    for rule in generation_rules:
        if set(rule) != {
            "address",
            "generation",
            "stack_id",
            "categories",
            "owner_witness",
            "reason",
            "reviewed_by",
        }:
            raise ValueError("exact generation/source/owner proof schema required")
        key = rule["address"], rule["generation"]
        row = rows.get(key)
        uses = rule["categories"]
        witness = rule["owner_witness"]
        if (
            row is None
            or row.get("stack_id") != rule["stack_id"]
            or rule["stack_id"] not in validated
            or not isinstance(uses, list)
            or not uses
            or not set(uses) <= CATEGORIES - {"model_weights"}
            or not set(uses).intersection(
                POLICY["persistent_classes"] + POLICY["workspace_classes"]
            )
            or not rule["reviewed_by"]
            or len(rule["reason"].strip()) < 20
            or set(witness) != {"observation_sha256", "owner_sha256", "required_tag", "source"}
        ):
            raise ValueError(
                "registered creation stack and explicit generation owner witness required"
            )
        observation = observations.get(witness["observation_sha256"])
        if observation is None:
            raise ValueError("generation witness observation is not in raw declared-owner evidence")
        owners = [
            owner for owner in observation["owners"] if digest(owner) == witness["owner_sha256"]
        ]
        if len(owners) != 1 or witness["required_tag"] not in owners[0]["roles"]:
            raise ValueError("exact unique declared owner/tag witness required")
        owner = owners[0]
        match = audit.source_ownership.generation_for_observation(
            by_address.get(owner["address"], ()), owner, observation, trace["clock_calibration"]
        )
        if match is None or (match["address"], match["generation"]) != key:
            raise ValueError("owner timestamp does not identify this unique allocator generation")
        proof = witness["source"]
        if source_bindings.get(proof["filename"]) != proof["source_sha256"]:
            raise ValueError("owner-tag instrumentation source is not hash-bound to producer")
        node = statement(proof)
        calls = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "track"
            and item.args
            and isinstance(item.args[0], ast.Constant)
            and item.args[0].value == witness["required_tag"]
        ]
        if len(calls) != 1:
            raise ValueError("exact source statement must track the literal declared owner tag")
        assignments.setdefault(key, set()).update(uses)
    # Actual observed cache owners are generation-linked, never excluded by Q rules.
    for observation in declared["observations"]:
        for owner in observation["owners"]:
            match = audit.source_ownership.generation_for_observation(
                by_address.get(owner["address"], ()), owner, observation, trace["clock_calibration"]
            )
            if match is None:
                continue
            key = match["address"], match["generation"]
            # A tag is not sufficient new evidence. Explicit rules decide its
            # category; expose conflicts instead of automatically trusting tags.
            if any(tag in {"original_prefill_KV", "selected_formats"} for tag in owner["roles"]):
                uses = assignments.get(key, set())
                if uses and not uses.intersection(POLICY["persistent_classes"]):
                    raise ValueError(
                        "source exclusion/workspace rule conflicts with a cache payload owner"
                    )
    return assignments


def run(root, *, parent, reviewed_rules, cpu_gate_root):
    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("new immutable scratch KV-refinement root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    values, paths, hashes = audit._inputs(parent)
    trace, declared = values["trace"], values["declared"]
    if "final_allocator_snapshot" not in trace:
        raise ValueError(
            "snapshot-bound trace-first CPU recovery bundle required; "
            "do not infer from standard archive"
        )
    audit.replay_trace(trace)
    ledger_path = Path(reviewed_rules)
    ledger_hash = sha(ledger_path)
    ledger = audit.load(ledger_path)
    if (
        ledger.get("artifact_hashes") != hashes
        or ledger.get("producer_sha") != values["producer_sha"]
    ):
        raise ValueError("cache-only ledger must bind exact immutable raw owner/trace artifacts")
    producer_parent = parent
    if (parent / "recovery.json").exists():
        producer_parent = Path(audit.load(parent / "recovery.json")["parent"])
    protocol = audit.load(producer_parent / "protocol.json")
    source_bindings = {
        name: info["sha256"]
        for name, info in trace["installed_and_local_source_hashes"].items()
        if "sha256" in info
    }
    source_bindings.update(protocol["bindings"])
    assignments = checked_assignments(trace, declared, ledger, source_bindings=source_bindings)
    conservative = values["conservative"]
    result = reduce_generations(
        trace["allocator_records"],
        assignments,
        complete_request=conservative["complete_request"],
        history_complete=conservative["allocator_history_complete"],
        structural_errors=[]
        if conservative["allocator_interval_bounds_valid"]
        else ["invalid_allocator_bounds"],
    )
    source_valid = bool(values.get("ownership_capture_complete", values["worker_complete"]))
    for metric in result["metrics"].values():
        metric["source_review_validation_still_required"] = False
        metric["exact_peak_certified"] = source_valid and metric["scalar_interval_bounds_equal"]
    if sha(ledger_path) != ledger_hash or any(
        sha(path) != hashes[name] for name, path in paths.items()
    ):
        raise ValueError("source review ledger/raw artifacts changed during refinement")
    if any(sha(Path(path)) != expected for path, expected in source_bindings.items()):
        raise ValueError("hash-bound producer sources/evidence changed during refinement")
    root.mkdir(parents=True)
    manifest = dict(
        source_sha=revision,
        producer_sha=values["producer_sha"],
        cpu_gate=cpu,
        parent=str(parent),
        inputs={name: dict(path=str(path), sha256=hashes[name]) for name, path in paths.items()},
        reviewed_rules_path=str(ledger_path),
        reviewed_rules_sha256=ledger_hash,
        final_allocator_snapshot_sha256=semantic_digest(trace["final_allocator_snapshot"]),
        source_bindings=source_bindings,
        measurement_type="derived",
        result_category="unified_comparison",
        result_origin="locally_measured",
        new_GPU_inference=0,
    )
    write_once(root / "manifest.json", manifest)
    with gzip.open(root / "refined_generation_ledger.json.gz", "xt", compresslevel=1) as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    gate = dict(
        status="source_reviewed_cache_boundary_analysis",
        source_sha=revision,
        metrics=result["metrics"],
        unknown_generations=result["unknown_generations"],
        historical_KV_roles_modified=False,
        whole_device_memory_certified=False,
        numerical_or_quality_guard_promoted=False,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        manifest_sha256=sha(root / "manifest.json"),
        derived_ledger_sha256=sha(root / "refined_generation_ledger.json.gz"),
    )
    write_once(root / "GATE_kv_only_refinement.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reviewed-rules", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.root,
                parent=args.parent,
                reviewed_rules=args.reviewed_rules,
                cpu_gate_root=args.cpu_gate_root,
            )
            if args.execute
            else dict(status="dry_run", new_GPU_inference=0),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
