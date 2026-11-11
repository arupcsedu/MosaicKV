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
STREAMING_KIND = "streaming_lossless_BF16_whole_request_ownership_v15"


# These complete predicates are allocation-producing sites observed in the
# pinned source trace. They are not arbitrary If/forward exemptions. Each
# caller still supplies a reviewed exact stack, file SHA, line SHA and scope.
VALIDATION_PREDICATES = {
    ("/scripts/evicastkv_review_source_controls.py", "protection_masks"): (
        "scores.ndim != 2 or not all(scores.shape) "
        "or not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) "
        "or not math.isfinite(fraction) or not 0 <= fraction <= 1 or type(seed) is not int",
    ),
    ("/evaluation/shared_source_format.py", "validate_source_tensors"): (
        "not torch.equal(union, torch.arange(n, device=key.device, dtype=torch.int32)) "
        "or not torch.equal(slots, union.to(torch.int16)[None].expand(h, n)) "
        "or not all(torch.isfinite(t).all() for t in (normalized, key, value, cos, sin)) "
        "or not torch.equal(cos[..., :64], cos[..., 64:]) "
        "or not torch.equal(sin[..., :64], sin[..., 64:])",
    ),
    ("/evaluation/precision_tier.py", "pack"): (
        "type(bits) is not int or bits not in (2, 3, 4, 8) or axis not in (2, 3) "
        "or type(group) is not int or group != 128 or values.dtype != torch.bfloat16 "
        "or values.ndim != 4 or values.shape[0] != 1 or not all(values.shape) "
        "or not bool(torch.isfinite(values).all())",
        'values.device.type == "cuda" and bool('
        "(torch.isfinite(scale) & (scale >= torch.finfo(torch.float32).tiny)).all())",
    ),
    ("/evaluation/dual_projection.py", "output_slots"): (
        "selected.ndim != 2 or selected.shape[-1] >= 32768 or not selected.shape[-1] "
        "or union.ndim != 1 or union.dtype not in (torch.int32, torch.int64) "
        "or selected.dtype not in (torch.int32, torch.int64) "
        "or bool((union < 0).any()) or bool((selected < 0).any()) "
        "or bool((union.long() >= 2**31).any()) "
        "or bool((selected.long() >= 2**31).any())",
        "not union.numel() or not bool((union[1:] > union[:-1]).all())",
        "not bool((selected[:, 1:] > selected[:, :-1]).all())",
    ),
    ("/evaluation/source_history.py", "build_chunk_direct"): (
        "not all(torch.equal(reference.view(torch.int16), value.view(torch.int16)) "
        "for reference, value in zip(layer.history, actual, strict=True))",
    ),
    ("/evaluation/source_history.py", "seal_source_full_tails"): (
        "not torch.equal(positions[1:], positions[:-1] + 1) or ("
        "cache.history_position_chunks and int(positions[0]) "
        "!= int(cache.history_position_chunks[-1][-1]) + 1)",
    ),
    ("/torch/testing/_comparison.py", "TensorLikePair._compare_regular_values_close"): (
        "torch.all(matches)",
    ),
}


def _predicate_statement(tree, rule, categories):
    """Admit only closed, source-specific If.test expressions at test lines."""
    line = rule["line"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.test.lineno <= line <= node.test.end_lineno:
            continue
        if any(isinstance(item, ast.NamedExpr) for item in ast.walk(node.test)):
            raise ValueError("named-expression validation aliases are forbidden")
        expected = {
            ast.dump(ast.parse(value, mode="eval").body, include_attributes=False)
            for (suffix, scope), values in VALIDATION_PREDICATES.items()
            if rule["filename"].endswith(suffix) and rule["qualified_function"] == scope
            for value in values
        }
        if ast.dump(node.test, include_attributes=False) not in expected:
            continue
        if categories not in ([], ["cache_auxiliary_workspace"]):
            raise ValueError("validation predicates require proof-only or auxiliary workspace")
        # A canonical torch import is allowed; arbitrary aliases/shadowed
        # builtins could change the meaning of the otherwise identical AST.
        functions = [
            item
            for item in ast.walk(tree)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.lineno <= node.lineno <= item.end_lineno
        ]
        enclosing = max(functions, key=lambda item: item.lineno)
        providers = {"torch", "math", "bool", "all", "int", "type", "zip"}
        for item in ast.walk(enclosing):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, (ast.Store, ast.Del))
                and item.id in providers
            ) or (isinstance(item, ast.arg) and item.arg in providers):
                raise ValueError("validation provider aliases/shadowing are forbidden")
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                for name in item.names:
                    bound = name.asname or name.name.split(".")[0]
                    if bound in providers and not (
                        isinstance(item, ast.Import)
                        and name.name in {"torch", "math"}
                        and name.asname is None
                    ):
                        raise ValueError("validation provider aliases/shadowing are forbidden")
        return node.test
    return None


def statement(rule, *, categories=None):
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
    tree = ast.parse(text)
    predicates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and node.test.lineno <= line <= node.test.end_lineno
    ]
    if predicates:
        if categories is not None:
            predicate = _predicate_statement(tree, rule, categories)
            if predicate is not None:
                return predicate
            # Reuse the already reviewed closed model operator. Its If form
            # is only the placeholder consistency gather/count. A same-line
            # If body is deliberately not accepted as a separate expression.
            if categories in ([], ["normal_model_activation"]) and audit.closed_non_KV_operator(
                text, rule["filename"], rule["qualified_function"], line
            ):
                return next(node.test for node in predicates if node.lineno == line)
        raise ValueError("one concrete allocation-producing expression required")
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
        and node.lineno <= line <= node.end_lineno
    ]
    if not nodes:
        raise ValueError("one concrete allocation-producing expression required")
    node = min(nodes, key=lambda value: value.end_lineno - value.lineno)
    if node.end_lineno - node.lineno > 12:
        raise ValueError("broad source-function ownership exemptions are forbidden")
    if any(isinstance(item, ast.NamedExpr) for item in ast.walk(node)):
        raise ValueError("named-expression operator aliases are forbidden")
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
        node = statement(rule, categories=uses)
        qualifiers = rule["qualified_function"].split(".")[:-1]
        broad_scope = uses and (
            rule["function"] in {"generate", "prefill", "decode_one_token"}
            or any(
                name.endswith(("Model", "DecoderLayer", "ForCausalLM", "ForConditionalGeneration"))
                for name in qualifiers
            )
        )
        if broad_scope and (
            uses != ["normal_model_activation"]
            or not audit.closed_non_KV_operator(
                Path(rule["filename"]).read_text(),
                rule["filename"],
                rule["qualified_function"],
                rule["line"],
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
    catalog_witnesses = ledger.get("catalog_model_identity_witnesses", [])
    if not isinstance(catalog_witnesses, list) or len(catalog_witnesses) > 100000:
        raise ValueError("bounded explicit named-model catalog witnesses required")
    for witness in catalog_witnesses:
        if (
            set(witness)
            != {
                "address",
                "generation",
                "catalog_sha256",
                "owner_sha256",
                "evidence_sha256",
                "source",
                "reason",
                "reviewed_by",
            }
            or trace.get("input_schema") != STREAMING_KIND
        ):
            raise ValueError("distinct typed source and exact catalog witness schema required")
        catalog = trace["baseline_model_catalog"]
        key = witness["address"], witness["generation"]
        row = rows.get(key)
        owners = [owner for owner in catalog["owners"] if digest(owner) == witness["owner_sha256"]]
        proof = witness["source"]
        if (
            row is None
            or row["baseline"] is not True
            or len(owners) != 1
            or catalog["owner_catalog_sha256"] != digest(catalog["owners"])
            or witness["catalog_sha256"] != catalog["owner_catalog_sha256"]
            or (owners[0]["address"], owners[0]["bytes"]) != (row["address"], row["bytes"])
            or not owners[0]["device"].startswith("cuda:")
            or not owners[0]["names"]
            or any(item.get("family") not in {"parameter", "buffer"} for item in owners[0]["names"])
            or digest(row["evidence"]) != witness["evidence_sha256"]
            or MODEL_WITNESS + " actual named catalog " + catalog["owner_catalog_sha256"]
            not in row["evidence"]
            or not witness["reviewed_by"]
            or len(witness["reason"].strip()) < 20
            or source_bindings.get(proof["filename"]) != proof["source_sha256"]
            or not proof["filename"].endswith("/scripts/evicastkv_streaming_lossless_ownership.py")
            or proof["qualified_function"] != "mark_baseline_model"
        ):
            raise ValueError("actual named baseline backing identity and source witness required")
        node = statement(proof)
        expected = ast.parse(
            'session.timeline.classify(owner["address"], "model", '
            '"pinned immutable resident model parameters/buffers actual named catalog " '
            '+ catalog["owner_catalog_sha256"])'
        ).body[0]
        if ast.dump(node, include_attributes=False) != ast.dump(expected, include_attributes=False):
            raise ValueError("exact named-model identity classification operator required")
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


def streaming_inputs(parent, index):
    """Validate the distinct streaming BF16 bundle, never impersonate old ARM."""
    import evicastkv_streaming_lossless_ownership as streamed
    import torch
    from evicastkv_review_source_prefix import (
        analysis_cpu_controls,
        comparisons,
        verify_recomputed_diagnostics,
    )

    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("exactly one registered streaming ownership worker required")
    plan = streamed.validate(parent)
    gate_path = parent / streamed.GATE
    gate = audit.load(gate_path)
    if (
        plan["policy"]["method"] != STREAMING_KIND
        or gate.get("source_sha") != plan["source_sha"]
        or gate.get("complete_numerical_and_trace_controls") is not True
        or gate.get("failures") != []
        or len(gate.get("summaries", [])) != 2
    ):
        raise ValueError("complete independently analyzed typed numerical/trace evidence required")
    out = parent / f"workers/{index}"
    result_path = out / "result.json"
    result = audit.load(result_path)
    if (
        result.get("method") != STREAMING_KIND
        or result.get("index") != index
        or result.get("source_sha") != plan["source_sha"]
        or result.get("status") != "completed_numerical_diagnostic"
        or result.get("failures") != []
        or result.get("all_32_128_guards_pass") is not True
        or result.get("complete_trace_with_final_snapshot_archived") is not True
    ):
        raise ValueError("complete actual streaming worker and unchanged guards required")
    required = {
        "allocator_trace.json.gz",
        "trace_receipt.json",
        "manifest.json",
        "raw.json",
        "logits.pt",
        "baseline_model_catalog.json",
        "baseline_model_allocator_matches.json",
        "GATE_cuda_operator.json",
    }
    if not required <= set(result["outputs"]):
        raise ValueError("typed worker required outputs missing")
    paths = {
        "protocol": parent / "protocol.json",
        "producer_gate": gate_path,
        "result": result_path,
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    for name, expected in result["outputs"].items():
        if Path(name).name != name or sha(out / name) != expected:
            raise ValueError("typed worker raw output hash changed")
        paths[name] = out / name
        hashes[name] = expected
    manifest = audit.load(out / "manifest.json")
    gpu = manifest.get("GPU", {})
    visible = [device for device in gpu.get("devices", []) if device.get("visible_to_process")]
    if (
        manifest.get("source_sha") != plan["source_sha"]
        or manifest.get("protocol_sha256") != hashes["protocol"]
        or manifest.get("method") != STREAMING_KIND
        or manifest.get("no_labels") is not True
        or manifest.get("no_native_reference_forward_inside_window") is not True
        or not isinstance(manifest.get("job_id"), str)
        or not manifest["job_id"].isdigit()
        or int(manifest["job_id"]) < 1
        or gpu.get("query_error")
        or gpu.get("background_processes")
        or len(visible) != 1
        or "A100" not in visible[0].get("name", "")
        or gpu.get("visible_process_concurrency") not in (0, 1)
        or audit.load(out / "GATE_cuda_operator.json").get("passed") is not True
    ):
        raise ValueError(
            "saved source/protocol/operator/one-Slurm-A100 isolation evidence required"
        )
    receipt = audit.load(out / "trace_receipt.json")
    value = streamed.load_bundle(out / "allocator_trace.json.gz", receipt)
    if value["schema"] != STREAMING_KIND or value["source_sha"] != plan["source_sha"]:
        raise ValueError("distinct streaming schema/source binding required")
    catalog = audit.load(out / "baseline_model_catalog.json")
    matches = audit.load(out / "baseline_model_allocator_matches.json")
    if (
        catalog != value["baseline_model_catalog"]
        or manifest["baseline_catalog_sha256"] != hashes["baseline_model_catalog.json"]
        or manifest["baseline_model_metadata_digest"] != catalog["owner_catalog_sha256"]
        or matches.get("catalog_sha256") != catalog["owner_catalog_sha256"]
        or matches.get("other_baseline_owners_exempted") is not False
    ):
        raise ValueError("actual named-model catalog evidence differs")
    raw = audit.load(out / "raw.json")
    if (
        raw.get("method") != STREAMING_KIND
        or raw.get("source_sha") != plan["source_sha"]
        or raw.get("index") != index
    ):
        raise ValueError("typed raw request identity differs")
    streamed.verify_request(raw["record"], raw["counters"])
    analysis_cpu_controls()
    saved = torch.load(out / "logits.pt", map_location="cpu", weights_only=True)
    if set(saved) != {"reference", "candidate"} or any(
        not isinstance(saved[name], torch.Tensor)
        or saved[name].ndim != 2
        or saved[name].shape[0] != 128
        for name in ("reference", "candidate")
    ):
        raise ValueError("complete saved 128-prefix logits required")
    diagnostics = comparisons(saved["reference"], saved["candidate"])
    verify_recomputed_diagnostics(diagnostics, raw["diagnostics"])
    difference = float((saved["reference"][0] - saved["candidate"][0]).abs().max())
    if (
        difference != raw["first_prefill_logit_difference"]
        or difference > streamed.POLICY["first_prefill_logit_tolerance"]
        or any(
            diagnostics[name]["registered_guard_pass"] is not True
            for name in (
                "registered_guard_32",
                "cache_dependent_guard_32",
                "prefix_64",
                "prefix_128",
            )
        )
    ):
        raise ValueError("recomputed unchanged 32/128 numerical controls failed")
    capture = value["capture_metadata"]
    allowed_unresolved = {
        "unclassified_allocation_generations",
        "opaque_allocator_workspace_not_classified",
        "deferred_owner_timestamp_not_unique",
        "conflicting_deferred_owner_role",
    }
    errors = set(capture["blocks_exact_certification"]) - allowed_unresolved
    if (
        capture["complete_request"] is not True
        or capture["allocator_history_complete"] is not True
        or errors
    ):
        raise ValueError("complete allocator window/history/structural controls required")
    trace = dict(value, input_schema=STREAMING_KIND, clock_calibration=capture["clock_calibration"])
    declared = dict(observations=value["declared_owner_observations"])
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("typed evidence changed during validation")
    return (
        dict(
            trace=trace,
            declared=declared,
            conservative=capture,
            producer_sha=plan["source_sha"],
            worker_complete=True,
            ownership_capture_complete=True,
            structural_errors=[],
            input_schema=STREAMING_KIND,
            producer_method=STREAMING_KIND,
        ),
        paths,
        hashes,
    )


def run(root, *, parent, reviewed_rules, cpu_gate_root, worker_index=None):
    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("new immutable scratch KV-refinement root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    plan_path = parent / "protocol.json"
    typed = (
        plan_path.exists()
        and audit.load(plan_path).get("policy", {}).get("method") == STREAMING_KIND
    )
    selected_worker_index = None
    if typed:
        selected_worker_index = 0 if worker_index is None else worker_index
        values, paths, hashes = streaming_inputs(parent, selected_worker_index)
    else:
        if worker_index is not None:
            raise ValueError("worker-index is exclusive to the distinct typed streaming study")
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
        structural_errors=values.get("structural_errors", [])
        if typed or conservative["allocator_interval_bounds_valid"]
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
        input_schema=values.get("input_schema", "source_ownership_trace_first_CPU_recovery"),
        producer_method=values.get("producer_method", "registered_legacy_source_variant"),
        worker_index=selected_worker_index,
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
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.root,
                parent=args.parent,
                reviewed_rules=args.reviewed_rules,
                cpu_gate_root=args.cpu_gate_root,
                worker_index=args.worker_index,
            )
            if args.execute
            else dict(status="dry_run", new_GPU_inference=0),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
