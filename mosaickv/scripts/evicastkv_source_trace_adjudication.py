#!/usr/bin/env python3
"""Offline exact-stack source audit; unknowns never become non-KV by default.

Rules are authored only after source inspection, separately reviewed, and bound
to one immutable trace. No default/historical rules are imported. This tool may
certify a scalar peak within the declared PyTorch allocator boundary, never
whole-device HBM, baseline superiority, or independent answer quality.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import evicastkv_source_ownership_closure as source_ownership
from evicastkv_source_ownership_closure import conservative_report, unknown_clusters
from evicastkv_source_trace_codec import KIND as CODEC_KIND
from evicastkv_source_trace_codec import decode, document_digests, explicit_limits

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.measurements.kv_timeline import KV_ROLES, AllocationTimeline

SOURCE = Path(__file__).resolve().parents[2]
BOUNDARY = "PyTorch_CUDA_requested_KV_backing_plus_language_attention_and_algorithm_workspace"
MAX_COMPRESSED = 512 * 2**20
MAX_JSON = 2 * 2**30
SCRATCH = Path("/scratch/djy8hg")
TRACE_FIRST_VARIANT = "source_owner_trace_first_indexed_generation_v13_lossless_metadata_DAG"
TRACE_FIRST_VARIANTS = {TRACE_FIRST_VARIANT, "source_owner_trace_first_indexed_generation_v12"}
RULE_KEYS = {
    "stack_id",
    "stack_sha256",
    "role",
    "reason",
    "frame_index",
    "filename",
    "function",
    "line",
    "source_sha256",
    "source_line_sha256",
    "qualified_function",
    "files",
    "reviewed_by",
    "review_status",
    "match_mode",
}


def load(path, *, expected_json_bytes=None, with_codec=False, codec_limits=None):
    limits = explicit_limits(codec_limits)
    path = Path(path)
    before = path.stat()
    compressed_limit = min(MAX_COMPRESSED, limits["compressed_bytes"])
    json_limit = min(MAX_JSON, limits["encoded_JSON_bytes"])
    if before.st_size > compressed_limit:
        raise ValueError("metadata file exceeds bounded offline audit size")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(json_limit + 1)
    if len(raw) > json_limit:
        raise ValueError("uncompressed metadata exceeds bounded offline audit size")
    if expected_json_bytes is not None and len(raw) != expected_json_bytes:
        raise ValueError("complete archive JSON size differs from receipt")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("immutable metadata changed during read")
    value = json.loads(raw)
    codec = value if isinstance(value, dict) and value.get("codec") == CODEC_KIND else None
    decoded = decode(codec, limits=limits) if codec is not None else value
    return (decoded, codec) if with_codec else decoded


def stack_fingerprint(frames):
    normalized = []
    for frame in frames:
        if (
            not isinstance(frame, (list, tuple))
            or len(frame) != 3
            or not isinstance(frame[0], str)
            or not isinstance(frame[1], str)
            or type(frame[2]) is not int
        ):
            raise ValueError("exact recorded filename/function/line triples required")
        normalized.append(list(frame))
    return digest(normalized)


def _scope(text, function, line):
    """Resolve the actual qualified Python scope at an exact source line."""
    matches = []

    def walk(node, parents=()):
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        current = (*parents, node.name) if named else parents
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
            and node.lineno <= line <= node.end_lineno
        ):
            matches.append((len(current), ".".join(current)))
        for child in ast.iter_child_nodes(node):
            walk(child, current)

    walk(ast.parse(text))
    if not matches:
        raise ValueError("review anchor does not resolve to the recorded Python function")
    return max(matches)[1]


def closed_non_KV_operator(text, filename, qualified, line):
    """Only the inspected, closed operator templates—not a forward exemption.

    Caller has already checked the actual full stack, captured file hash, exact
    source line hash and qualified scope. Match the allocation-producing AST,
    not a substring or an arbitrary model/attention/clone/capture operation.
    """
    templates = {
        ("/transformers/models/llama/modeling_llama.py", "LlamaDecoderLayer.forward"): (
            "hidden_states = residual + hidden_states",
        ),
        ("/transformers/models/llava/modeling_llava.py", "LlavaModel.forward"): (
            "inputs_embeds = self.get_input_embeddings()(input_ids)",
            "image_features = torch.cat(image_features, dim=0).to("
            "inputs_embeds.device, inputs_embeds.dtype)",
            "inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)",
        ),
        ("/transformers/models/llava/modeling_llava.py", "LlavaModel.get_placeholder_mask"): (
            "special_image_mask = input_ids == self.config.image_token_id",
            "n_image_tokens = special_image_mask.sum()",
        ),
    }
    expected = [
        snippet
        for (suffix, scope), snippets in templates.items()
        if str(filename).endswith(suffix) and scope == qualified
        for snippet in snippets
    ]
    candidates = [
        node for node in ast.walk(ast.parse(text)) if getattr(node, "lineno", None) == line
    ]
    allowed = {
        ast.dump(ast.parse(snippet).body[0], include_attributes=False) for snippet in expected
    }
    if any(
        isinstance(node, ast.Assign) and ast.dump(node, include_attributes=False) in allowed
        for node in candidates
    ):
        return True
    if (
        str(filename).endswith("/transformers/models/llava/modeling_llava.py")
        and qualified == "LlavaModel.get_placeholder_mask"
    ):
        predicate = ast.parse(
            "inputs_embeds[special_image_mask].numel() != image_features.numel()", mode="eval"
        ).body
        return any(
            isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False)
            == ast.dump(predicate, include_attributes=False)
            for node in candidates
        )
    return False


def checked_rules(trace, ledger):
    """Require explicit exact-stack and file/line evidence, with no fallbacks."""
    if (
        ledger.get("schema_version") != 1
        or ledger.get("boundary") != BOUNDARY
        or ledger.get("review_status") != "explicit_source_inspection_complete"
        or not isinstance(ledger.get("reviewed_by"), str)
        or not ledger["reviewed_by"].strip()
    ):
        raise ValueError("explicit reviewed source ledger and declared boundary required")
    rules = ledger.get("rules")
    if not isinstance(rules, list):
        raise ValueError("explicit finite rule list required; empty means no new attribution")
    by_stack = {}
    all_files = {}
    for rule in rules:
        if set(rule) != RULE_KEYS:
            raise ValueError("exact reviewed rule schema required; no generic/fallback fields")
        if (
            rule["match_mode"] != "exact_full_stack"
            or rule["review_status"] != "explicit_source_inspection_complete"
            or not isinstance(rule["reviewed_by"], str)
            or not rule["reviewed_by"].strip()
            or not isinstance(rule["reason"], str)
            or len(rule["reason"].strip()) < 20
            or rule["role"] not in KV_ROLES | {"non_kv"}
        ):
            raise ValueError("reviewed semantic role/reason and exact full-stack matching required")
        index = rule["stack_id"]
        if type(index) is not int or not 0 <= index < len(trace["stack_table"]):
            raise ValueError("rule stack index is outside this immutable trace")
        frames = trace["stack_table"][index]
        if rule["stack_sha256"] != stack_fingerprint(frames) or index in by_stack:
            raise ValueError("changed or duplicate/ambiguous exact stack rule")
        anchor = rule["frame_index"]
        if type(anchor) is not int or not 0 <= anchor < len(frames):
            raise ValueError("concrete exact operator frame required")
        if list(frames[anchor]) != [rule["filename"], rule["function"], rule["line"]]:
            raise ValueError("reviewed operator anchor differs from the recorded frame")
        captured_files = trace["installed_and_local_source_hashes"]
        expected = {
            frame[0]: captured_files[frame[0]]["sha256"]
            for frame in frames
            if captured_files.get(frame[0], {}).get("sha256")
        }
        if rule["files"] != expected or rule["filename"] not in expected:
            raise ValueError("rule must bind every available exact-stack source file")
        for filename, expected_hash in expected.items():
            path = Path(filename)
            if (
                not path.is_absolute()
                or not path.is_file()
                or str(path.resolve(strict=True)) != captured_files[filename]["canonical_path"]
                or sha(path) != expected_hash
            ):
                raise ValueError("recorded operator/source file changed or is unavailable")
            all_files[filename] = expected_hash
        if rule["source_sha256"] != expected[rule["filename"]]:
            raise ValueError("operator file hash differs from the captured source")
        text = Path(rule["filename"]).read_text()
        lines = text.splitlines(keepends=True)
        line = rule["line"]
        if type(line) is not int or not 1 <= line <= len(lines):
            raise ValueError("exact positive operator line required")
        if rule["source_line_sha256"] != hashlib.sha256(lines[line - 1].encode()).hexdigest():
            raise ValueError("reviewed operator line hash changed")
        qualified = _scope(text, rule["function"], line)
        if rule["qualified_function"] != qualified:
            raise ValueError("reviewed qualified operator is not the actual source scope")
        # A whole decoder/model method is not evidence that its allocations are
        # non-KV. Specific MLP/vision/normalization operators can be reviewed.
        qualifiers = qualified.split(".")[:-1]
        broad = rule["function"] in {"generate", "prefill", "decode_one_token"} or any(
            name.endswith(("Model", "DecoderLayer", "ForCausalLM", "ForConditionalGeneration"))
            or ("Attention" in name and not name.startswith("CLIP"))
            for name in qualifiers
        )
        if (
            rule["role"] == "non_kv"
            and (broad or qualified == "forward")
            and not closed_non_KV_operator(text, rule["filename"], qualified, line)
        ):
            raise ValueError("broad model-forward/non-KV exemptions are forbidden")
        by_stack[index] = rule
    if any(sha(Path(filename)) != expected for filename, expected in all_files.items()):
        raise ValueError("reviewed operator/source file changed during adjudication")
    return by_stack


def replay_trace(trace):
    """Independently reconstruct generations from actual alloc/free events."""
    expected = trace["allocator_records"]
    identities = {(row["address"], row["generation"]) for row in expected}
    if len(identities) != len(expected) or any(
        type(row["baseline"]) is not bool for row in expected
    ):
        raise ValueError("duplicate/malformed raw allocation generations")
    allowed_actions = {
        "alloc",
        "free_requested",
        "free_completed",
        "oom",
        "snapshot",
        "segment_alloc",
        "segment_free",
        "segment_map",
        "segment_unmap",
    }
    if any(
        event.get("sequence") != index or event.get("action") not in allowed_actions
        for index, event in enumerate(trace["actual_events"])
    ):
        raise ValueError("unrecognized or out-of-order actual allocator event")
    timeline = AllocationTimeline()
    for row in expected:
        if row["baseline"]:
            timeline.seed(row["address"], row["bytes"])
    timeline.ingest(
        [
            {key: value for key, value in event.items() if key != "sequence"}
            for event in trace["actual_events"]
        ]
    )
    if timeline.errors or set(timeline.allocations) != identities:
        raise ValueError("allocator event replay is incomplete or structurally inconsistent")
    for row in expected:
        actual = timeline.allocations.get((row["address"], row["generation"]))
        if actual is None or any(
            actual.get(key) != row.get(key)
            for key in (
                "address",
                "generation",
                "bytes",
                "start",
                "end",
                "baseline",
                "time_us",
                "end_time_us",
            )
        ):
            raise ValueError("raw allocation generations differ from independently replayed events")
        if not row["baseline"] and stack_fingerprint(
            trace["stack_table"][row["stack_id"]]
        ) != stack_fingerprint(timeline.stacks[actual["stack_id"]]):
            raise ValueError("allocation stack differs from actual event frames")
    return timeline


def indexed_conservative_report(capture, observations, *, events):
    """Keep the V11 interval predicate, searching only a pointer's generations."""
    by_address = {}
    for row in capture["records"]:
        by_address.setdefault(row["address"], []).append(row)
    original_match = source_ownership.generation_for_observation

    def indexed(_records, owner, observation, clock):
        return original_match(by_address.get(owner["address"], ()), owner, observation, clock)

    with patch.object(source_ownership, "generation_for_observation", indexed):
        return conservative_report(capture, observations, events=events)


def adjudicate(trace, conservative, declared, ledger):
    if any(
        type(conservative[key]) is not bool
        for key in ("complete_request", "allocator_history_complete")
    ):
        raise ValueError("strict boolean observer completeness metadata required")
    timeline = replay_trace(trace)
    raw_capture = dict(
        records=trace["allocator_records"],
        events=len(trace["actual_events"]),
        stack_table=trace["stack_table"],
        clock_calibration=trace["clock_calibration"],
        complete_request=conservative["complete_request"],
        allocator_history_complete=conservative["allocator_history_complete"],
        blocks_exact_certification=conservative["blocks_exact_certification"],
    )
    rebuilt = indexed_conservative_report(
        raw_capture, deepcopy(declared["observations"]), events=trace["actual_events"]
    )
    stored = {(row["address"], row["generation"]): row for row in conservative["records"]}
    if len(stored) != len(rebuilt["records"]):
        raise ValueError("duplicate or missing conservative allocation generations")
    rules = checked_rules(trace, ledger)
    resolved = compatible_KV_matches = 0
    for row in rebuilt["records"]:
        identity = row["address"], row["generation"]
        if stored[identity]["role"] != row["role"]:
            raise ValueError("saved conservative attribution does not reproduce declared owners")
        rule = rules.get(row.get("stack_id"))
        if rule is not None:
            compatible_KV = row["role"] in KV_ROLES and rule["role"] in KV_ROLES
            if row["role"] not in {"unknown", rule["role"]} and not compatible_KV:
                raise ValueError("reviewed role conflicts with an actual declared/model owner")
            if compatible_KV and row["role"] != rule["role"]:
                compatible_KV_matches += 1
            if row["role"] == "unknown":
                row["role"] = rule["role"]
                row["evidence"].append(
                    f"reviewed_exact_stack:{rule['stack_sha256']}:{rule['reason']}"
                )
                resolved += 1
        timeline.allocations[identity] = row
    unknown = sum(row["role"] == "unknown" for row in timeline.allocations.values())
    valid = rebuilt["allocator_interval_bounds_valid"]
    result = timeline.report(
        complete=conservative["complete_request"],
        history_complete=conservative["allocator_history_complete"],
        opaque_workspace_classified=unknown == 0,
    )
    lower = result["known_KV_peak_lower_bound_bytes"]
    upper = result["KV_peak_upper_bound_including_unknown_bytes"]

    # Verify scalar interval arithmetic independently of AllocationTimeline.
    def interval_peak(roles):
        changes = {}
        for row in result["records"]:
            if row["role"] in roles:
                changes[row["start"]] = changes.get(row["start"], 0) + row["bytes"]
                if row["end"] is not None:
                    changes[row["end"]] = changes.get(row["end"], 0) - row["bytes"]
        peak = live = 0
        for sequence in sorted(changes):
            live += changes[sequence]
            if live < 0:
                raise ValueError("negative live-storage interval balance")
            peak = max(peak, live)
        return peak

    if lower != interval_peak(KV_ROLES) or upper != interval_peak(KV_ROLES | {"unknown"}):
        raise ValueError("independent interval reducer disagrees")
    exact = valid and lower == upper
    result.update(
        boundary=BOUNDARY,
        reviewed_rules=len(rules),
        newly_classified_generations=resolved,
        compatible_KV_rule_matches_retaining_known_owner_role=compatible_KV_matches,
        allocator_interval_bounds_valid=valid,
        known_KV_peak_lower_bound_bytes=lower if valid else None,
        KV_peak_upper_bound_including_unknown_bytes=upper if valid else None,
        exact_PyTorch_boundary_peak_certified=exact,
        exact_scoped_KV_peak_certified=exact,
        exact_KV_peak_bytes=lower if exact else None,
        exact_whole_request_KV_peak_certified=exact,
        exact_whole_request_source_KV_plus_attention_workspace_peak_certified=exact,
        exact_whole_request_KV_only_peak_certified=False,
        whole_device_memory_certified=False,
        complete_allocation_attribution=unknown == 0,
        certificate_kind=(
            "complete_scoped_attribution" if unknown == 0 else "equal_interval_bounds"
        )
        if exact
        else "unresolved_interval_bounds",
        unknown_generations_preserved=unknown,
        certification_requires_explicit_reviewed_rule_ledger=True,
        complete_model_request_observer_window=conservative["complete_request"],
        certification_prerequisites=[
            "Complete prefill, construction, original-source retirement, "
            "decode and cleanup window.",
            "Unoverwritten allocator history and independent event/generation replay agreement.",
            "Exact source-reviewed roles or provably equal lower/upper interval peaks.",
        ],
        KV_only_claim_not_certified_because_boundary_includes_attention_and_reader_workspace=True,
        no_automatic_unknown_or_non_KV_fallback=True,
        independent_interval_reducer_checked=True,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        exclusions="CPU owners, allocator rounding/reservation, "
        "non-PyTorch/vendor/device allocations",
    )
    return result


def _inputs(parent):
    parent = Path(parent)
    if (parent / "recovery.json").exists():
        recovery = load(parent / "recovery.json")
        clean_source(Path(recovery["source_root"]), recovery["source_sha"])
        if recovery["kind"] != "CPU_only_trace_first_recovery" or any(
            sha(Path(path)) != expected for path, expected in recovery["bindings"].items()
        ):
            raise ValueError("CPU recovery provenance or immutable raw binding changed")
        paths = {name: Path(path) for name, path in recovery["artifact_paths"].items()}
        hashes = recovery["artifact_hashes"]
        if set(paths) != {"trace", "conservative", "declared", "unknown_clusters"} or any(
            sha(path) != hashes[name] for name, path in paths.items()
        ):
            raise ValueError("CPU-derived ownership bundle changed")
        values = {name: load(path) for name, path in paths.items()}
        values.update(
            producer_sha=recovery["producer_sha"],
            worker_complete=False,
            ownership_capture_complete=recovery["ownership_capture_complete"],
            CPU_recovered=True,
        )
        return values, paths, hashes
    protocol = load(parent / "protocol.json")
    clean_source(Path(protocol["source_root"]), protocol["source_sha"])
    result = load(parent / "worker/result.json")
    if result["source_sha"] != protocol["source_sha"]:
        raise ValueError("ownership trace producer differs from registered source")
    for name, expected in result["outputs"].items():
        if Path(name).name != name or sha(parent / "worker" / name) != expected:
            raise ValueError("immutable ownership raw artifact hash changed")
    paths = dict(
        trace=parent / "worker/allocator_trace.json.gz",
        conservative=parent / "worker/allocator_ledger.json.gz",
        declared=parent / "worker/declared_owners.json",
        unknown_clusters=parent / "analysis/unknown_allocation_clusters.json",
    )
    hashes = {name: sha(path) for name, path in paths.items()}
    if load(paths["unknown_clusters"])["raw_trace_sha256"] != hashes["trace"]:
        raise ValueError("unknown-stack report is not bound to this raw allocator trace")
    values = {name: load(path) for name, path in paths.items()}
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("immutable source audit artifact changed during loading")
    values.update(
        producer_sha=protocol["source_sha"], worker_complete=result["status"] == "completed"
    )
    return values, paths, hashes


def recover_trace_first(root, *, parent, cpu_gate_root):
    """Rebuild a complete early allocator archive, never a GPU-completed result.

    No second GPU attempt is allowed or requested. Numerical/model decisions
    remain the original producer's evidence and are not promoted by this path.
    """
    from evicastkv_review_followups import cpu_gate

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("new immutable scratch CPU recovery root required")
    revision = clean_source(SOURCE)
    checked_cpu = cpu_gate(Path(cpu_gate_root), revision)
    protocol_path = parent / "protocol.json"
    manifest_path = parent / "worker/manifest.json"
    receipt_path = parent / "worker/trace_pre_attribution_receipt.json"
    trace_path = parent / "worker/allocator_trace_pre_attribution.json.gz"
    declared_path = parent / "worker/declared_owners_pre_attribution.json"
    raw_path, logits_path = parent / "worker/raw.json", parent / "worker/logits.pt"
    paths = [
        protocol_path,
        manifest_path,
        receipt_path,
        trace_path,
        declared_path,
        raw_path,
        logits_path,
        Path(cpu_gate_root) / "full-cpu/cpu-tests.json",
    ]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("complete regular early archive/receipt/request files required")
    bindings = {str(path): sha(path) for path in paths}
    protocol, receipt, manifest = load(protocol_path), load(receipt_path), load(manifest_path)
    clean_source(Path(protocol["source_root"]), protocol["source_sha"])
    if any(sha(Path(path)) != value for path, value in protocol["bindings"].items()):
        raise ValueError("registered producer source/evidence hash changed")
    bindings.update(protocol["bindings"])
    if (
        receipt["observer_variant"] not in TRACE_FIRST_VARIANTS
        or protocol["policy"].get("observer_variant") != receipt["observer_variant"]
        or manifest["source_sha"] != protocol["source_sha"]
        or manifest["source_dirty"] is not False
        or manifest["policy_sha256"] != digest(protocol["policy"])
        or receipt["source_sha"] != protocol["source_sha"]
        or receipt["protocol_sha256"] != sha(protocol_path)
        or Path(receipt["trace_path"]).resolve() != trace_path.resolve()
        or receipt["trace_sha256"] != sha(trace_path)
        or receipt["declared_owners_sha256"] != sha(declared_path)
        or receipt["complete_gzip_published_atomically"] is not True
        or receipt["attribution_not_yet_run"] is not True
        or receipt["new_roles_assigned"] != 0
        or type(receipt["uncompressed_bytes"]) is not int
        or not 0 < receipt["uncompressed_bytes"] <= MAX_JSON
    ):
        raise ValueError("early archive receipt differs from exact source/protocol/raw evidence")
    trace, codec = load(
        trace_path, expected_json_bytes=receipt["uncompressed_bytes"], with_codec=True
    )
    hashes = (
        document_digests(
            codec,
            tuple(
                (key,)
                for key in (
                    "actual_events",
                    "allocator_records",
                    "final_allocator_snapshot",
                    "capture_metadata",
                )
            ),
        )
        if codec is not None
        else {}
    )

    def semantic_hash(key):
        return hashes[(key,)] if codec is not None else digest(trace[key])

    capture, declared = trace["capture_metadata"], load(declared_path)
    if (
        trace["observer_variant"] != receipt["observer_variant"]
        or declared["observer_variant"] != receipt["observer_variant"]
        or semantic_hash("actual_events") != receipt["actual_events_sha256"]
        or semantic_hash("allocator_records") != receipt["allocation_records_sha256"]
        or semantic_hash("final_allocator_snapshot") != receipt["final_allocator_snapshot_sha256"]
        or semantic_hash("capture_metadata") != receipt["capture_sha256"]
        or (codec is not None and receipt.get("archive_codec_statistics") != codec["statistics"])
        or capture["records"] != trace["allocator_records"]
        or capture["stack_table"] != trace["stack_table"]
        or capture.get("clock_calibration") != trace["clock_calibration"]
        or len(trace["actual_events"]) != capture["events"]
        or len(trace["actual_events"]) > protocol["policy"]["max_allocator_entries"]
        or not isinstance(trace["final_allocator_snapshot"], dict)
    ):
        raise ValueError("complete trace/capture/event/snapshot hashes or structure differ")
    for field in ("complete_request", "allocator_history_complete"):
        if any(item.get(field) is not True for item in (trace, capture, receipt)):
            raise ValueError("complete model-window and allocator-history booleans required")
    raw = load(raw_path)
    if (
        raw.get("arm") != protocol["policy"]["arm"]
        or raw.get("record", {}).get("output_tokens") != 128
        or raw.get("counters", {}).get("updates") != 127
        or raw.get("record", {}).get("storage", {}).get("history_updates_per_layer") != [127] * 32
    ):
        raise ValueError("complete consumed 128-prediction model-request evidence required")
    for filename, info in trace["installed_and_local_source_hashes"].items():
        if "sha256" in info:
            path = Path(filename)
            if (
                not path.is_file()
                or str(path.resolve()) != info["canonical_path"]
                or sha(path) != info["sha256"]
            ):
                raise ValueError("captured source/operator hash changed")
            bindings[str(path)] = info["sha256"]
    replay_trace(trace)
    report = indexed_conservative_report(
        capture, deepcopy(declared["observations"]), events=trace["actual_events"]
    )
    if report["allocator_interval_bounds_valid"] is not True:
        raise ValueError("structural/coverage failure blocks CPU ownership recovery")
    if any(sha(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("immutable early archive inputs changed during CPU replay")
    root.mkdir(parents=True)
    ledger_path = root / "allocator_ledger_CPU_derived.json.gz"
    with gzip.open(ledger_path, "xt", compresslevel=1) as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    cluster_path = root / "unknown_allocation_clusters.json"
    write_once(
        cluster_path,
        dict(
            clusters=unknown_clusters(report),
            raw_trace_sha256=sha(trace_path),
            no_unknowns_waived=True,
        ),
    )
    artifacts = dict(
        trace=trace_path,
        conservative=ledger_path,
        declared=declared_path,
        unknown_clusters=cluster_path,
    )
    record = dict(
        kind="CPU_only_trace_first_recovery",
        source_sha=revision,
        source_root=str(SOURCE),
        producer_sha=protocol["source_sha"],
        parent=str(parent),
        bindings=bindings,
        artifact_paths={name: str(path) for name, path in artifacts.items()},
        artifact_hashes={name: sha(path) for name, path in artifacts.items()},
        cpu_gate=checked_cpu,
        ownership_capture_complete=True,
        GPU_worker_completed=False,
        numerical_guard_promoted=False,
        new_GPU_inference=0,
        raw_artifacts_modified=False,
        measurement_type="derived",
        result_category="unified_comparison",
        result_origin="locally_measured",
    )
    write_once(root / "recovery.json", record)
    gate = dict(
        status="CPU_recovered_trace_inventory",
        recovery_sha256=sha(root / "recovery.json"),
        source_sha=revision,
        producer_sha=protocol["source_sha"],
        ownership_capture_complete=True,
        GPU_worker_completed=False,
        new_GPU_inference=0,
        numerical_guard_promoted=False,
        known_KV_peak_lower_bound_bytes=report["known_KV_peak_lower_bound_bytes"],
        KV_peak_upper_bound_including_unknown_bytes=report[
            "KV_peak_upper_bound_including_unknown_bytes"
        ],
        exact_whole_request_KV_peak_certified=False,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        raw_artifacts_modified=False,
    )
    write_once(root / "GATE_source_trace_recovery.json", gate)
    return gate


def inventory(parent):
    """Review material only; never emits a proposed role or executable rules."""
    values, paths, hashes = _inputs(parent)
    trace = values["trace"]
    return dict(
        producer_sha=values["producer_sha"],
        artifact_hashes=hashes,
        artifact_paths={name: str(path) for name, path in paths.items()},
        boundary=BOUNDARY,
        unknown_stacks=[
            dict(
                stack_id=group["stack_id"],
                stack_sha256=stack_fingerprint(group["frames"]),
                frames=group["frames"],
                simultaneous_peak_bytes=group["simultaneous_peak_bytes"],
                allocation_generations=group["allocation_generations"],
            )
            for group in values["unknown_clusters"]["clusters"]
        ],
        source_files=trace["installed_and_local_source_hashes"],
        rules_generated=0,
        exact_PyTorch_boundary_peak_certified=False,
        whole_device_memory_certified=False,
        superiority_established=False,
    )


def run(root, *, parent, reviewed_rules):
    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch adjudication root required")
    source_sha = clean_source(SOURCE)
    values, paths, hashes = _inputs(parent)
    rule_hash = sha(Path(reviewed_rules))
    ledger = load(reviewed_rules)
    if (
        ledger.get("artifact_hashes") != hashes
        or ledger.get("producer_sha") != values["producer_sha"]
    ):
        raise ValueError("reviewed rules do not bind this exact trace and ownership evidence")
    result = adjudicate(values["trace"], values["conservative"], values["declared"], ledger)
    if sha(Path(reviewed_rules)) != rule_hash or any(
        sha(path) != hashes[name] for name, path in paths.items()
    ):
        raise ValueError("review ledger or source artifacts changed during adjudication")
    if not values.get("ownership_capture_complete", values["worker_complete"]):
        result.update(
            exact_PyTorch_boundary_peak_certified=False,
            exact_scoped_KV_peak_certified=False,
            exact_whole_request_KV_peak_certified=False,
            exact_whole_request_source_KV_plus_attention_workspace_peak_certified=False,
            exact_KV_peak_bytes=None,
            certificate_kind="producer_incomplete_not_certified",
        )
    root.mkdir(parents=True)
    write_once(
        root / "manifest.json",
        dict(
            source_sha=source_sha,
            producer_sha=values["producer_sha"],
            parent=str(parent),
            inputs={
                name: dict(path=str(paths[name]), sha256=value) for name, value in hashes.items()
            },
            reviewed_rules_path=str(reviewed_rules),
            reviewed_rules_sha256=rule_hash,
            measurement_type="derived",
            result_category="unified_comparison",
            result_origin="locally_measured",
            new_GPU_inference=0,
            authors_or_human_review_certified=False,
        ),
    )
    with gzip.open(root / "adjudicated_ledger.json.gz", "xt") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    write_once(
        root / "unresolved_stacks.json",
        dict(clusters=unknown_clusters(result), no_unknowns_waived=True),
    )
    gate = {
        key: result[key]
        for key in (
            "boundary",
            "exact_PyTorch_boundary_peak_certified",
            "exact_whole_request_source_KV_plus_attention_workspace_peak_certified",
            "exact_whole_request_KV_only_peak_certified",
            "whole_device_memory_certified",
            "certificate_kind",
            "unknown_generations_preserved",
            "known_KV_peak_lower_bound_bytes",
            "KV_peak_upper_bound_including_unknown_bytes",
            "exact_KV_peak_bytes",
            "complete_allocation_attribution",
            "newly_classified_generations",
            "reviewed_rules",
        )
    }
    gate.update(
        status="certified_declared_PyTorch_boundary_only"
        if result["exact_PyTorch_boundary_peak_certified"]
        else "unresolved_declared_boundary_not_certified",
        source_sha=source_sha,
        manifest_sha256=sha(root / "manifest.json"),
        derived_ledger_sha256=sha(root / "adjudicated_ledger.json.gz"),
        raw_artifacts_modified=False,
        new_GPU_inference=0,
        superiority_established=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_source_trace_adjudication.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--reviewed-rules", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and (args.root is None or args.reviewed_rules is None):
        parser.error("execution requires --root and --reviewed-rules")
    result = (
        run(args.root, parent=args.parent, reviewed_rules=args.reviewed_rules)
        if args.execute
        else inventory(args.parent)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
