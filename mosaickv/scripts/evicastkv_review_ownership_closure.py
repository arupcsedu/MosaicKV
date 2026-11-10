#!/usr/bin/env python3
"""CPU-only, hash-bound attribution of the six saved Fox ownership captures.

No allocation start/end, address generation, size, prediction, or timestamp is
changed. Unknown storage remains unknown. This is a derived review of the
PyTorch allocator boundary, not a whole-device/HBM ownership certificate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.measurements.kv_timeline import KV_ROLES, ROLES, valid_whole_request_certificate
from mosaickv.measurements.ownership_review import review

SOURCE = Path(__file__).resolve().parents[2]
PRODUCER_SHA = "7cf31bf285ea7b5f4ae7d7caa615921f2e89f417"
SITE = Path("/scratch/djy8hg/env/mosaickv/lib/python3.11/site-packages")
INSTALLED_HASHES = {
    "transformers/models/mistral/modeling_mistral.py": (
        "657d56f27c6d1236ced31c59dba182dc2c1129c3377ac9c05c80b3457a36ddc0"
    ),
    "transformers/models/clip/modeling_clip.py": (
        "ca945514f82423d23d206a3dc5474025bbe80b90a62b04af731666c2134b08d6"
    ),
    "transformers/models/llava_next/modeling_llava_next.py": (
        "08fa074f658adddad1123fc6ecc06379b64ee473997172b99c581d691fc0eec7"
    ),
    "transformers/masking_utils.py": (
        "e29afe5c7058506b22c0e35e5efd8bb749dcfc2e79f32643d442b3e59088eb70"
    ),
    "torch/nn/functional.py": "2b6eb4aff305990cff9e32c6ff40ff82c1fec4b5ceba1f43540146a8eb64e952",
    "transformers/activations.py": (
        "044f510850aa8214c8212e6b572691213a36ff56d977fc723da81be1ea10b8bb"
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def rule_definitions():
    """Exact reviewed semantic scopes, never model.forward or nn.Linear alone.

    The boundary charges standard language-attention temporaries, explicit KV
    projection, forecasting, evidence-graph, selection, packing and masks to KV.
    Vision, MLP, RMSNorm, vocabulary projection and token-output checks are not
    cache backing. A caller line is used where multiple classes share forward.
    """
    output = []

    def add(file, function, role, lines, reason):
        for line in lines:
            output.append(dict(file=file, function=function, role=role, line=line, reason=reason))

    mistral = "transformers/models/mistral/modeling_mistral.py"
    add(mistral, "forward", "non_kv", [46], "MistralMLP gate/up/product/down, no cache output")
    add(mistral, "forward", "non_kv", [196, 197, 198, 199], "MistralRMSNorm arithmetic")
    add(mistral, "forward", "non_kv", [180, 238, 244, 448], "output projection/residual/vocabulary")
    add(mistral, "forward", "kv_shared_workspace", [151], "language query projection")
    add(mistral, "forward", "kv_workspace", [152, 153], "language key/value projection")
    add(
        mistral,
        "forward",
        "kv_shared_workspace",
        [289, 293, 294, 295, 296, 298],
        "shared rotary position embeddings",
    )
    for name, lines in (
        ("eager_attention_forward", [106, 107, 109, 112, 114, 116, 117]),
        ("repeat_kv", [93]),
        ("rotate_half", [54]),
        ("apply_rotary_pos_emb", [79, 80]),
    ):
        add(mistral, name, "kv_shared_workspace", lines, "language attention/RoPE workspace")
    clip = "transformers/models/clip/modeling_clip.py"
    add(
        clip,
        "eager_attention_forward",
        "non_kv",
        [266, 269, 272, 273],
        "vision attention; no autoregressive KV cache",
    )
    add(
        clip,
        "forward",
        "non_kv",
        [203, 207, 211, 313, 314, 315, 340, 353, 369, 370, 371, 410, 415],
        "exact vision embeddings/attention/MLP/residual call sites",
    )
    llava = "transformers/models/llava_next/modeling_llava_next.py"
    add(
        llava,
        "forward",
        "non_kv",
        [218, 219, 220, 496, 505, 509, 691],
        "vision projector, image insertion and input embeddings",
    )
    add(
        llava,
        "get_image_features",
        "non_kv",
        [398],
        "vision patch concatenation before language prefill",
    )
    add(
        llava,
        "get_placeholder_mask",
        "non_kv",
        [442, 444, 446],
        "image token shape validation; not cache payload",
    )
    add(
        llava,
        "pack_image_features",
        "non_kv",
        [327, 331, 341, 345, 348],
        "vision-feature concatenation before language prefill",
    )
    add(
        "transformers/activations.py",
        "forward",
        "non_kv",
        [69, 87],
        "exact GELU/QuickGELU implementation",
    )
    add("torch/nn/functional.py", "silu", "non_kv", [2375], "MLP activation")
    add("torch/nn/functional.py", "layer_norm", "non_kv", [2905], "vision LayerNorm")
    add("torch/nn/functional.py", "embedding", "non_kv", [2546], "input embedding lookup")
    mask = "transformers/masking_utils.py"
    add(mask, "eager_mask", "kv_shared_workspace", [504], "language attention mask")
    add(
        mask,
        "sdpa_mask_recent_torch",
        "kv_shared_workspace",
        [362, 369, 370, 375],
        "language attention mask indices/vmap",
    )
    add(mask, "causal_mask_function", "kv_shared_workspace", [75], "causal mask predicate")
    add(mask, "and_mask", "kv_shared_workspace", [49], "combined attention-mask predicate")
    add(mask, "_preprocess_mask_arguments", "kv_shared_workspace", [705], "mask dtype conversion")

    # These local sources are checked against their exact producer Git blobs.
    hf = "mosaickv/backends/hf_runtime.py"
    add(
        hf,
        "_prompt_token_attention_by_head",
        "kv_workspace",
        [1068, 1091, 1093, 1096, 1099],
        "prompt attention evidence pooling",
    )
    add(
        hf,
        "_gather_dense_heads",
        "kv_workspace",
        [5016, 5018, 5022, 5023, 5027],
        "exact cache gather/validity/zero padding",
    )
    add(
        hf,
        "_pack_exact_runtime_blocks",
        "kv_workspace",
        [5128, 5138],
        "cache finite-value verification workspace",
    )
    add(
        hf,
        "generate",
        "non_kv",
        [9374, 11165, 11422],
        "logit finite-value verification and generated token IDs",
    )
    add(hf, "<lambda>", "non_kv", [11416], "final logit validation reduction")
    base = "mosaickv/adapters/huggingface/base.py"
    add(base, "hook", "kv_workspace", [201, 203], "retained prompt attention collector")
    add(base, "_move_model_inputs", "non_kv", [677, 679], "request input/media GPU transfer")
    add(base, "prefill", "kv_workspace", [827, 902], "position and attention-mask metadata")
    add(base, "_decode_cache_position", "kv_workspace", [1127], "logical decode position buffer")
    add(base, "_append_decode_attention_mask", "kv_workspace", [1102], "decode attention mask")
    add(base, "_greedy_token", "non_kv", [1247], "argmax token IDs, not KV state")
    pool = "mosaickv/graph/pooling.py"
    for function, lines in (
        ("_torch_token_feature_matrix", [142]),
        ("_pool_torch_token_feature_batch_device", [188, 190, 196, 197, 198]),
        ("_pool_attention_block_batch", [795, 796, 797, 798, 799]),
        ("_pool_torch_descriptors_one_transfer", [382, 391, 395]),
        ("pool_prompt_attention_coactivation", [854, 875, 902, 904, 940, 942, 943]),
    ):
        add(pool, function, "kv_workspace", lines, "KV evidence-graph descriptor construction")
    forecast = "mosaickv/forecasting/core.py"
    add(forecast, "<genexpr>", "kv_workspace", [501], "forecast prompt query conversion")
    add(forecast, "prepare_samples", "kv_workspace", [500, 510], "forecast query samples")
    add(forecast, "statistics", "kv_workspace", [547, 548], "forecast mean and variance")
    add(
        "mosaickv/measurements/kv_timeline.py",
        "_flush_cuda_events",
        "non_kv",
        [361],
        "explicit one-byte allocator instrumentation tensor",
    )
    return output


def validate_bindings(bindings):
    for path, expected in bindings.items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            raise ValueError(f"source/input version hash mismatch: {path}")


def source_rules(producer_root):
    """Bind reviewed line semantics to exact installed bytes and Git blobs."""
    definitions = rule_definitions()
    bindings, rules, descriptions = {}, [], []
    for definition in definitions:
        relative = definition["file"]
        if relative.startswith("mosaickv/"):
            path = producer_root / "mosaickv/src" / relative
            git_path = "mosaickv/src/" + relative
            original = subprocess.check_output(
                ["git", "show", f"{PRODUCER_SHA}:{git_path}"], cwd=producer_root
            )
            expected = hashlib.sha256(original).hexdigest()
        else:
            path = SITE / relative
            expected = INSTALLED_HASHES[relative]
        if sha(path) != expected:
            raise ValueError(f"reviewed source version mismatch: {path}")
        bindings[str(path)] = expected
        rules.append(
            ("/" + relative, definition["function"], definition["role"], definition["line"])
        )
        lines = path.read_text().splitlines()
        descriptions.append(
            dict(
                definition,
                path=str(path),
                sha256=expected,
                source_line=lines[definition["line"] - 1].strip(),
            )
        )
    return tuple(rules), bindings, descriptions


def validate_capture(report):
    """Reject forged identities/lifetimes even if a producer set pass booleans."""
    rows = report.get("records", ())
    if not rows or len(rows) != report.get("allocation_generations"):
        raise ValueError("complete unique allocation-generation records required")
    count = report.get("events")
    if type(count) is not int or count < 1:
        raise ValueError("allocator event count required")
    identities, by_address = set(), defaultdict(list)
    for row in rows:
        identity = row.get("address"), row.get("generation")
        start, end, size = row.get("start"), row.get("end"), row.get("bytes")
        if (
            identity in identities
            or any(type(x) is not int or x < 0 for x in identity)
            or type(size) is not int
            or size < 0
            or type(start) is not int
            or not -1 <= start < count
            or row.get("role") not in ROLES
            or (end is not None and (type(end) is not int or not start < end < count))
        ):
            raise ValueError("invalid/duplicate allocation generation or lifetime")
        identities.add(identity)
        by_address[row["address"]].append((start, count if end is None else end))
        if row.get("role") != "unknown" and not row.get("evidence"):
            raise ValueError("classified storage requires recorded attribution evidence")
    for intervals in by_address.values():
        intervals.sort()
        if any(a[1] > b[0] for a, b in pairwise(intervals)):
            raise ValueError("pointer-reuse generations overlap")
    if report.get("exact_whole_request_KV_peak_certified") and not valid_whole_request_certificate(
        report
    ):
        raise ValueError("producer's exact ownership certificate is invalid")


def validate_rule_frames(report, rules, bindings):
    """Suffix convenience cannot authorize a same-named unbound source file."""
    bound_paths = {Path(path).resolve() for path in bindings}
    for frames in report["stack_table"]:
        for filename, function, line in frames:
            if (
                any(
                    filename.endswith(suffix) and function == name and line == expected_line
                    for suffix, name, _role, expected_line in rules
                )
                and Path(filename).resolve() not in bound_paths
            ):
                raise ValueError("matching ownership frame is not bound to reviewed source bytes")


def peak_decomposition(report):
    """Report the concurrently live KV roles at the earliest reviewed peak."""
    deltas = defaultdict(lambda: defaultdict(int))
    for row in report["records"]:
        if row["role"] not in KV_ROLES:
            continue
        deltas[row["start"]][row["role"]] += row["bytes"]
        if row["end"] is not None:
            deltas[row["end"]][row["role"]] -= row["bytes"]
    live, peak, sequence, roles = defaultdict(int), 0, None, {}
    for index in sorted(deltas):
        for role, change in deltas[index].items():
            live[role] += change
        if sum(live.values()) > peak:
            peak, sequence, roles = sum(live.values()), index, dict(live)
    if peak != report["known_KV_peak_lower_bound_bytes"]:
        raise ValueError("independent role decomposition differs from reviewed peak")
    return dict(allocator_sequence=sequence, total_bytes=peak, roles=roles)


def unresolved_groups(report):
    groups = {}
    for row in report["records"]:
        if row["role"] != "unknown":
            continue
        frames = report["stack_table"][row["stack_id"]] if "stack_id" in row else []
        fingerprint = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
        item = groups.setdefault(
            fingerprint,
            dict(
                stack_sha256=fingerprint,
                frames=frames,
                allocation_generations=0,
                cumulative_allocated_bytes=0,
                largest_generation_bytes=0,
                examples=[],
            ),
        )
        item["allocation_generations"] += 1
        item["cumulative_allocated_bytes"] += row["bytes"]
        item["largest_generation_bytes"] = max(item["largest_generation_bytes"], row["bytes"])
        if len(item["examples"]) < 3:
            item["examples"].append(
                {
                    key: row.get(key)
                    for key in (
                        "address",
                        "generation",
                        "bytes",
                        "start",
                        "end",
                        "time_us",
                        "end_time_us",
                    )
                }
            )
    return sorted(groups.values(), key=lambda group: -group["largest_generation_bytes"])


def close_report(report, rules):
    validate_capture(report)
    result = review(report, rules=rules)
    # Independent reducer checks every generation and recomputes the overlap
    # peak. Timestamp-rejected marks are never assigned to a guessed generation.
    validate_capture(result)
    certified = valid_whole_request_certificate(result)
    if result["exact_whole_request_KV_peak_certified"] != certified:
        raise ValueError("independent exact ownership certificate disagrees")
    unknown = unresolved_groups(result)
    result.update(
        derived_review_cpu_only=True,
        independent_certificate_verified=certified,
        rejected_timestamp_marks_never_reassigned=True,
        peak_scope="PyTorch requested backing storage, not whole-device memory",
        common_boundary_includes_language_attention_and_algorithm_workspace=True,
        unresolved_stack_groups=unknown,
        peak_role_decomposition=peak_decomposition(result),
    )
    return result


def prepare(root, parent):
    root, parent = Path(root), Path(parent)
    if root.exists():
        raise FileExistsError("new immutable ownership-closure root required")
    source_sha = clean_source(SOURCE)
    parent = parent / "ownership" if (parent / "ownership").is_dir() else parent
    lock = read(parent / "protocol_lock.json")
    if lock["source_sha"] != PRODUCER_SHA:
        raise ValueError("review rules are bound to the inspected producer SHA")
    gate = read(parent / "analysis/GATE.json")
    if gate.get("status") != "passed_capture" or gate.get("rows") != 6 or gate.get("errors"):
        raise ValueError("all six complete captures required")
    summaries = read(parent / "analysis/ownership.json")
    if len(summaries) != 6:
        raise ValueError("six distinct source capture bindings required")
    rules, bindings, descriptions = source_rules(Path(lock["source_root"]))
    paths = []
    for summary in summaries:
        path = Path(summary["allocator_path"])
        path.resolve().relative_to(parent.resolve())
        if str(path) in paths:
            raise ValueError("duplicate capture path")
        paths.append(str(path))
        bindings[str(path)] = summary["allocator_sha256"]
    for path in (
        parent / "protocol_lock.json",
        parent / "analysis/GATE.json",
        parent / "analysis/ownership.json",
        Path(__file__),
    ):
        bindings[str(path)] = sha(path)
    validate_bindings(bindings)
    protocol = dict(
        schema_version=1,
        measurement_type="derived",
        result_category="unified_comparison",
        result_origin="locally_measured",
        source_root=str(SOURCE),
        source_sha=source_sha,
        parent=str(parent),
        paths=paths,
        bindings=bindings,
        rules=rules,
        reviewed_rules=descriptions,
        cpu_only=True,
        new_inference_authorized=False,
        fresh_confirmation_authorized=False,
        raw_artifacts_read_only=True,
        boundary="same allocator/KV workspace definition for every method",
    )
    write_once(root / "protocol_lock.json", protocol)
    return dict(status="prepared", captures=6, protocol_sha256=sha(root / "protocol_lock.json"))


def write_compressed_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as destination,
        gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as stream,
    ):
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def run(root, parent=None):
    root = Path(root)
    lock = read(root / "protocol_lock.json")
    if parent is not None and Path(parent).resolve() not in (
        Path(lock["parent"]).resolve(),
        Path(lock["parent"]).resolve().parent,
    ):
        raise ValueError("parent differs from prepared closure protocol")
    if clean_source(SOURCE) != lock["source_sha"]:
        raise ValueError("ownership closure source SHA changed")
    validate_bindings(lock["bindings"])
    reports, errors = [], []
    for path in lock["paths"]:
        try:
            original = read(path)
            validate_rule_frames(original, lock["rules"], lock["bindings"])
            reviewed = close_report(original, lock["rules"])
            destination = root / "reviewed" / (Path(path).stem + ".json.gz")
            write_compressed_once(destination, reviewed)
            reports.append(
                dict(
                    run_id=original["run_id"],
                    sample_id=original["sample_id"],
                    input_path=path,
                    input_sha256=lock["bindings"][path],
                    reviewed_path=str(destination),
                    reviewed_sha256=sha(destination),
                    certified=reviewed["independent_certificate_verified"],
                    exact_KV_peak_bytes=reviewed["exact_KV_peak_bytes"],
                    lower_bound_bytes=reviewed["known_KV_peak_lower_bound_bytes"],
                    upper_bound_bytes=reviewed["KV_peak_upper_bound_including_unknown_bytes"],
                    unknown_generations=reviewed["unknown_allocation_generations"],
                    blockers=reviewed["blocks_exact_certification"],
                    original_blockers=reviewed["original_blocks_exact_certification"],
                    peak_role_decomposition=reviewed["peak_role_decomposition"],
                    unresolved_groups=reviewed["unresolved_stack_groups"],
                )
            )
        except Exception as exception:
            errors.append(dict(path=path, error=f"{type(exception).__name__}: {exception}"))
    all_certified = len(reports) == 6 and not errors and all(r["certified"] for r in reports)
    gate = dict(
        status="passed_cpu_review" if len(reports) == 6 and not errors else "failed_cpu_review",
        measurement_type="derived",
        result_category=lock["result_category"],
        source_sha=lock["source_sha"],
        reports=len(reports),
        errors=errors,
        all_six_exact_KV_peaks_certified=all_certified,
        new_gpu_inference=False,
        whole_device_memory_certified=False,
        superiority_claim_established=False,
        confirmation_authorized=False,
    )
    write_once(root / "analysis/reports.json", reports)
    write_once(root / "analysis/GATE_ownership_closure.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare" and args.parent is None:
        parser.error("prepare requires --parent")
    result = (
        prepare(args.root, args.parent) if args.mode == "prepare" else run(args.root, args.parent)
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
