#!/usr/bin/env python3
"""Write-once CPU review of saved allocator records; never launches inference."""

import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.request_ownership import STACK_RULES
from mosaickv.measurements.kv_timeline import valid_whole_request_certificate
from mosaickv.measurements.ownership_review import RULES, SHARED_ATTENTION_RULE, review

SOURCE = Path(__file__).resolve().parents[2]
ARMS = {"storage_compact", "shared_source_prototypes"}


def read(path):
    return json.loads(path.read_text())


def source_bindings(producer, versions):
    """Bind reviewed functions to immutable repository blobs or wheel RECORD."""
    result = {}
    for suffix in sorted({rule[0] for rule in (*RULES, *STACK_RULES, SHARED_ATTENTION_RULE)}):
        if suffix.startswith("/mosaickv/"):
            relative = "mosaickv/src" + suffix
            blob = subprocess.check_output(["git", "show", f"{producer}:{relative}"], cwd=SOURCE)
            expected = hashlib.sha256(blob).hexdigest()
            path = SOURCE / relative
            if sha(path) != expected:
                raise ValueError("review cannot change the executed source: " + relative)
            result[str(path)] = dict(sha256=expected, provenance=f"git:{producer}:{relative}")
        else:
            package = suffix.split("/")[1]
            distribution = importlib.metadata.distribution(package)
            if distribution.version != versions[package]:
                raise ValueError("installed package differs from producer manifest: " + package)
            relative = suffix.removeprefix("/")
            record = next((f for f in distribution.files if str(f) == relative), None)
            if record is None or record.hash is None or record.hash.mode != "sha256":
                raise ValueError("reviewed upstream source requires wheel RECORD: " + relative)
            path = Path(distribution.locate_file(record))
            actual = sha(path)
            recorded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
            if recorded != record.hash.value:
                raise ValueError("upstream source differs from installed wheel RECORD: " + relative)
            result[str(path)] = dict(
                sha256=actual,
                provenance=f"{package}=={distribution.version}:wheel_RECORD",
                capture_time_extra_hash_not_available=True,
            )
    return result


def summarize(root, output, expected):
    review_sha = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("new immutable derived-review directory required")
    if sha(root / "plan.json") != expected:
        raise ValueError("producer plan hash changed")
    p = read(root / "plan.json")
    bindings = dict(p["bound_files"])
    for name, digest in bindings.items():
        if sha(Path(name)) != digest:
            raise ValueError("producer-bound artifact changed: " + name)
    versions = read(root / "workers/0/manifest.json")["environment"]["packages"]
    sources = source_bindings(p["source_sha"], versions)
    # Bind complete original records, not the superseded producer analysis.
    for i in (0, 1):
        for name in (
            "raw.jsonl",
            "result.json",
            "manifest.json",
            "storage_compact-audit.logits.pt",
            "shared_source_prototypes-audit.logits.pt",
        ):
            path = root / f"workers/{i}" / name
            bindings[str(path)] = sha(path)
    bindings[str(root / "plan.json")] = expected
    config = dict(
        producer_source_sha=p["source_sha"],
        producer_plan_sha=expected,
        reviewer_source_sha=review_sha,
        source_bindings=sources,
        bound_files=bindings,
        measurement_type="derived",
        result_category="unified_comparison",
        result_origin="locally_measured",
        retrospective_consumed_engineering_diagnostic=True,
        numerical_thresholds_unchanged=True,
        no_GPU_inference=True,
        boundary="requested_PyTorch_KV_storage_plus_declared_KV_and_shared_attention_workspace",
        exclusions=[
            "CPU_residuals_reported_separately",
            "allocator_rounding_and_reservation",
            "non_PyTorch_vendor_allocations",
            "model_weights_and_non_KV_activations",
        ],
        source_identity_limitation="Additional upstream files are wheel-RECORD-verified "
        "at CPU review; the producer pinned package versions and Llama source, "
        "but not every extra file hash at capture.",
        prototype_primitive_not_six_component_planner=True,
    )
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    write_once(output / "review_manifest.json", dict(config, config_sha=config_sha))
    audits, canaries, failures = [], [], []
    for i in (0, 1):
        manifest = read(root / f"workers/{i}/manifest.json")
        if manifest["source_dirty"] or manifest["git_sha"] != p["source_sha"]:
            raise ValueError("audit producer source identity mismatch")
        if read(root / f"workers/{i}/result.json")["status"] != "completed":
            failures.append(dict(input=i, error="producer worker status is not completed"))
        receipt = read(Path(p["reuse_canaries"]) / f"{i}.json")
        for raw in receipt["rows"]:
            canaries.append(
                dict(
                    input=i,
                    arm=raw["arm"],
                    phase=raw["phase"],
                    status=raw["status"],
                    numerically_admissible=raw["numerically_admissible"],
                    fidelity=raw["fidelity"],
                    producer_source_sha=receipt["producer_source_sha"],
                )
            )
        if not receipt["native"]["passed"]:
            failures.append(dict(input=i, error="native projection/repair control failed"))
        records = [
            json.loads(line) for line in (root / f"workers/{i}/raw.jsonl").read_text().splitlines()
        ]
        if len(records) != 2 or {r["arm"] for r in records} != ARMS:
            failures.append(dict(input=i, error="missing or duplicate audit arms"))
        for raw in records:
            if raw["status"] != "completed":
                failures.append(dict(input=i, arm=raw["arm"], error=raw.get("error", "failed")))
                continue
            ledger = review(raw["ownership_timeline"])
            certificate = output / f"{i}-{raw['arm']}.ledger.json.gz"
            with gzip.open(certificate, "xt", compresslevel=1) as stream:
                json.dump(ledger, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            receipts = ledger["all_head_source_retirement_receipts"]
            retired = len(receipts) == 32 and all(r["retired_before_decode"] for r in receipts)
            row = dict(
                input=i,
                sample_id=raw["sample_id"],
                arm=raw["arm"],
                status=raw["status"],
                producer_manifest=str(root / f"workers/{i}/manifest.json"),
                reviewer_manifest=str(output / "review_manifest.json"),
                producer_source_sha=p["source_sha"],
                reviewer_source_sha=review_sha,
                ledger_path=str(certificate),
                ledger_sha256=sha(certificate),
                initial_GPU_KV_bytes=raw["detail"]["initial_encoded_bytes"],
                final_GPU_KV_bytes=raw["detail"]["final_encoded_bytes"],
                whole_request_KV_plus_workspace_peak_bytes=ledger["exact_KV_peak_bytes"],
                CPU_residual_bytes=raw["mechanisms"].get("CPU_residual_bytes", 0),
                source_generations_retired_before_decode=sum(
                    r["retired_before_decode"] for r in receipts
                ),
                source_retirement_passed=retired
                if raw["arm"] == "shared_source_prototypes"
                else None,
                whole_request_certificate_valid=valid_whole_request_certificate(ledger),
                native_snapshot_identity_verified=ledger.get(
                    "native_segment_snapshot_identity_verified", False
                ),
                unknown_generations=ledger["unknown_allocation_generations"],
                original_blockers=ledger["original_blocks_exact_certification"],
                blockers=ledger["blocks_exact_certification"],
                fidelity=raw["fidelity"],
                numerically_admissible=raw["numerically_admissible"],
                audit_wall_seconds_NOT_clean_latency=raw["request_wall_seconds"],
            )
            audits.append(row)
    numerical = len(canaries) == 4 and all(
        r["status"] == "completed" and r["numerically_admissible"] for r in canaries
    )
    numerical &= len(audits) == 4 and all(r["numerically_admissible"] for r in audits)
    certificates = len(audits) == 4 and all(r["whole_request_certificate_valid"] for r in audits)
    retirement = len([r for r in audits if r["source_retirement_passed"] is True]) == 2
    comparisons = []
    for i in (0, 1):
        paired = {r["arm"]: r for r in audits if r["input"] == i}
        if set(paired) != ARMS:
            continue
        base, candidate = paired["storage_compact"], paired["shared_source_prototypes"]
        keys = (
            "initial_GPU_KV_bytes",
            "final_GPU_KV_bytes",
            "whole_request_KV_plus_workspace_peak_bytes",
        )
        reductions = {
            k: 1 - candidate[k] / base[k] if candidate[k] is not None and base[k] else None
            for k in keys
        }
        comparisons.append(
            dict(
                input=i,
                GPU_reductions=reductions,
                final_GPU_plus_CPU_residual_reduction=1
                - (candidate["final_GPU_KV_bytes"] + candidate["CPU_residual_bytes"])
                / base["final_GPU_KV_bytes"],
            )
        )
    for name, digest in bindings.items():
        if sha(Path(name)) != digest:
            raise ValueError("producer artifacts changed during CPU review: " + name)
    clean_source(SOURCE, review_sha)
    result = dict(
        complete=len(audits) == 4 and len(canaries) == 4 and not failures,
        failures=failures,
        numerical_gate_passed=bool(numerical),
        actual_all_head_source_retirement_passed=retirement,
        whole_request_certificates_passed=certificates,
        all_whole_request_peaks_smaller=len(comparisons) == 2
        and all(
            (r["GPU_reductions"]["whole_request_KV_plus_workspace_peak_bytes"] or 0) > 0
            for r in comparisons
        ),
        independent_quality_established=False,
        strict_latency_advantage_established=False,
        six_component_superiority_established=False,
        larger_campaign_authorized=False,
        source_identity_limitation=config["source_identity_limitation"],
        audits=audits,
        reused_canaries=canaries,
        comparisons=comparisons,
        config_sha=config_sha,
        reviewer_source_sha=review_sha,
    )
    write_once(output / "decision.json", result)
    lines = [
        "# All-head source replacement and whole-request KV ownership review",
        "",
        "CPU-derived, retrospective diagnostic on TWO CONSUMED COCO inputs; "
        "not paper confirmation.",
        "",
        f"Complete: {result['complete']}; numerical controls: {bool(numerical)}; "
        f"actual source retirement: {retirement}; "
        f"classified interval certificates: {certificates}.",
        "",
        "| Input | Variant | Initial GPU KV MiB | Final GPU KV MiB | "
        "Whole-request KV + workspace peak MiB | CPU residual MiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in audits:
        keys = (
            "initial_GPU_KV_bytes",
            "final_GPU_KV_bytes",
            "whole_request_KV_plus_workspace_peak_bytes",
            "CPU_residual_bytes",
        )
        sizes = [f"{row[k] / 2**20:.6f}" if row[k] is not None else "uncertified" for k in keys]
        lines.append(f"| {row['input']} | {row['arm']} | " + " | ".join(sizes) + " |")
    lines += [
        "",
        "Reductions relative to the paired storage-only control:",
        "",
        "| Input | Initial GPU | Final GPU | Whole-request KV + workspace peak | "
        "Final GPU + CPU residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        values = [
            *comparison["GPU_reductions"].values(),
            comparison["final_GPU_plus_CPU_residual_reduction"],
        ]
        values = [f"{100 * value:.4f}%" if value is not None else "uncertified" for value in values]
        lines.append(f"| {comparison['input']} | " + " | ".join(values) + " |")
    all_tokens_equal = len(audits) == 4 and all(
        row["fidelity"].get("all_128_tokens_match") is True for row in audits
    )
    lines += [
        "",
        "## Interpretation",
        "",
        "The prototype owns a physically smaller shared source across all 32 K and 32 V heads, "
        "not an additive single-head overlay. The actual retirement verdict is shown above, "
        f"and the all-128-greedy-token agreement verdict is {all_tokens_equal}. First-32 "
        "KL/logit thresholds remain unchanged. "
        "Controlled native all-head repair evidence is reused; "
        "natural model-level repair and the original six-component planner were NOT validated.",
        "",
        "The complete timeline charges persistent KV, reconstruction, "
        "temporary replacement owners, "
        "and declared shared workspaces, including language attention scores/probabilities. "
        "Allocation aliases count once; free-requested storage remains live until completed. "
        "Exactness concerns requested PyTorch tensor-storage bytes, not rounded/reserved allocator "
        "blocks, process HBM, non-PyTorch vendor workspace, "
        "or the separately listed CPU residuals.",
        "",
        "Use the measured per-input reductions above, including negative values. "
        "Initial/final GPU packet savings cannot establish a whole-request or GPU-plus-CPU win. "
        "Audit wall times include instrumentation "
        "and are NOT clean latency measurements.",
        "",
        "Rejected ambiguous timestamp marks are not used to assign allocation identities. "
        "Independent, version-bound source-stack roles resolve them on the original lifetimes. "
        "Original blockers and raw negative reports are preserved; unknowns and lifecycle errors "
        "continue to block certification. " + config["source_identity_limitation"],
        "",
        "The reused native operator's older mixed-unit peak certificate "
        "is superseded, not promoted. "
        "Only its actual projection/repair numerical evidence is reused. The producer analyzer "
        "counted reused canaries too late; this review counts "
        "all eight rows before completeness checks.",
        "",
        "Validated storage-only defaults, main worktree, paper files, and historical raw results "
        "are unchanged. No superiority, independent benchmark-quality, "
        "or latency claim is established.",
    ]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("audits", "reused_canaries")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    summarize(args.root.resolve(), args.output.resolve(), args.plan_sha)
