#!/usr/bin/env python3
"""Build immutable vLLM jobs from completed, hash-verified HF byte-cap arms.

Never edits HF inputs/results or updates the shared Python environment.
This mirrors serving projections, not official per-layer/head algorithms.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import yaml

from mosaickv.backends.vllm_runtime import _resolve_model_source, _vllm_kv_cache_geometry
from mosaickv.config import load_config
from mosaickv.types import Precision

BUILD = runpy.run_path(str(Path(__file__).with_name("build_vllm_selector_plan.py")))[
    "build_catalog"
]
METHODS = {
    "prompt_attention_topk",
    "mosaickv_exact",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "random_kv",
}
TASKS = {"mmstar", "textvqa", "docvqa", "coco_caption"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable output conflict: {path}")
        return
    with path.open("x") as stream:
        stream.write(encoded)


def read_bound(path: str | Path, expected: str) -> dict:
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(path.read_text())


def controls(config: dict) -> dict:
    """Compare all shared HF controls; selector internals intentionally differ."""
    return {
        "model": config["model"],
        "dataset": config["dataset"],
        "generation": config["generation"],
        "execution": config["execution"],
        "cache": {
            key: config["cache"][key]
            for key in (
                "budget_unit",
                "budget_value",
                "block_size",
                "physical_budget_block_size",
                "physical_budget_mandatory_policy",
                "packing_mode",
                "exact_gather_mode",
            )
        },
    }


def development_source_binding(attestation: Path | None, source_root: Path, stage: str) -> dict:
    """Permit exploratory HF sources only from the same preserved snapshot.

    This is not an exception to the canonical campaign's clean-source gate.
    Both the file inventory and the git dirty-patch fingerprint are required.
    """
    if attestation is None or stage != "preflight":
        raise ValueError("dirty HF sources require an attested development preflight")
    from mosaickv.manifest import _worktree_fingerprint

    inventory = json.loads(attestation.read_text())
    if Path(inventory["source_root"]).resolve() != source_root.resolve():
        raise ValueError("HF development source belongs to another snapshot")
    if not inventory.get("files") or inventory.get("patch_sha") != _worktree_fingerprint(
        source_root
    ):
        raise ValueError("HF development source fingerprint differs")
    for name, expected in inventory["files"].items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root.resolve()) or sha(path) != expected:
            raise ValueError("HF development source inventory differs")
    return inventory


def project_backend_config(config: dict) -> dict:
    """Drop HF-only decode transport, not the recorded source-selector policy.

    The vLLM transaction replays prefill selections and then appends new tokens;
    it cannot execute PrefixKV's per-layer decode eviction. The caller records
    both policies separately. Never relabel this as official PrefixKV.
    """
    config = copy.deepcopy(config)
    if config["method"] == "prefixkv_reimpl":
        config["prefixkv"].update(
            decode_policy="realized_ratio", hf_cache_transport="common_guarded"
        )
    return config


def prepare(
    completion_path: Path,
    output: Path,
    *,
    cap: int,
    sample_limit: int | None,
    stage: str,
    source_root: Path,
    attestation: Path | None,
    preflight_report: Path | None = None,
    allow_development_hf: bool = False,
) -> dict:
    completion = json.loads(completion_path.read_text())
    development = (
        development_source_binding(attestation, source_root, stage)
        if allow_development_hf else None
    )
    if completion.get("status") != "completed" or (
        completion.get("source_git_dirty") and development is None
    ):
        raise ValueError("only clean, completed HF source shards may be mirrored")
    task = completion.get("task")
    if task not in TASKS:
        raise ValueError(f"no validated vLLM scoring route for {task}; do not substitute metrics")
    if "job_spec" in completion:
        job = read_bound(completion["job_spec"], completion["job_spec_sha256"])
    elif completion.get("classification") == "byte_capped_published_baseline_pilot":
        if completion.get("configured_byte_cap") != cap:
            raise ValueError("HF published-baseline completion uses another byte cap")
        job = {
            "sample_ledger_path": completion["sample_ledger"],
            "sample_ledger_sha256": completion["sample_ledger_sha256"],
        }
    else:
        raise ValueError("HF completion has no hash-bound source job/ledger contract")
    ledger = read_bound(job["sample_ledger_path"], job["sample_ledger_sha256"])
    if ledger["task"] != task:
        raise ValueError("HF ledger task mismatch")
    sample_ids = ledger["sample_ids"]
    if sample_limit is not None:
        if sample_limit < 2:
            raise ValueError("at least two samples are required for A-B-A isolation")
        sample_ids = sample_ids[:sample_limit]
    if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("need at least two unique paired sample identities")
    if stage == "campaign" and attestation is not None:
        raise ValueError("development snapshots cannot produce canonical campaign jobs")
    source_sha = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if stage == "campaign":
        if preflight_report is None:
            raise ValueError("campaign requires a passed canonical preflight report")
        gate = json.loads(preflight_report.read_text())
        if (
            gate.get("status") != "passed"
            or gate.get("stage") != "preflight"
            or gate.get("classification") != "canonical_vllm_projection"
            or gate.get("source_git_sha") != source_sha
            or gate.get("task") != task
            or gate.get("telemetry_checks_passed") is not True
        ):
            raise ValueError("preflight does not authorize this task/source campaign")
    arms = [
        record
        for record in completion["records"]
        if record["method"] in METHODS and record.get("budget_bytes", cap) == cap
    ]
    if not {"prompt_attention_topk", "mosaickv_exact"}.issubset(
        {record["method"] for record in arms}
    ):
        raise ValueError("matched Prompt Top-K and MosaicKV HF arms are required")
    if len({record["method"] for record in arms}) != len(arms):
        raise ValueError("duplicate HF method/cap arms")
    manifests = []
    for record in arms:
        source_rows = record["rows"]
        if {row["sample_id"] for row in source_rows} != set(ledger["sample_ids"]):
            raise ValueError("HF arm sample IDs differ from ledger")
        if any(row["status"] != "completed" for row in source_rows):
            raise ValueError("failed HF arm cannot be silently reused")
        manifest_path = Path(source_rows[0]["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"]["git_dirty"] and development is None:
            raise ValueError("HF manifest is not clean")
        if development is not None and (
            manifest["source"]["git_sha"] != source_sha
            or manifest["source"]["patch_sha"] != development["patch_sha"]
            or manifest["source"]["canonical_eligible"]
        ):
            raise ValueError("HF manifest differs from exploratory source attestation")
        if manifest["resolved_config"]["cache"]["budget_value"] != cap:
            raise ValueError("HF source arm is not at requested physical cap")
        raw = manifest_path.with_name("results.jsonl")
        if sha(raw) != manifest["artifacts"]["raw_output_sha"]:
            raise ValueError("HF prediction artifact was modified")
        trace_root = completion_path.parent / "traces" / record["run_id"]
        digest = hashlib.sha256()
        for path in sorted(trace_root.glob("*.json")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        if digest.hexdigest() != manifest["artifacts"]["log_sha"]:
            raise ValueError("HF source traces were modified")
        manifests.append((record, manifest_path, manifest, trace_root))
    shared = controls(manifests[0][2]["resolved_config"])
    if any(controls(manifest["resolved_config"]) != shared for _, _, manifest, _ in manifests):
        raise ValueError("HF arms differ in shared comparison controls")
    gate_binding = {
        "model": shared["model"],
        "dataset": shared["dataset"],
        "generation": shared["generation"],
        "execution": shared["execution"],
        "serving_block_size": 16,
        "source_selector_commits": {
            record["method"]: manifest["source"]["git_sha"] for record, _, manifest, _ in manifests
        },
    }
    if stage == "campaign" and gate.get("gate_binding") != gate_binding:
        raise ValueError("preflight model/data/generation/method/source binding differs")
    # Local model metadata only: no weights, downloads, package installation, or CUDA.
    from transformers import AutoConfig

    model = shared["model"]
    model_source = _resolve_model_source(model["id"], model["revision"], local_files_only=True)
    hf_config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    geometry = _vllm_kv_cache_geometry(hf_config, Precision(model["precision"]))
    derived = copy.deepcopy(ledger)
    derived.update(
        sample_ids=sample_ids,
        sample_count=len(sample_ids),
        parent_ledger_sha256=job["sample_ledger_sha256"],
        sample_use="fixed_hf_identity_backend_verification_not_untouched",
    )
    ledger_path = output / "inputs/ledger.json"
    write_once(ledger_path, derived)
    specs = []
    if stage == "preflight":
        # FullKV is solely a correctness/quality control; never a latency competitor.
        base = next(item for item in manifests if item[0]["method"] == "mosaickv_exact")
        reference = next(item for item in manifests if item[0]["method"] == "prompt_attention_topk")
        arm_inputs = [("full_kv", reference), ("native_identity", base)]
    else:
        arm_inputs = []
    arm_inputs += [(item[0]["method"], item) for item in manifests]
    for name, (_, manifest_path, manifest, trace_root) in arm_inputs:
        config = project_backend_config(manifest["resolved_config"])
        config["execution"]["backend"] = "vllm"
        config["execution"]["attention_implementation"] = "eager"
        config["execution"]["decode_attention_implementation"] = "eager"
        config["cache"].update(
            block_size=16,
            physical_budget_block_size=16,
            physical_budget_mandatory_policy="last_token",
        )
        config["prototypes"]["enabled"] = False
        config["residual"]["enabled"] = False
        config["repair"].update(enabled=False, policy="none", max_blocks_per_step=0)
        config["cache"]["total_storage_budget_bytes"] = None
        catalog = None
        if name == "full_kv":
            config["method"] = "full_kv"
            config["cache"].update(retention_ratio=1.0, budget_value=2**63 - 1)
        elif name == "native_identity":
            config["cache"].update(retention_ratio=1.0, budget_value=2**63 - 1)
            catalog = BUILD(
                manifest_path,
                trace_root,
                task=task,
                target_block_size=16,
                target_retention_ratio=1.0,
            )
        else:
            catalog = BUILD(
                manifest_path,
                trace_root,
                task=task,
                target_block_size=16,
                physical_byte_cap=cap,
                bytes_per_position=geometry.bytes_per_position,
            )
        if catalog is not None:
            catalog["entries"] = [
                entry for entry in catalog["entries"] if entry["sample_id"] in sample_ids
            ]
            if {entry["sample_id"] for entry in catalog["entries"]} != set(sample_ids):
                raise ValueError("selector catalog is missing requested sample IDs")
            catalog_path = output / f"inputs/{name}.plan.json"
            write_once(catalog_path, catalog)
        else:
            catalog_path = None
        config_path = output / f"inputs/{name}.yaml"
        payload = yaml.safe_dump(config, sort_keys=True)
        if config_path.exists() and config_path.read_text() != payload:
            raise ValueError("resolved vLLM configuration changed")
        if not config_path.exists():
            with config_path.open("x") as stream:
                stream.write(payload)
        load_config(config_path)  # Strict CPU-only validation before submission.
        specs.append(
            {
                "name": name,
                "config": str(config_path),
                "config_sha256": sha(config_path),
                "catalog": str(catalog_path) if catalog_path else None,
                "catalog_sha256": sha(catalog_path) if catalog_path else None,
                "hf_manifest": str(manifest_path),
                "hf_manifest_sha256": sha(manifest_path),
                "source_decode_policy": manifest["resolved_config"].get("prefixkv", {}).get(
                    "decode_policy"
                ) if name == "prefixkv_reimpl" else "append_only",
                "serving_decode_policy": "append_only_requestwide_projection",
            }
        )
    spec = {
        "schema_version": 1,
        "stage": stage,
        "task": task,
        "physical_byte_cap": cap,
        "hf_completion": str(completion_path),
        "hf_completion_sha256": sha(completion_path),
        "source_root": str(source_root),
        "source_git_sha": source_sha,
        "gate_binding": gate_binding,
        "preflight_report": str(preflight_report) if preflight_report else None,
        "preflight_report_sha256": sha(preflight_report) if preflight_report else None,
        "sample_ids": sample_ids,
        "ledger": str(ledger_path),
        "ledger_sha256": sha(ledger_path),
        "source_attestation": str(attestation) if attestation else None,
        "source_attestation_sha256": sha(attestation) if attestation else None,
        "classification": "development_only" if attestation else "canonical_vllm_projection",
        "output_root": str(output),
        "arms": specs,
        "shared_controls": shared,
        "serving_block_size": 16,
        "serving_mandatory_policy": "first_and_last_whole_block",
        "hf_source_mandatory_policy": shared["cache"]["physical_budget_mandatory_policy"],
        "cross_backend_identical_layout_claim": False,
        "fullkv_latency_excluded": True,
        "timing_scope": "offline_hf_selector_replay_plus_engine_timing_not_online_native_selector",
    }
    write_once(output / "job.json", spec)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-completion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-mib", type=int, required=True)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--stage", choices=("preflight", "campaign"), default="preflight")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--development-attestation", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.hf_completion.resolve(),
        args.output_root.resolve(),
        cap=args.cap_mib * 2**20,
        sample_limit=args.sample_limit,
        stage=args.stage,
        source_root=args.source_root.resolve(),
        attestation=args.development_attestation,
        preflight_report=args.preflight_report,
    )
    print(
        json.dumps(
            {
                "job": str(args.output_root / "job.json"),
                "sha256": sha(args.output_root / "job.json"),
                "stage": result["stage"],
                "sample_count": len(result["sample_ids"]),
            }
        )
    )


if __name__ == "__main__":
    main()
