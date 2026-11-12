#!/usr/bin/env python3
"""Read-only, fail-closed COCO confirmation-admission report; never a launcher.

Only fixed configuration/gate metadata and explicitly supplied metadata reports
are opened. No dataset, prediction, reference, image, catalog, or scheduler API
is accessed. A positive synthetic/reporting contract is not execution consent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
CANDIDATE_SHA = "389b80ff6529df2072c2a1839469f4be3687be0ec45e38bc124cd834273004cd"
CANDIDATE = SOURCE / "mosaickv/configs/experiments/evicastkv_fox/coco/mosaickv_exact.json"
REVIEW_TEMPLATE = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-v3-discovery-20261109/configs/coco-historical.json"
)
SELECTION_GATES = (
    (
        Path(
            "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-opportunity-development-"
            "reanalysis-v1/analysis/evidence_development/GATE_evidence_development.json"
        ),
        "75dd7e234d78222af232a260e3c7af366374f7b0d1e9a730ad0595abb7423c6f",
    ),
    (
        Path(
            "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-opportunity-followup-r16-"
            "recovery-v1/r16-development/analysis/evidence_development/"
            "GATE_evidence_development.json"
        ),
        "b895e9a40104c2b4fa89bb9936f062f8d1c4be4055a822f06ec74f93616f37c9",
    ),
)
KNOWN_ROOTS = (
    Path("/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109"),
    SELECTION_GATES[0][0].parents[2],
    SELECTION_GATES[1][0].parents[2],
)
FREEZE_NAMES = ("selected_config.json", "frozen_candidate.yaml", "frozen_config.json")
CONTROLS = dict(
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    model_revision="2424fdd47412fccc66d91719126b420e9fbd7065",
    precision="fp16",
    attention_backend="eager",
    backend="huggingface",
    dataset_id="lmms-lab/COCO-Caption",
    dataset_revision="c3213b54e173ae86517d481de17f31439d6f8c72",
    split="val",
    method="mosaickv_exact",
    variant="complete",
    initial_budgets_mib=[20, 40],
    block_size=1,
    maximum_predictions=64,
    continuation_steps=63,
    generated_history="append_only",
)
FROZEN_SCHEMA = "evicastkv_coco_confirmation_freeze_v35"
DISALLOWED_SCOPE_TRANSFER = (
    "FP16_GQA_COCO_FullKV_evaluation_only_boundary_control does not validate a compressed cache",
    "BF16 LLaVA-1.5 full-MHA source-reader evidence does not validate FP16 GQA COCO compression",
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class MetadataReader:
    """Record all opened metadata paths/hashes, including rejected bindings."""

    def __init__(self):
        self.bindings = {}

    def read(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 8 * 2**20:
            raise ValueError("metadata must be a regular JSON file of at most 8 MiB")
        actual = file_sha(path)
        self.bindings[str(path)] = dict(
            actual_sha256=actual,
            expected_sha256=expected,
            hash_verified=expected is None or actual == expected,
        )
        if expected is not None and actual != expected:
            raise ValueError(f"immutable metadata hash mismatch: {path}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("metadata must be a JSON object")
        return value

    def linked(self, record, key):
        binding = record.get(key)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None
        ):
            raise ValueError(f"explicit path and SHA256 binding required: {key}")
        return self.read(binding["path"], binding["sha256"])


def config_controls(config):
    return dict(
        model_id=config["model"]["id"],
        model_revision=config["model"]["revision"],
        precision=config["model"]["precision"],
        attention_backend=config["execution"]["attention_implementation"],
        backend=config["execution"]["backend"],
        dataset_id=config["dataset"]["id"],
        dataset_revision=config["dataset"]["revision"],
        split=config["dataset"]["split"],
        method=config["method"],
        variant="complete",
        initial_budgets_mib=[20, 40],
        block_size=config["cache"]["block_size"],
        maximum_predictions=config["generation"]["max_new_tokens"],
        continuation_steps=config["generation"]["max_new_tokens"] - 1,
        generated_history="append_only",
    )


def gate_matches(gate, frozen):
    return (
        gate.get("complete") is True
        and gate.get("failures") == []
        and gate.get("source_sha") == frozen["source_sha"]
        and gate.get("candidate_config_sha256") == CANDIDATE_SHA
        and gate.get("controls") == CONTROLS
    )


def validate_freeze(frozen, reader):
    """Validate explicit proof links, not a filename or permissive `passed` label."""
    if (
        frozen.get("schema") != FROZEN_SCHEMA
        or frozen.get("frozen") is not True
        or frozen.get("parameters_locked_before_confirmation") is not True
        or frozen.get("confirmation_outcomes_used_for_selection") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(frozen.get("source_sha"))) is None
        or frozen.get("candidate_config_sha256") != CANDIDATE_SHA
        or frozen.get("controls") != CONTROLS
    ):
        raise ValueError("missing immutable original-candidate freeze or changed model/policy")
    selection = reader.linked(frozen, "selection_gate")
    if (
        not gate_matches(selection, frozen)
        or selection.get("positive_development_signal") is not True
    ):
        raise ValueError("frozen candidate lacks a positive registered selection gate")
    numerical = reader.linked(frozen, "numerical_gate")
    if (
        not gate_matches(numerical, frozen)
        or numerical.get("source_scope") != "FP16_GQA_COCO_compressed_exact_selection"
        or numerical.get("full_retention_identity_pass") is not True
        or numerical.get("retained_cache_fidelity_pass") is not True
        or numerical.get("tolerance") != 1e-4
    ):
        raise ValueError("model/source-matched compressed-cache numerical proof is missing")
    storage = reader.linked(frozen, "byte_matching_gate")
    if not gate_matches(storage, frozen) or any(
        storage.get(name) is not True
        for name in (
            "per_sample_initial_byte_match_pass",
            "per_sample_final_byte_match_pass",
            "per_sample_peak_byte_match_pass",
            "physical_backing_padding_preallocation_charged",
            "matched_forward_step_schedule",
        )
    ):
        raise ValueError("model/source-matched per-sample physical-byte proof is missing")
    baseline = reader.linked(frozen, "baseline_fidelity_gate")
    if (
        not gate_matches(baseline, frozen)
        or baseline.get("method") != "vl_cache_reimpl"
        or baseline.get("qualified_reimplementation") is not True
        or baseline.get("fidelity_validated") is not True
    ):
        raise ValueError("qualified matching VL-Cache reimplementation fidelity proof is missing")
    freshness = reader.linked(frozen, "cohort_disjointness_gate")
    if (
        freshness.get("complete") is not True
        or freshness.get("failures") != []
        or freshness.get("metadata_only") is not True
        or freshness.get("candidate_count") != 256
        or freshness.get("candidate_config_sha256") != CANDIDATE_SHA
        or freshness.get("id_disjointness_pass") is not True
        or freshness.get("underlying_media_source_group_disjointness_pass") is not True
        or freshness.get("history_report_sha256") != frozen.get("history_report_sha256")
        or freshness.get("media_report_sha256") != frozen.get("media_report_sha256")
    ):
        raise ValueError(
            "complete hash-bound ID and media/source-group disjointness proof is missing"
        )
    return True


def coverage(record, kind):
    """Subset completion or saved-file hash reduction is not global freshness."""
    shared = record.get("complete") is True and record.get("errors") == []
    if kind == "history":
        return (
            shared
            and record.get("global_historical_inventory_complete") is True
            and record.get("complete_historical_id_inventory") is True
            and record.get("frontier_exhausted") is True
            and record.get("missing_original_worker_indices") == []
        )
    return (
        shared
        and record.get("global_historical_media_inventory_complete") is True
        and record.get("underlying_media_source_group_inventory_complete") is True
        and record.get("unresolved_source_groups") == []
    )


def inspect(root_paths=KNOWN_ROOTS):
    """Inspect exact filenames only, without recursive discovery or data reads."""
    paths, checked = [], []
    for root in root_paths:
        for directory in (
            Path(root),
            Path(root) / "analysis",
            Path(root) / "freeze",
            Path(root) / "configs",
        ):
            for name in FREEZE_NAMES:
                path = directory / name
                checked.append(str(path))
                if path.is_file():
                    paths.append(str(path.resolve()))
    return dict(checked_paths=checked, discovered_paths=sorted(set(paths)))


def evaluate(*, history_report=None, media_report=None, freeze_artifact=None):
    reader, failures = MetadataReader(), []
    verdicts = dict(
        registered_candidate_unchanged=False,
        pinned_negative_selection_evidence_preserved=False,
        global_identity_coverage_complete=False,
        global_media_source_group_coverage_complete=False,
        explicit_frozen_candidate_valid=False,
    )
    facts = dict(
        known_freeze_search=inspect(),
        historical_selection=[],
        candidate_controls=CONTROLS,
        scope_transfer_rejected=list(DISALLOWED_SCOPE_TRANSFER),
    )

    def attempt(name, action):
        try:
            verdicts[name] = action() is True
            if not verdicts[name]:
                failures.append(dict(prerequisite=name, error="required metadata gate is negative"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(dict(prerequisite=name, error=str(error)))

    def candidate():
        current = reader.read(CANDIDATE, CANDIDATE_SHA)
        reviewed = reader.read(REVIEW_TEMPLATE, CANDIDATE_SHA)
        return config_controls(current) == CONTROLS and current == reviewed

    attempt("registered_candidate_unchanged", candidate)

    def selections():
        for path, expected in SELECTION_GATES:
            gate = reader.read(path, expected)
            decisions = gate.get("quality_decisions")
            if not isinstance(decisions, dict) or not decisions:
                raise ValueError("registered negative selection decisions are missing")
            if any(
                decision.get("required_contrasts") != 4
                or decision.get("positive_development_signal") is not False
                for decision in decisions.values()
            ):
                raise ValueError("pinned historical negative decision changed")
            facts["historical_selection"].append(
                dict(
                    path=str(path),
                    sha256=expected,
                    status=gate.get("status"),
                    source_sha=gate.get("source_sha"),
                    quality_decisions=decisions,
                    confirmation_authorized=gate.get("confirmation_authorized"),
                )
            )
        return True

    attempt("pinned_negative_selection_evidence_preserved", selections)
    for name, path, kind in (
        ("global_identity_coverage_complete", history_report, "history"),
        ("global_media_source_group_coverage_complete", media_report, "media"),
    ):
        attempt(
            name, lambda path=path, kind=kind: coverage(reader.read(path), kind) if path else False
        )

    def freeze():
        if freeze_artifact is None:
            raise ValueError(
                "no reviewed frozen candidate supplied; known files are not auto-selected"
            )
        frozen = reader.read(freeze_artifact)
        for name, path in (
            ("history_report_sha256", history_report),
            ("media_report_sha256", media_report),
        ):
            if path is None or frozen.get(name) != file_sha(path):
                raise ValueError("frozen candidate does not bind both supplied coverage reports")
        return validate_freeze(frozen, reader)

    attempt("explicit_frozen_candidate_valid", freeze)
    return dict(
        schema="evicastkv_coco_confirmation_admission_v35",
        status="prerequisites_satisfied_not_execution_authorization"
        if all(verdicts.values())
        else "blocked_confirmation_prerequisites",
        metadata_only=True,
        verdicts=verdicts,
        facts=facts,
        failures=failures,
        evidence_bindings=reader.bindings,
        new_sample_ids_accessed=False,
        dataset_or_prediction_content_accessed=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_claim_allowed=False,
        GPU_jobs_submitted=0,
        jobs_submitted=0,
        prospective_confirmation=dict(
            proposed_samples=256,
            power_calculation_available=False,
            primary_requests_before_timing=1792,
            primary_family_size_retained=6,
            bootstrap_replicates=10000,
            quality_rule="strictly_positive_simultaneous_lower_bound",
            latency_adjusted_upper_ratio_maximum=1.0,
            minimum_memory_reduction_not_registered=True,
            confirmation_runner_currently_blocked=True,
        ),
    )


def write_report(output, report):
    """Both paths are write-once; never overwrite even an incomplete report."""
    output = Path(output)
    sidecar = output.with_name(output.name + ".bindings.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("immutable report or binding sidecar already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sidecar_record = dict(
        report_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        artifact_bindings=report["evidence_bindings"],
        checked_absent_or_present_paths=report["facts"]["known_freeze_search"],
    )
    for path, contents in (
        (sidecar, json.dumps(sidecar_record, indent=2, sort_keys=True) + "\n"),
        (output, rendered),
    ):
        with path.open("x") as stream:
            stream.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-report", type=Path)
    parser.add_argument("--media-report", type=Path)
    parser.add_argument("--freeze-artifact", type=Path)
    args = parser.parse_args()
    report = evaluate(
        history_report=args.history_report,
        media_report=args.media_report,
        freeze_artifact=args.freeze_artifact,
    )
    report["reporting_source_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    report["reporting_worktree_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()
    )
    report["reporting_script_sha256"] = file_sha(Path(__file__))
    write_report(args.output, report)
    print(json.dumps(dict(status=report["status"], output=str(args.output), jobs_submitted=0)))


if __name__ == "__main__":
    main()
