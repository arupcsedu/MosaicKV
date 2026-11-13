#!/usr/bin/env python3
"""Reuse V49 hash proofs and recover only independently attested writer SHAs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import evicastkv_recover_bindings_v49 as prior

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding
from mosaickv.evaluation.parquet_startup_bindings_v50 import validate_startup

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v49/run-20261113-5558403-1")
PARENT_SHA = "6dc34f11a5b87ddceaa300beb3d7fb4d0dbb5cc4acaf1c1632bed4f54d2c6bb8"
PARENT_DECISION_SHA = "2112af8cbef9b0a215226f05b4779db48d4f5b985a9f18c6205e0c2c35fed06c"
MILE = prior.HISTORY / "mosaickv/milebench-isoquality-e6-fdeca17-20261020"
RECOVERY = prior.HISTORY / "mosaickv/milebench-lookm-retry-c4d733b-20261019-v2"
ANALYZER = "mosaickv/scripts/analyze_milebench_shared_bytes.py"
GUARD = "mosaickv/scripts/assert_clean_worktree.sh"
GUARD_SHA = "5522a24655bb8d55e4866759902462c0363bd8258429cc7e03c514d0279dd70e"
STARTUPS = (
    dict(
        job_id="21671480",
        stage="validation",
        output_path=str(MILE / "analysis/validation"),
        source_sha="fdeca177c4a3eaf77df53780b3586a4dc462365c",
        accounting_sha256="de1ca606d980b29e2b90bc61984a65495339b17c1992c7f4a301880e9582db99",
        startup_sha256="bac234c54c838cd9fcc4a4d8af6a611213d6e08b065caccd522b5bead9eb376b",
        writer_sha256="1bf2b4d07c8251b6583c2f2c84c907167b715a5a595ed5c31e6ca6714fed043a",
        stdout_path=str(MILE / "slurm/analysis-21671480.out"),
    ),
    dict(
        job_id="21748934",
        stage="larger",
        output_path=str(MILE / "analysis/larger"),
        source_sha="fdeca177c4a3eaf77df53780b3586a4dc462365c",
        accounting_sha256="802576ed6ebdee0c273685c8e8d7c5025e833ea22ff34a781de774d853c3ff9c",
        startup_sha256="bac234c54c838cd9fcc4a4d8af6a611213d6e08b065caccd522b5bead9eb376b",
        writer_sha256="1bf2b4d07c8251b6583c2f2c84c907167b715a5a595ed5c31e6ca6714fed043a",
        stdout_path=str(MILE / "slurm/analysis-21748934.out"),
    ),
    dict(
        job_id="21610690",
        stage="larger",
        output_path=str(RECOVERY / "analysis/recovered-larger"),
        source_sha="c4d733b7512aa6576ec37ddcbbbf6902e21a8096",
        accounting_sha256="b6fafa6c59035ff44e9fc7b0c89c7da28beeb328f9dec40c3e6e4768a6d3dbc5",
        startup_sha256="705bf42f74d7d643edb16ac44bf044d751a77e8c75c91ed7d8f8a51d5e436d86",
        writer_sha256="7fe1145b7f5ac684c681d687cc81985330f670500f8b96b75daee6d6b2da43b5",
        stdout_path=str(RECOVERY / "slurm/analysis-21610690.out"),
    ),
)
DIRECT_METADATA = (
    prior.HISTORY
    / "evicastkv/request-validation-20261031-v3-single-pass-r1/cohort/analysis/manifest.json"
)
DIRECT = {
    "commit": "e7d7fcda7bca568fb28325fee4982d12088aa936",
    "writer": "mosaickv/scripts/evicastkv_request_validation.py",
    "writer_sha256": "373856c41ea86c49ad84e1b3761687ea91013789f4e824cea71aa92189c43c2a",
    "guard": "mosaickv/scripts/evicastkv_lifetime_history.py",
    "guard_sha256": "5edfa011ae82d12742acb8e5994e708cd232df13df9a106ba60ed7b599fe38a6",
    "metadata_sha256": "0eb57ef0fb6e6eec811e272adeab3f61c4f3814c9cc83c24d2bd9214b1017259",
}


def raw_startup(path, budget):
    path = prior.allowed(path)
    before = prior.geometry(path)
    if before[2] != 55:
        raise ValueError("only the registered 55-byte CPU startup record may be opened")
    budget.authorize(55)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("CPU startup descriptor differs")
        raw = stream.read(55)
        budget.add(len(raw))
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("CPU startup changed during extraction")
    if prior.geometry(path) != before:
        raise ValueError("CPU startup geometry changed")
    return raw


def accounting(job_id, budget):
    maximum = 32 * 2**10
    budget.authorize(maximum + 1)
    raw = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,User,State,ExitCode,WorkDir%2048,SubmitLine%16384",
        ],
        timeout=15,
    )
    budget.add(len(raw))
    if len(raw) > maximum:
        raise ValueError("registered accounting metadata exceeds bounded record size")
    return raw


def source_closure(commit, writer, expected, guard, guard_sha, budget):
    result = {
        writer: prior.git_blob(commit, writer, budget),
        guard: prior.git_blob(commit, guard, budget),
    }
    if result != {writer: expected, guard: guard_sha}:
        raise ValueError("actual historical analyzer/clean-source guard differs")
    return result


def preserved_accounting(raw):
    """Retain the three reviewed public command records for offline verification."""
    lowered = raw.lower()
    if any(
        value in lowered
        for value in (b"hf_token", b"api_key", b"password", b"secret", b"authorization")
    ):
        raise ValueError("credential-bearing command records cannot be persisted")
    return raw.decode("utf-8")


def reused_transaction(row, witness, parent):
    """Derive bounds from a hash-bound, source-validated completed producer proof.

    V49 validated the two complete passes before publishing this witness. Its
    whole-recovery elapsed time is an observed upper bound, NOT a new per-file
    timing measurement. No target body is opened in this continuation.
    """
    if witness["status"] == "binding_transaction_unqualified":
        raise ValueError("an unqualified old transaction cannot be reused")
    transaction = dict(
        path=row["path"],
        signature=row["signature"],
        before_sha256=witness["artifact_sha256"],
        after_sha256=witness["artifact_sha256"],
        bytes_read=2 * row["signature"][2],
        seconds=parent["wall_seconds"],
    )
    original = classify_binding(
        row, transaction, parent["binding_claims"][row["path"]], reviewed_writers=prior.REVIEWED
    )
    if digest(original) != digest(witness):
        raise ValueError("saved completed witness differs from its original classifier")
    return transaction


def promote(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = prior.Budget()
    parent_paths = (
        PARENT / "recovered_bindings.json",
        PARENT / "decision.json",
        prior.PARENT / "inventory.json",
    )
    budget.authorize(sum(p.stat().st_size for p in parent_paths))
    parent = prior.resume.reader.common.history.read(PARENT / "recovered_bindings.json", PARENT_SHA)
    decision = prior.resume.reader.common.history.read(
        PARENT / "decision.json", PARENT_DECISION_SHA
    )
    base_inventory = prior.resume.reader.common.history.read(
        prior.PARENT / "inventory.json", prior.BOUND["inventory.json"]
    )
    budget.add(
        sum(
            p.stat().st_size
            for p in (
                PARENT / "recovered_bindings.json",
                PARENT / "decision.json",
                prior.PARENT / "inventory.json",
            )
        )
    )
    if (
        parent["source_sha"] != "555840326ecfb953a6c5b0e3665189d1bb4f0461"
        or decision["recovered_bindings_sha256"] != PARENT_SHA
        or len(parent["rows"]) != 134
        or parent["newly_resolved_occurrences"] != 0
    ):
        raise ValueError("complete immutable V49 hash-transaction evidence required")
    rows = {r["path"]: r for r in base_inventory["rows"] if r["operation"] == "parquet"}
    registry, attestations = dict(prior.REVIEWED), []
    supplemental = {path: [] for path in rows}
    for spec in STARTUPS:
        selected = {
            k: spec[k]
            for k in (
                "job_id",
                "accounting_sha256",
                "startup_sha256",
                "source_sha",
                "output_path",
                "stage",
            )
        }
        accounting_raw = accounting(spec["job_id"], budget)
        startup_raw = raw_startup(Path(spec["stdout_path"]), budget)
        facts = validate_startup(accounting_raw, startup_raw, selected)
        blobs = source_closure(
            spec["source_sha"], ANALYZER, spec["writer_sha256"], GUARD, GUARD_SHA, budget
        )
        record = dict(
            facts=facts,
            original_accounting_record=preserved_accounting(accounting_raw),
            original_startup_record=startup_raw.decode("ascii"),
            source_blobs=blobs,
            startup_path=spec["stdout_path"],
            binding_kind="derived_startup_accounting_and_original_artifact_chain",
        )
        att = root / "attestations" / f"{spec['job_id']}.json"
        write_once(att, record)
        attestations.append(dict(path=str(att), sha256=sha(att)))
        registry[(spec["source_sha"], ANALYZER)] = spec["writer_sha256"]
        report = str(Path(spec["output_path"]) / "report.json")
        for path, claims in parent["binding_claims"].items():
            matching = [c for c in claims if c["metadata_path"] == report]
            for claim in matching:
                supplemental[path].append(
                    dict(
                        metadata_path=str(att),
                        metadata_sha256=sha(att),
                        artifact_sha256=claim["artifact_sha256"],
                        role="runtime_writer",
                        recorded_source_sha=spec["source_sha"],
                        source_blob_sha256=spec["writer_sha256"],
                        writer_path=ANALYZER,
                        source_evidence_verified=True,
                    )
                )
    blobs = source_closure(
        DIRECT["commit"],
        DIRECT["writer"],
        DIRECT["writer_sha256"],
        DIRECT["guard"],
        DIRECT["guard_sha256"],
        budget,
    )
    helper = "mosaickv/src/mosaickv/evaluation/fox_campaign.py"
    blobs[helper] = prior.git_blob(DIRECT["commit"], helper, budget)
    if blobs[helper] != "4fa92bd9908d48ab6b040e09475d87a982521a1e934532bc5bca4e615ce0d421":
        raise ValueError("original request-validation clean-source implementation differs")
    registry[(DIRECT["commit"], DIRECT["writer"])] = DIRECT["writer_sha256"]
    for path, claims in parent["binding_claims"].items():
        for claim in claims:
            if claim["metadata_path"] == str(DIRECT_METADATA):
                if (
                    claim["metadata_sha256"] != DIRECT["metadata_sha256"]
                    or claim["recorded_source_sha"] != DIRECT["commit"]
                ):
                    raise ValueError("exact original request-validation source/hash map differs")
                supplemental[path].append(
                    dict(
                        claim,
                        role="runtime_writer",
                        writer_path=DIRECT["writer"],
                        source_blob_sha256=blobs[DIRECT["writer"]],
                        source_evidence_verified=True,
                    )
                )
    results, promoted = [], []
    for old in parent["rows"]:
        row = rows[old["path"]]
        # Stat-only current-file check, never another independent body read.
        actual_path = prior.allowed(row["path"])
        if str(actual_path) != row["path"] or prior.geometry(actual_path) != row["signature"]:
            raise ValueError("previously proved source changed; body replay is not authorized")
        transaction = reused_transaction(row, old, parent)
        new = classify_binding(
            row,
            transaction,
            parent["binding_claims"][row["path"]] + supplemental[row["path"]],
            reviewed_writers=registry,
        )
        if new["status"] != old["status"]:
            promoted.append(row["path"])
        results.append(new)
    result = dict(
        source_sha=revision,
        parent_report_sha256=PARENT_SHA,
        parent_decision_sha256=PARENT_DECISION_SHA,
        rows=results,
        supplemental_claims=supplemental,
        startup_attestations=attestations,
        direct_writer_source_closure=blobs,
        status_counts=dict(Counter(r["status"] for r in results)),
        promoted_paths=promoted,
        reused_hash_transactions=len(results),
        reused_timing_kind="V49_whole_recovery_observed_upper_bound_not_new_per_file_measurement",
        original_occurrence_counts=parent["original_occurrence_counts"],
        newly_resolved_occurrences=0,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs=0,
        historical_media_exposure_resolved=False,
    )
    prior.projected_bound(result)
    prior.resume.require_artifact_capacity(root, prior.LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"],
        promoted=len(promoted),
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "recovered_bindings.json"
    if not path.exists():
        return dict(
            status="startup_recovery_failed_or_incomplete",
            source_sha=revision,
            fresh_confirmation_authorized=False,
            GPU_jobs=0,
        )
    result = prior.resume.reader.common.history.read(path)
    if result["source_sha"] != revision or result["parquet_bodies_reopened"] != 0:
        raise ValueError("startup recovery must reuse prior source-body proofs")
    return dict(
        source_sha=revision,
        report_sha256=sha(path),
        status_counts=result["status_counts"],
        reused_hash_transactions=result["reused_hash_transactions"],
        promoted_paths=result["promoted_paths"],
        newly_resolved_occurrences=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "promote", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only immutable output required")
    try:
        result = prior.resume.cpu(root) if args.mode == "cpu" else globals()[args.mode](root)
        if args.mode == "analyze":
            write_once(root / "decision.json", result)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="startup provenance recovery failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
