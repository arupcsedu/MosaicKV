#!/usr/bin/env python3
"""Retry exactly one alias-mismatched derived proof after V41 reconciliation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import evicastkv_derived_alias_recovery_v42 as correction
import evicastkv_global_history_v36 as history
from evicastkv_occurrence_recovery_v40 import checked_roots
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation import metadata_recovery_v41 as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v41/run-20261112-1")
PARENT_SOURCE = "d37fc8daf75af02d66ce35cf76abe0d7c3a5446b"


def cpu(root):
    import os

    revision = clean_source(SOURCE)
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    from evicastkv_review_closure_v2 import cpu_suite

    cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return cpu_gate(root / "cpu", revision)


def recover(root):
    revision = clean_source(SOURCE)
    cpu_gate(root / "cpu", revision)
    gate = history.read(PARENT / "decision.json")
    if gate["source_sha"] != PARENT_SOURCE or gate["fresh_confirmation_authorized"] is not False:
        raise ValueError("exact immutable V41 analysis is required")
    parent = history.read(PARENT / "reconciled_history.json", gate["reconciled_history_sha256"])
    plan = history.read(PARENT / "registration.json")
    if plan["source_sha"] != PARENT_SOURCE:
        raise ValueError("V41 registration source differs")
    selected = [row for row in plan["rows"] if row["path"] == str(correction.ORIGINAL)]
    if len(selected) != 1:
        raise ValueError("only the exact original V6 occurrence is registered")
    row = selected[0]
    unresolved = {
        (r["owner_index"], r["historical_error_ordinal"]): r
        for r in parent["unresolved_occurrences"]
    }
    if (row["owner_index"], row["historical_error_ordinal"]) not in unresolved:
        result = dict(
            source_sha=revision,
            status="already_resolved_no_retry",
            fresh_confirmation_authorized=False,
        )
        write_once(root / "decision.json", result)
        return result
    proof = correction.prove(row, checked_roots(), time.monotonic() + 210)
    path = root / "proof.json"
    write_once(path, proof)
    before = proof.get("whole_file_sha256_before", proof.get("full_file_sha256_before"))
    after = proof.get("whole_file_sha256_after", proof.get("full_file_sha256_after"))
    if proof["path"] != row["path"] or digest(proof["signature"]) != digest(row["signature"]):
        raise ValueError("corrected proof differs from its original occurrence")
    receipt = dict(
        registration_sha256=digest(row),
        operation="small",
        path=row["path"],
        signature=row["signature"],
        proof_path=str(path),
        proof_sha256=sha(path),
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        source_hash_kind="file_bytes",
        old_unbound_historical_version_byte_identity_claimed=False,
        **dict.fromkeys(core.contract.FALSE_FIELDS, False),
    )
    if proof.get("identities") is not None:
        receipt["identities"] = {
            k: v for k, v in proof["identities"].items() if k != "projected_records"
        }
    bindings = {str(path): sha(path)}
    bindings.update(
        {
            str(PARENT / name): sha(PARENT / name)
            for name in ("decision.json", "reconciled_history.json", "registration.json")
        }
    )
    bindings.update(proof.get("bindings", {}))
    output = core.apply_metadata_proofs(
        parent,
        [row],
        [receipt],
        source_sha=revision,
        bindings=bindings,
        seed_error_counts=plan["seed_error_counts"],
    )
    history.put(root / "reconciled_history.json", output)
    result = dict(
        source_sha=revision,
        recovered_occurrences=1,
        GPU_jobs=0,
        error_counts=output["counts"],
        parent_source_sha=PARENT_SOURCE,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
        fresh_confirmation_authorized=False,
        historical_media_exposure_resolved=False,
        materialization_authorized=False,
        historical_numeric_validity_claimed=False,
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only new output root required")
    try:
        print(globals()[args.mode](root))
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                fresh_confirmation_authorized=False,
            ),
        )
        raise
