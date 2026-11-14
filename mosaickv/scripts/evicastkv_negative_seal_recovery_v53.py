#!/usr/bin/env python3
"""Recover original sealed artifact digests, never infer runtime producers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import evicastkv_manifest_recovery_v52 as packets
import evicastkv_recover_bindings_v49 as prior
import evicastkv_startup_recovery_v50 as startup
import yaml
from yaml import events

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.negative_seal_bindings_v53 import artifact_claim, augment_seal_binding

SOURCE = Path(__file__).resolve().parents[2]
REGISTRY = SOURCE / "mosaickv/configs/audits/negative_seals_v53.json"
PARENT = packets.COMPLETED_BINDINGS
PARENT_SHA = packets.COMPLETED_BINDINGS_SHA
CATALOG_WITNESS = Path(
    "/scratch/djy8hg/workdir/fox-review-v53/discovery-20261113-1/modest_source_hashes.json"
)
CATALOG_WITNESS_SHA = "d4ac8232626539b2f96d248e89d63283b2b723ddd34aeaba1a7085e19c63759a"
FIELDS = {"classification", "immutable_result", "frozen_candidate_created", "run_root", "artifacts"}
OPTIONAL_FIELDS = {"immutable_ledger_root"}
DROP = object()


def project_yaml(raw, budget):
    """Event-only projection: no constructors, aliases, tags or outcome retention."""
    if not isinstance(raw, bytes) or len(raw) > prior.LIVE_BYTES:
        raise ValueError("unchanged 8-MiB YAML input bound exceeded")
    stream = iter(yaml.parse(raw))
    nodes = 0

    def event():
        nonlocal nodes
        budget.check()
        nodes += 1
        if nodes > prior.LIVE_NODES:
            raise ValueError("unchanged 100,000-node YAML bound exceeded")
        value = next(stream)
        if isinstance(value, events.ScalarEvent) and len(value.value.encode("utf-8")) > 2**20:
            raise ValueError("unchanged 1-MiB scalar limit exceeded")
        if getattr(value, "anchor", None) is not None or getattr(value, "tag", None) is not None:
            raise ValueError("source seal aliases, anchors and explicit tags are unsupported")
        return value

    def node(first, keep, depth):
        if depth > 64:
            raise ValueError("unchanged 64-level YAML bound exceeded")
        if isinstance(first, events.ScalarEvent):
            return first.value if keep else DROP
        if isinstance(first, events.MappingStartEvent):
            result, keys = {}, set()
            while True:
                key = event()
                if isinstance(key, events.MappingEndEvent):
                    return result if keep else DROP
                if not isinstance(key, events.ScalarEvent) or key.value == "<<":
                    raise ValueError("plain scalar YAML keys without merges required")
                if key.value in keys:
                    raise ValueError("duplicate YAML source key")
                keys.add(key.value)
                selected = keep and (depth != 0 or key.value in FIELDS | OPTIONAL_FIELDS)
                value = node(event(), selected, depth + 1)
                if selected:
                    result[key.value] = value
        if isinstance(first, events.SequenceStartEvent):
            result = []
            while True:
                child = event()
                if isinstance(child, events.SequenceEndEvent):
                    return result if keep else DROP
                value = node(child, keep, depth + 1)
                if keep:
                    result.append(value)
        raise ValueError("unsupported YAML source event")

    if not isinstance(event(), events.StreamStartEvent) or not isinstance(
        event(), events.DocumentStartEvent
    ):
        raise ValueError("one explicit source-seal document required")
    first = event()
    if not isinstance(first, events.MappingStartEvent):
        raise ValueError("source-seal mapping required")
    result = node(first, True, 0)
    if not isinstance(event(), events.DocumentEndEvent) or not isinstance(
        event(), events.StreamEndEvent
    ):
        raise ValueError("exactly one YAML source document required")
    if not set(result) >= FIELDS or set(result) - FIELDS - OPTIONAL_FIELDS:
        raise ValueError("complete original seal projection required")
    for field in ("immutable_result", "frozen_candidate_created"):
        if result[field] not in ("true", "false"):
            raise ValueError("explicit source seal booleans required")
        result[field] = result[field] == "true"
    prior.projected_bound(result)
    return result


def git_output(arguments, budget, *, maximum=prior.LIVE_BYTES):
    budget.authorize(maximum)
    result = subprocess.check_output(
        ["git", "--no-replace-objects", *arguments],
        cwd=SOURCE,
        timeout=min(15, max(0.01, prior.SECONDS - budget.check())),
    )
    budget.add(len(result))
    if len(result) > maximum:
        raise ValueError("bounded original Git record exceeded its limit")
    return result


def original_source(spec, ancestor, revision, budget):
    commit, path = spec["original_seal_commit"], spec["repository_path"]
    if (
        prior.COMMIT.fullmatch(commit or "") is None
        or not path.startswith("mosaickv/configs/ledgers/choice_c_v")
        or not path.endswith("_negative_calibration_seal.yaml")
        or ".." in Path(path).parts
    ):
        raise ValueError("exact reviewed historical seal commit and source path required")
    # A seal revision is metadata provenance, never a model/analysis runtime SHA.
    subprocess.run(
        ["git", "--no-replace-objects", "merge-base", "--is-ancestor", commit, ancestor],
        cwd=SOURCE,
        timeout=min(15, max(0.01, prior.SECONDS - budget.check())),
        check=True,
    )
    address = f"{commit}:{path}"
    old_blob = git_output(["rev-parse", "--verify", address], budget, maximum=128).decode().strip()
    current_blob = (
        git_output(["rev-parse", "--verify", f"{revision}:{path}"], budget, maximum=128)
        .decode()
        .strip()
    )
    if old_blob != spec["git_blob_oid"] or current_blob != old_blob:
        raise ValueError("original and current tracked seal blobs differ")
    size = int(git_output(["cat-file", "-s", address], budget, maximum=128))
    if not 0 < size <= prior.LIVE_BYTES:
        raise ValueError("unchanged original source live-byte limit exceeded")
    raw = git_output(["show", address], budget, maximum=size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("original committed seal source hash differs")
    return raw


def recover(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = prior.Budget()
    registry = packets.saved_json(REGISTRY, None, budget)
    if (
        registry["schema"] != "historical_negative_seal_registry_v53"
        or registry["reviewed_ancestor_sha"] != "4b2cf0b2e9e4dfdfd82b6bc08b253cb331f9b0bf"
        or registry["parent_report_sha256"] != PARENT_SHA
        or len(registry["seals"]) != 19
    ):
        raise ValueError("exact reviewed nineteen-seal registry required")
    parent = packets.saved_json(PARENT / "recovered_bindings.json", PARENT_SHA, budget)
    original = packets.saved_json(
        startup.PARENT / "recovered_bindings.json", startup.PARENT_SHA, budget
    )
    inventory = packets.saved_json(
        prior.PARENT / "inventory.json", prior.BOUND["inventory.json"], budget
    )
    if (
        parent["source_sha"] != "b3a8d9fe4e3240e4766d4762d430354cdade4cdc"
        or parent["status_counts"]["unrecovered"] != 102
    ):
        raise ValueError("exact successful V52 binding receipt required")
    rows = {row["path"]: row for row in parent["rows"]}
    old_rows = {row["path"]: row for row in original["rows"]}
    initial = {row["path"]: row for row in inventory["rows"] if row["operation"] == "parquet"}
    if len(rows) != 134 or rows.keys() != old_rows.keys() or rows.keys() != initial.keys():
        raise ValueError("all 134 complete saved transactions must remain present")
    supplemental, source_records = {}, []
    for spec in registry["seals"]:
        raw = original_source(spec, registry["reviewed_ancestor_sha"], revision, budget)
        projection = project_yaml(raw, budget)
        for target in spec["targets"]:
            path = target["path"]
            if (
                path not in rows
                or path in supplemental
                or rows[path]["status"] != "unrecovered"
                or target["sha256"] != rows[path]["artifact_sha256"]
            ):
                raise ValueError("original registered target is missing, duplicated, or changed")
            claim = artifact_claim(
                projection, str(SOURCE / spec["repository_path"]), spec["sha256"], path
            )
            if claim["artifact_sha256"] != target["sha256"]:
                raise ValueError("original seal artifact digest differs from saved proof")
            supplemental[path] = [claim]
        source_records.append(
            dict(
                spec,
                projected_original_metadata=projection,
                source_role="historical_seal_metadata_not_runtime_writer",
            )
        )
    if len(supplemental) != 33:
        raise ValueError("all thirty-three original artifact seals must qualify")
    catalog = packets.saved_json(CATALOG_WITNESS, CATALOG_WITNESS_SHA, budget)
    if (
        catalog.get("runtime_writer_attested") is not False
        or catalog.get("bindings_promoted") != 0
        or catalog.get("credential_bodies_read") is not False
    ):
        raise ValueError("the external catalog must remain an unpromoted discovery")
    output = []
    for previous in parent["rows"]:
        name = previous["path"]
        path = prior.allowed(name)
        if str(path) != name or prior.geometry(path) != previous["signature"]:
            raise ValueError("saved artifact signature changed; do not replay source body")
        transaction = startup.reused_transaction(initial[name], old_rows[name], original)
        output.append(
            augment_seal_binding(
                previous,
                initial[name],
                transaction,
                original["binding_claims"][name],
                supplemental[name],
                reviewed_writers=prior.REVIEWED,
            )
            if name in supplemental
            else previous
        )
    result = dict(
        source_sha=revision,
        parent_report_path=str(PARENT / "recovered_bindings.json"),
        parent_report_sha256=PARENT_SHA,
        registry_sha256=sha(REGISTRY),
        rows=output,
        seal_source_records=source_records,
        unpromoted_partial_source_catalog=dict(
            discovery_witness_path=str(CATALOG_WITNESS),
            discovery_witness_sha256=CATALOG_WITNESS_SHA,
            original_catalog_path=catalog["original_path"],
            original_catalog_sha256=catalog["original_file_sha256"],
            runtime_writer_established=False,
            promoted_records=0,
            reason="catalog_not_bound_to_original_execution",
        ),
        supplemental_claims=supplemental,
        recovered_paths=sorted(supplemental),
        status_counts=dict(Counter(row["status"] for row in output)),
        reused_hash_transactions=134,
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        historical_code_executed=False,
        original_occurrence_counts=parent["original_occurrence_counts"],
        newly_resolved_occurrences=0,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
        historical_media_exposure_resolved=False,
    )
    prior.projected_bound(result)
    prior.resume.require_artifact_capacity(root, prior.LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"],
        recovered=len(supplemental),
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "recovered_bindings.json"
    if not path.exists():
        return dict(
            source_sha=revision,
            status="seal_recovery_failed_or_incomplete",
            GPU_jobs=0,
            fresh_confirmation_authorized=False,
        )
    report = packets.saved_json(path, None, prior.Budget())
    if (
        report["source_sha"] != revision
        or report["reused_hash_transactions"] != 134
        or report["parquet_bodies_reopened"] != 0
        or report["newly_resolved_occurrences"] != 0
    ):
        raise ValueError("completed proof reuse without semantic or execution promotion required")
    return dict(
        source_sha=revision,
        report_sha256=sha(path),
        status_counts=report["status_counts"],
        recovered_paths=report["recovered_paths"],
        reused_hash_transactions=134,
        newly_resolved_occurrences=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
        GPU_jobs=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "analyze"))
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
                error="original negative-seal recovery failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
