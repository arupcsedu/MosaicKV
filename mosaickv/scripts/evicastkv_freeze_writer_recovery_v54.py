#!/usr/bin/env python3
"""Validate saved freeze execution capsules; never replay historical artifact I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path

import evicastkv_negative_seal_recovery_v53 as seals
import evicastkv_recover_bindings_v49 as prior
import evicastkv_startup_recovery_v50 as startup
import yaml
from yaml import events

from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.evaluation.freeze_binding_lineage_v54 import promote_freeze_binding
from mosaickv.evaluation.freeze_writer_bindings_v54 import (
    validate_freeze_runtime,
    validate_registered_spec,
)
from mosaickv.evaluation.parquet_binding_recovery_v49 import _path as source_path

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v53/run-20261113-f9fe413-1")
PARENT_SHA = "844d7f99f18e64fb36baab75bfd13cfd21826a24ad8281f587708f6c27c1e4e3"
REGISTRY = SOURCE / "mosaickv/configs/audits/freeze_writers_v54.json"
DISCOVERY = Path("/scratch/djy8hg/workdir/fox-review-v53/discovery-20261113-1")
CAPSULES = {
    "freeze_runtime_candidates.json": (
        "0c0a2dad327eb5851df2b81bda428839c1356f5e7a56797f9e0394f977bc9ad4"
    ),
    "freeze_runtime_remaining_candidates.json": (
        "0b1ab599d367412eb1cf1673b604bda6d48d1c28dd42dfeabcb8887b1b1ce186"
    ),
}
SEAL_PATHS = {
    ("classification",),
    ("frozen_candidate_created",),
    ("run_root",),
    ("protocol", "path"),
    ("protocol", "sha256"),
    ("slurm_jobs", "freeze"),
    ("execution_jobs", "freeze_analysis"),
    ("execution_jobs", "freeze_state"),
}
DROP = object()
FOLLOWUPS = {
    Path("/scratch/djy8hg/workdir/fox-review-v54/discovery-20261113-1/")
    / "compound_inventory_scope_negative.json": (
        "c42784c8c57727e0ad3f0a740aca46ba5db66d912513ed246b1d17683f55da3f"
    ),
    Path("/scratch/djy8hg/workdir/fox-review-v54/discovery-20261113-1/")
    / "modest_analysis_replacement_pointer.json": (
        "09057511ce982267ed659b0e629fb7afc4ac14847b20ab87e5a72faf00a59e6d"
    ),
}
SAVED_OWNER_ROOT = prior.HISTORY.parent
SAVED_ALIAS_ROOT = Path("/scratch/djy8hg")
SAVED_INPUTS = frozenset(
    {
        REGISTRY,
        PARENT / "recovered_bindings.json",
        seals.PARENT / "recovered_bindings.json",
        startup.PARENT / "recovered_bindings.json",
        prior.PARENT / "inventory.json",
        *(DISCOVERY / name for name in CAPSULES),
        *FOLLOWUPS,
    }
)


def saved_path(path):
    text = os.fspath(path)
    path = Path(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or ".." in path.parts
        or "\\" in text
        or len(text.encode()) > 4096
        or any(ord(char) < 32 or ord(char) == 127 for char in text)
        or any(
            part.casefold() in {".codex", "auth.json", "credentials.json", ".env", ".netrc"}
            for part in path.parts
        )
    ):
        raise ValueError("canonical noncredential saved-metadata path required")
    if path.is_relative_to(SAVED_ALIAS_ROOT):
        return SAVED_OWNER_ROOT / path.relative_to(SAVED_ALIAS_ROOT)
    if path.is_relative_to(SAVED_OWNER_ROOT):
        return path
    raise ValueError("saved metadata outside reviewed scratch root")


def json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate saved-metadata JSON key")
        value[key] = item
    return value


def metadata_bounds(value):
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > prior.LIVE_NODES or depth > 64:
            raise ValueError("unchanged saved-metadata node/depth bound exceeded")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
            pending.extend((k, depth + 1) for k in item)
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str) and len(item.encode()) > 2**20:
            raise ValueError("unchanged saved-metadata scalar bound exceeded")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("finite saved metadata required")
    prior.projected_bound(value)


def saved_json(path, expected, budget, *, output_root=None):
    """One confined read; no symlink following or post-validation reopens."""
    original = Path(path)
    canonical = saved_path(path)
    allowed = {saved_path(value) for value in SAVED_INPUTS}
    if output_root is not None:
        allowed.add(saved_path(Path(output_root) / "recovered_bindings.json"))
    if canonical not in allowed or canonical.suffix != ".json" or original.is_symlink():
        raise ValueError("unregistered or redirected saved-metadata path")
    if original.resolve(strict=True) != canonical:
        raise ValueError("saved-metadata ancestor redirect is forbidden")
    before = canonical.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("single-owner regular saved metadata required")
    size = before.st_size
    if not 0 < size <= prior.LIVE_BYTES:
        raise ValueError("unchanged 8-MiB saved-metadata limit exceeded")

    def signature(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    budget.authorize(size + 1)
    descriptor = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("saved-metadata descriptor changed before read")
        raw = stream.read(size + 1)
        budget.add(len(raw))
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError("saved metadata changed during read")
    if len(raw) != size or signature(canonical.lstat()) != signature(before):
        raise ValueError("saved metadata changed after read")
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("original saved-metadata digest differs")

    def nonfinite(_):
        raise ValueError("finite saved metadata required")

    try:
        value = json.loads(raw, object_pairs_hook=json_object, parse_constant=nonfinite)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("bounded UTF-8 saved metadata required") from error
    if not isinstance(value, dict):
        raise ValueError("saved-metadata object required")
    metadata_bounds(value)
    if not hasattr(budget, "metadata_hashes"):
        budget.metadata_hashes = {}
    budget.metadata_hashes[str(canonical)] = actual
    return value


def saved_sha(path, budget):
    return budget.metadata_hashes[str(saved_path(path))]


def execution_projection(raw, budget):
    """Retain only original job/protocol links, never scoring or output values."""
    if not isinstance(raw, bytes) or len(raw) > prior.LIVE_BYTES:
        raise ValueError("unchanged 8-MiB source projection bound exceeded")
    stream, nodes = iter(yaml.parse(raw)), 0

    def event():
        nonlocal nodes
        budget.check()
        nodes += 1
        if nodes > prior.LIVE_NODES:
            raise ValueError("unchanged 100,000-node bound exceeded")
        item = next(stream)
        if getattr(item, "anchor", None) is not None or getattr(item, "tag", None) is not None:
            raise ValueError("source aliases, anchors and explicit tags are unsupported")
        if isinstance(item, events.ScalarEvent) and len(item.value.encode()) > 2**20:
            raise ValueError("unchanged 1-MiB scalar bound exceeded")
        return item

    def node(item, path, depth):
        if depth > 64:
            raise ValueError("unchanged 64-level bound exceeded")
        keep = any(selected[: len(path)] == path for selected in SEAL_PATHS)
        if isinstance(item, events.ScalarEvent):
            return item.value if path in SEAL_PATHS else DROP
        if isinstance(item, events.MappingStartEvent):
            result, keys = {}, set()
            while True:
                key = event()
                if isinstance(key, events.MappingEndEvent):
                    return result if keep else DROP
                if not isinstance(key, events.ScalarEvent) or key.value == "<<":
                    raise ValueError("scalar source mapping keys without merges required")
                if key.value in keys:
                    raise ValueError("duplicate source YAML key")
                keys.add(key.value)
                value = node(event(), (*path, key.value), depth + 1)
                if value is not DROP:
                    result[key.value] = value
        if isinstance(item, events.SequenceStartEvent):
            if keep:
                raise ValueError("job/protocol metadata cannot be an array")
            while True:
                child = event()
                if isinstance(child, events.SequenceEndEvent):
                    return DROP
                node(child, path, depth + 1)
        raise ValueError("unsupported source YAML event")

    if not isinstance(event(), events.StreamStartEvent) or not isinstance(
        event(), events.DocumentStartEvent
    ):
        raise ValueError("one source YAML document required")
    first = event()
    if not isinstance(first, events.MappingStartEvent):
        raise ValueError("source metadata mapping required")
    selected = node(first, (), 0)
    if not isinstance(event(), events.DocumentEndEvent) or not isinstance(
        event(), events.StreamEndEvent
    ):
        raise ValueError("exactly one complete source document required")
    common = {"classification", "frozen_candidate_created", "run_root", "protocol"}
    if not common <= set(selected) or set(selected) - common not in (
        {"slurm_jobs"},
        {"execution_jobs"},
    ):
        raise ValueError("unambiguous original freeze-job field required")
    if selected["frozen_candidate_created"] != "false":
        raise ValueError("original result must remain a negative freeze")
    protocol = selected["protocol"]
    if not isinstance(protocol, dict) or set(protocol) != {"path", "sha256"}:
        raise ValueError("original complete protocol binding required")
    robust = "execution_jobs" in selected
    jobs = selected["execution_jobs" if robust else "slurm_jobs"]
    job_field = "freeze_analysis" if robust else "freeze"
    if not isinstance(jobs, dict) or set(jobs) != (
        {job_field, "freeze_state"} if robust else {job_field}
    ):
        raise ValueError("exact original freeze execution metadata required")
    result = dict(
        classification=selected["classification"],
        run_root=selected["run_root"],
        frozen_candidate_created=False,
        protocol_path=protocol["path"],
        protocol_sha256=protocol["sha256"],
        freeze_job=jobs[job_field],
    )
    if robust:
        result["freeze_state"] = jobs["freeze_state"]
    prior.projected_bound(result)
    return result


def source_bytes(commit, path, expected, budget):
    path = source_path(path, "reviewed original source", relative=True)
    if (
        prior.COMMIT.fullmatch(commit or "") is None
        or not path.startswith("mosaickv/")
        or ".." in Path(path).parts
        or Path(path).is_absolute()
        or Path(path).suffix not in {".py", ".sh", ".sbatch", ".yaml"}
    ):
        raise ValueError("exact local immutable source blob required")
    address = f"{commit}:{path}"
    size = int(seals.git_output(["cat-file", "-s", address], budget, maximum=128))
    if not 0 < size <= prior.LIVE_BYTES:
        raise ValueError("unchanged live-source byte limit exceeded")
    raw = seals.git_output(["show", address], budget, maximum=size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original critical source blob differs")
    return raw


def recover(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = prior.Budget()
    registry = saved_json(REGISTRY, None, budget)
    parent = saved_json(PARENT / "recovered_bindings.json", PARENT_SHA, budget)
    before = saved_json(seals.PARENT / "recovered_bindings.json", seals.PARENT_SHA, budget)
    original = saved_json(startup.PARENT / "recovered_bindings.json", startup.PARENT_SHA, budget)
    inventory = saved_json(prior.PARENT / "inventory.json", prior.BOUND["inventory.json"], budget)
    if (
        registry["schema"] != "historical_freeze_writer_registry_v54"
        or len(registry["specs"]) != 6
        or len(parent["rows"]) != 134
        or parent["source_sha"] != "f9fe413d85207606a21aaaecc8b7f977b9ab7f66"
    ):
        raise ValueError("exact six-job registry and successful V53 receipt required")
    records = {}
    for name, expected in CAPSULES.items():
        capsule = saved_json(DISCOVERY / name, expected, budget)
        for record in capsule["records"]:
            if record["job_id"] in records:
                raise ValueError("duplicate saved execution capsule")
            records[record["job_id"]] = record
    facts_by_path, runtime_claims, writer_registry = {}, {}, dict(prior.REVIEWED)
    verification_records = []
    for spec in registry["specs"]:
        spec = validate_registered_spec(spec)
        capsule = records[spec["job_id"]]
        seal = source_bytes(spec["seal_commit"], spec["seal_path"], spec["seal_sha256"], budget)
        seal_execution = execution_projection(seal, budget)
        blobs = {
            path: source_bytes(spec["source_sha"], path, expected, budget)
            for path, expected in spec["source_blob_sha256s"].items()
        }
        tree = (
            seals.git_output(
                ["rev-parse", "--verify", spec["source_sha"] + "^{tree}"], budget, maximum=128
            )
            .decode("ascii")
            .strip()
        )
        facts = validate_freeze_runtime(
            capsule["original_public_accounting_record"].encode("utf-8"),
            capsule["original_clean_startup_prefix"].encode("ascii"),
            capsule["original_public_batch_script_record"].encode("utf-8"),
            seal_execution,
            spec,
            blobs,
            tree,
        )
        for output in spec["outputs"]:
            path = output["path"]
            if path in facts_by_path:
                raise ValueError("duplicate registered writer target")
            facts_by_path[path] = facts
            old_claims = parent["supplemental_claims"][path]
            if len(old_claims) != 1:
                raise ValueError("one preserved original seal claim required")
            old = old_claims[0]
            runtime_claims[path] = dict(
                old,
                role="runtime_writer",
                recorded_source_sha=spec["source_sha"],
                source_blob_sha256=output["writer_sha256"],
                writer_path=output["writer_path"],
                source_evidence_verified=True,
            )
            writer_registry[(spec["source_sha"], output["writer_path"])] = output["writer_sha256"]
        verification_records.append(facts)
    if len(runtime_claims) != 7:
        raise ValueError("all seven candidate links must validate without fallback")
    unpromoted_followups = []
    for path, expected in FOLLOWUPS.items():
        saved_json(path, expected, budget)
        unpromoted_followups.append(
            dict(
                path=str(path),
                sha256=saved_sha(path, budget),
                used_for_binding_promotion=False,
            )
        )
    old_rows = {row["path"]: row for row in original["rows"]}
    before_rows = {row["path"]: row for row in before["rows"]}
    inventory_rows = {
        row["path"]: row for row in inventory["rows"] if row["operation"] == "parquet"
    }
    paths = {row["path"] for row in parent["rows"]}
    if (
        len(paths) != 134
        or paths != old_rows.keys()
        or paths != inventory_rows.keys()
        or paths != before_rows.keys()
    ):
        raise ValueError("all original 134 saved transactions must remain present")
    output_rows = []
    for previous in parent["rows"]:
        path = previous["path"]
        canonical = prior.allowed(path)
        if str(canonical) != path or prior.geometry(canonical) != previous["signature"]:
            raise ValueError("saved artifact changed; do not replay its completed proof")
        transaction = startup.reused_transaction(inventory_rows[path], old_rows[path], original)
        if path in runtime_claims:
            previous = promote_freeze_binding(
                previous,
                before_rows[path],
                inventory_rows[path],
                transaction,
                original["binding_claims"][path],
                parent["supplemental_claims"][path],
                runtime_claims[path],
                runtime_facts=facts_by_path[path],
                previous_reviewed_writers=prior.REVIEWED,
                reviewed_writers=writer_registry,
            )
        output_rows.append(previous)
    result = dict(
        source_sha=revision,
        parent_report_path=str(PARENT / "recovered_bindings.json"),
        parent_report_sha256=PARENT_SHA,
        registry_sha256=saved_sha(REGISTRY, budget),
        rows=output_rows,
        runtime_verification=verification_records,
        unpromoted_followup_receipts=unpromoted_followups,
        promoted_paths=sorted(runtime_claims),
        status_counts=dict(Counter(row["status"] for row in output_rows)),
        original_occurrence_counts=parent["original_occurrence_counts"],
        reused_hash_transactions=134,
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        original_startup_logs_reopened=0,
        Slurm_records_queried=0,
        credential_bodies_read=False,
        historical_code_executed=False,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        newly_resolved_occurrences=0,
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
        promoted=7,
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
            status="freeze_writer_validation_failed_or_incomplete",
            GPU_jobs=0,
            fresh_confirmation_authorized=False,
        )
    budget = prior.Budget()
    report = saved_json(path, None, budget, output_root=root)
    if (
        report["source_sha"] != revision
        or report["parquet_bodies_reopened"] != 0
        or report["reused_hash_transactions"] != 134
        or report["newly_resolved_occurrences"] != 0
    ):
        raise ValueError("unchanged proof reuse and scientific boundaries required")
    return dict(
        source_sha=revision,
        report_sha256=saved_sha(path, budget),
        status_counts=report["status_counts"],
        promoted_paths=report["promoted_paths"],
        reused_hash_transactions=134,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
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
                error="original freeze-writer validation failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
