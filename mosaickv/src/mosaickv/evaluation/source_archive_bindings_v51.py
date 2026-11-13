"""Pure source-content bindings for preserved dirty archives and ad-hoc code.

The I/O coordinator verifies original metadata hashes and archive/member bytes.
This module checks their complete correspondence. A dirty source archive is not
a clean Git execution, and an input producer SHA is not a generator's identity.
Nothing here resolves historical identity coverage or authorizes experiments.
"""

from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import _commit, _path
from mosaickv.evaluation.residual_occurrence_proofs import _key

MAXIMUM_BYTES = 8 * 1024**2
MAXIMUM_NODES = 100000
MAXIMUM_DEPTH = 64
ARCHIVE_FIELDS = {
    "analysis_source_sha",
    "analysis_worktree_dirty",
    "source_code_hashes",
    "derived_artifact_hashes",
}
GENERATOR_FIELDS = {"generator_sha256", "outputs", "producer_source_sha"}
DIRTY_STATUS = "dirty_archive_source_content_and_artifact_binding_verified"
GENERATOR_STATUS = "ad_hoc_generator_source_content_and_artifact_binding_verified"


def _bounded(value: Any) -> None:
    nodes, pending = 0, [(value, 0)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAXIMUM_NODES or depth > MAXIMUM_DEPTH:
            raise ValueError("unchanged projected metadata node/depth bound exceeded")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("string metadata keys required")
            nodes += len(item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str) and len(item) > MAXIMUM_BYTES:
            raise ValueError("unchanged projected metadata byte bound exceeded")
    size = 0
    for part in json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":")).iterencode(
        value
    ):
        size += len(part.encode())
        if size > MAXIMUM_BYTES:
            raise ValueError("unchanged projected metadata byte bound exceeded")


def _schema(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"exact {label} metadata schema required")
    _bounded(value)


def _normalize(value: str) -> str:
    """Normalize only the repository's explicitly known scratch mount alias."""
    if value.startswith("/scratch/djy8hg/"):
        return "/sfs/weka" + value
    return value


def _hash_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"nonempty explicit {label} hash map required")
    result = {}
    for name, expected in value.items():
        name = _normalize(_path(name, label))
        if name in result:
            raise ValueError("duplicate canonical source/artifact alias")
        result[name] = contract._hash(expected, label)
    return dict(sorted(result.items()))


def _size(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value <= MAXIMUM_BYTES:
        raise ValueError(f"unchanged positive 8-MiB {label} bound required")
    return value


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.update(
        canonical_eligible=False,
        clean_runtime_execution_claimed=False,
        original_producer_commit=None,
        occurrence_resolved=False,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
    value["source_content_sha256"] = digest(value)
    _bounded(value)
    return value


def validate_archive(report_projection: dict[str, Any], witness: dict[str, Any]) -> dict[str, Any]:
    """Bind every recorded source hash bijectively to an archived member."""
    _schema(report_projection, ARCHIVE_FIELDS, "dirty archive report projection")
    _schema(witness, {"path", "sha256", "bytes", "members"}, "source archive witness")
    base = _commit(report_projection["analysis_source_sha"])
    if report_projection["analysis_worktree_dirty"] is not True:
        raise ValueError("this lane requires an explicitly dirty original analysis")
    sources = _hash_map(report_projection["source_code_hashes"], "original source code")
    artifacts = _hash_map(report_projection["derived_artifact_hashes"], "original artifacts")
    archive = _normalize(_path(witness["path"], "source archive"))
    archive_sha = contract._hash(witness["sha256"], "recorded archive")
    if PurePosixPath(archive).suffix != ".zip" or artifacts.get(archive) != archive_sha:
        raise ValueError("archive does not match its original artifact-map hash")
    archive_bytes = _size(witness["bytes"], "source archive")
    if not isinstance(witness["members"], list) or not witness["members"]:
        raise ValueError("complete nonempty source-member sequence required")
    by_name, basenames, total = {}, set(), 0
    for member in witness["members"]:
        _schema(member, {"name", "bytes", "sha256"}, "source member")
        name = _path(member["name"], "archive member", relative=True)
        basename = PurePosixPath(name).name
        if name in by_name or basename in basenames:
            raise ValueError("duplicate or aliased archive member basename")
        count = _size(member["bytes"], "expanded member")
        total += count
        if total > MAXIMUM_BYTES:
            raise ValueError("unchanged 8-MiB total expanded-source bound exceeded")
        by_name[name] = dict(
            name=name, bytes=count, sha256=contract._hash(member["sha256"], "member")
        )
        basenames.add(basename)
    source_basenames = [PurePosixPath(path).name for path in sources]
    if len(set(source_basenames)) != len(sources):
        raise ValueError("original source basenames are ambiguous")
    if set(source_basenames) != basenames or len(sources) != len(by_name):
        raise ValueError("archive must cover all and only the recorded source closure")
    members_by_basename = {PurePosixPath(name).name: member for name, member in by_name.items()}
    matched = []
    for source, expected in sources.items():
        member = members_by_basename[PurePosixPath(source).name]
        if member["sha256"] != expected:
            raise ValueError("archived source differs from its original full source hash")
        matched.append(
            dict(
                source_path=source,
                member_name=member["name"],
                bytes=member["bytes"],
                sha256=expected,
            )
        )
    return _seal(
        dict(
            kind="dirty_source_archive",
            analysis_base_commit=base,
            analysis_worktree_dirty=True,
            original_projection=copy.deepcopy(report_projection),
            original_witness=copy.deepcopy(witness),
            archive_path=archive,
            archive_sha256=archive_sha,
            archive_bytes=archive_bytes,
            expanded_source_bytes=total,
            source_files=matched,
            artifact_map_sha256=digest(artifacts),
            committed_source_equivalence_claimed=False,
        )
    )


def validate_generator(
    metadata_projection: dict[str, Any], witness: dict[str, Any]
) -> dict[str, Any]:
    """Bind ad-hoc generator content without relabeling the input's producer."""
    _schema(metadata_projection, GENERATOR_FIELDS, "ad-hoc generator report projection")
    _schema(witness, {"path", "sha256", "bytes"}, "ad-hoc generator witness")
    producer = _commit(metadata_projection["producer_source_sha"])
    recorded = contract._hash(metadata_projection["generator_sha256"], "original generator")
    if contract._hash(witness["sha256"], "generator source") != recorded:
        raise ValueError("generator bytes differ from their original recorded hash")
    artifacts = _hash_map(metadata_projection["outputs"], "original generator outputs")
    return _seal(
        dict(
            kind="ad_hoc_generator",
            generator_path=_normalize(_path(witness["path"], "generator source")),
            generator_sha256=recorded,
            generator_bytes=_size(witness["bytes"], "generator source"),
            input_producer_source_sha=producer,
            input_producer_is_generator_claimed=False,
            original_projection=copy.deepcopy(metadata_projection),
            original_witness=copy.deepcopy(witness),
            artifact_map_sha256=digest(artifacts),
        )
    )


def augment_binding(
    previous: dict[str, Any], artifact_map: dict[str, str], closure: dict[str, Any]
) -> dict[str, Any]:
    """Extend an artifact-only witness with explicitly noncanonical source content."""
    if not isinstance(previous, dict) or not isinstance(closure, dict):
        raise ValueError("explicit previous witness and validated source closure required")
    _bounded(previous)
    _key(previous)
    if (
        previous.get("status") != "artifact_binding_only"
        or previous.get("original_producer_commit") is not None
        or previous.get("original_producer_writer_path") is not None
        or previous.get("original_producer_source_blob_sha256") is not None
        or previous.get("occurrence_resolved") is not False
        or type(previous.get("newly_resolved_occurrences")) is not int
        or previous["newly_resolved_occurrences"] != 0
        or type(previous.get("GPU_jobs")) is not int
        or previous["GPU_jobs"] != 0
        or any(previous.get(name) is not False for name in contract.FALSE_FIELDS)
    ):
        raise ValueError("only an unresolved artifact-only binding may be extended")
    kind = closure.get("kind")
    if kind == "dirty_source_archive":
        expected = validate_archive(
            closure.get("original_projection"), closure.get("original_witness")
        )
        status = DIRTY_STATUS
    elif kind == "ad_hoc_generator":
        expected = validate_generator(
            closure.get("original_projection"), closure.get("original_witness")
        )
        status = GENERATOR_STATUS
    else:
        raise ValueError("unreviewed source-content closure kind")
    if digest(expected) != digest(closure):
        raise ValueError("source-content closure fields or permissions were changed")
    artifacts = _hash_map(artifact_map, "original bound artifact map")
    path = _normalize(_path(previous.get("path"), "previous artifact"))
    if PurePosixPath(path).suffix != ".parquet":
        raise ValueError("only the previously proved Parquet binding may be extended")
    if digest(artifacts) != closure["artifact_map_sha256"] or artifacts.get(path) != contract._hash(
        previous.get("artifact_sha256"), "previous artifact"
    ):
        raise ValueError("previous artifact bytes differ from the source-bound output map")
    return dict(
        copy.deepcopy(previous),
        schema="historical_source_content_binding_v51",
        status=status,
        previous_binding_sha256=digest(previous),
        source_content_kind=kind,
        source_content_sha256=closure["source_content_sha256"],
        canonical_eligible=False,
        clean_runtime_execution_claimed=False,
        original_producer_commit=None,
        original_producer_source_blob_sha256=None,
        occurrence_resolved=False,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
