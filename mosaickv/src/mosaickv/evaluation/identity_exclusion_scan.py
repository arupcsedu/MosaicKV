"""Fail-closed, identity-only prior-use scanning for paper evaluation.

The scanner reads experiment artifacts, never benchmark datasets.  It projects
only identity and provenance fields, skips catalog files entirely, and emits a
hash-bound inventory suitable for a write-once exclusion manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    SampleIdentity,
    SampleProtocolError,
    canonical_bytes,
)

_SUPPORTED_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml", ".parquet"})
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "attention",
        "error",
        "failure_identity",
        "generated_tokens",
        "image",
        "images",
        "logit_kl",
        "logits",
        "media",
        "prompt",
        "reference",
        "references",
        "task_score",
        "token_agreement",
        "video",
    }
)
_PARQUET_IDENTITY_COLUMNS = frozenset(
    {
        "canonical_sample_identity",
        "dataset_config",
        "dataset_id",
        "dataset_name",
        "dataset_revision",
        "sample_id",
        "sample_ids",
        "split",
        "task",
        "task_name",
    }
)


@dataclass(frozen=True, slots=True)
class _DatasetContext:
    dataset_id: str | None = None
    dataset_config: str | None = None
    dataset_revision: str | None = None
    split: str | None = None

    def overlay(self, value: Mapping[str, object]) -> _DatasetContext:
        dataset = value.get("dataset")
        nested = dataset if isinstance(dataset, Mapping) else {}
        resolved = value.get("resolved_config")
        if isinstance(resolved, Mapping) and isinstance(resolved.get("dataset"), Mapping):
            nested = cast("Mapping[str, object]", resolved["dataset"])

        def field(direct: str, *nested_names: str) -> str | None:
            candidate = value.get(direct)
            if candidate is None:
                for name in nested_names:
                    if nested.get(name) is not None:
                        candidate = nested[name]
                        break
            if candidate is None:
                return None
            if not isinstance(candidate, str) or not candidate.strip():
                return "__invalid__"
            return candidate

        return _DatasetContext(
            dataset_id=field("dataset_id", "id") or self.dataset_id,
            dataset_config=(
                field("dataset_config", "config", "dataset_config", "name")
                or self.dataset_config
            ),
            dataset_revision=field("dataset_revision", "revision") or self.dataset_revision,
            split=field("split", "split") or self.split,
        )

    def is_target(self, target: DatasetContract) -> bool:
        return self.dataset_id == target.dataset_id

    def matches(self, target: DatasetContract) -> bool:
        return (
            self.dataset_id == target.dataset_id
            and self.dataset_config == target.dataset_config
            and self.dataset_revision == target.dataset_revision
            and self.split == target.split
        )

    def missing_fields(self) -> tuple[str, ...]:
        values = {
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
        }
        return tuple(name for name, value in values.items() if value in (None, "__invalid__"))


@dataclass(slots=True)
class _Projection:
    identities: set[SampleIdentity]
    target_mentions: int = 0


def _task_names(value: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for key in ("task", "task_name"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            result.append(candidate.casefold())
    return tuple(result)


def _sample_ids(value: Mapping[str, object]) -> tuple[str, ...]:
    raw_single = value.get("sample_id")
    raw_many = value.get("sample_ids")
    candidates: list[object] = []
    if raw_single is not None:
        candidates.append(raw_single)
    if raw_many is not None:
        if not isinstance(raw_many, list):
            raise SampleProtocolError("sample_ids must be an array in an identity artifact")
        candidates.extend(raw_many)
    result: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise SampleProtocolError("sample identity must be a nonempty string")
        result.append(candidate)
    return tuple(result)


def _project_value(
    value: object,
    *,
    target: DatasetContract,
    projection: _Projection,
    context: _DatasetContext | None = None,
    source: Path,
) -> None:
    if context is None:
        context = _DatasetContext()
    if isinstance(value, list):
        for item in value:
            _project_value(
                item,
                target=target,
                projection=projection,
                context=context,
                source=source,
            )
        return
    if not isinstance(value, Mapping):
        return

    mapping = cast("Mapping[str, object]", value)
    local = context.overlay(mapping)
    tasks = _task_names(mapping)
    ids = _sample_ids(mapping)
    task_targets = target.task.casefold() in tasks
    prefixed_target = any(
        sample_id.casefold().startswith(f"{target.task.casefold()}:") for sample_id in ids
    )
    relevant = local.is_target(target) or task_targets or prefixed_target
    if local.is_target(target):
        projection.target_mentions += 1
    if ids and relevant:
        missing = local.missing_fields()
        if missing:
            raise SampleProtocolError(
                f"RealWorldQA sample provenance is incomplete in {source}: {missing}"
            )
        if not local.matches(target):
            raise SampleProtocolError(
                f"RealWorldQA sample provenance conflicts with the pinned contract: {source}"
            )
        projection.identities.update(target.identity(sample_id) for sample_id in ids)

    for key, child in mapping.items():
        if key in _FORBIDDEN_KEYS or key in {"sample_id", "sample_ids"}:
            continue
        if isinstance(child, (Mapping, list)):
            _project_value(
                child,
                target=target,
                projection=projection,
                context=local,
                source=source,
            )


def _load_json_or_yaml(path: Path) -> tuple[object, int]:
    try:
        if path.suffix.casefold() == ".json":
            return json.loads(path.read_text(encoding="utf-8")), 1
        return yaml.safe_load(path.read_text(encoding="utf-8")), 1
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise SampleProtocolError(f"invalid structured exclusion source {path}: {error}") from error


def _project_jsonl(
    path: Path,
    *,
    target: DatasetContract,
    projection: _Projection,
    context: _DatasetContext,
) -> int:
    record_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record_count += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SampleProtocolError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise SampleProtocolError(
                        f"JSONL record is not an object: {path}:{line_number}"
                    )
                _project_value(
                    value,
                    target=target,
                    projection=projection,
                    context=context,
                    source=path,
                )
    except UnicodeDecodeError as error:
        raise SampleProtocolError(f"invalid UTF-8 exclusion source {path}") from error
    return record_count


def _project_parquet(
    path: Path,
    *,
    target: DatasetContract,
    projection: _Projection,
    context: _DatasetContext,
) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - locked environment includes pyarrow.
        raise SampleProtocolError(
            "pyarrow is required to project Parquet identity columns"
        ) from error
    parquet = pq.ParquetFile(path)
    record_count = int(parquet.metadata.num_rows)
    columns = sorted(set(parquet.schema_arrow.names) & _PARQUET_IDENTITY_COLUMNS)
    if not columns:
        return record_count
    for row in parquet.read(columns=columns).to_pylist():
        if not isinstance(row, Mapping):  # pragma: no cover - pyarrow returns mappings.
            raise SampleProtocolError(f"Parquet projection returned a non-object row: {path}")
        _project_value(
            row,
            target=target,
            projection=projection,
            context=context,
            source=path,
        )
    return record_count


def _companion_context(path: Path, target: DatasetContract) -> _DatasetContext:
    if path.name == "manifest.json":
        return _DatasetContext()
    manifest = path.parent / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        return _DatasetContext()
    value, _ = _load_json_or_yaml(manifest)
    if not isinstance(value, Mapping):
        raise SampleProtocolError(f"companion manifest is not an object: {manifest}")
    context = _DatasetContext().overlay(cast("Mapping[str, object]", value))
    if context.is_target(target) and context.missing_fields():
        raise SampleProtocolError(f"RealWorldQA companion manifest is incomplete: {manifest}")
    return context


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_identity_sources(
    roots: Iterable[Path], *, ignored: Iterable[Path] = ()
) -> tuple[Path, ...]:
    """Return supported non-catalog artifacts without following symlinks."""

    ignored_paths = {path.resolve() for path in ignored}
    discovered: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            raise SampleProtocolError(f"exclusion scan root does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SampleProtocolError(f"exclusion scan refuses symbolic links: {path}")
            if not path.is_file() or path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in ignored_paths or "catalog" in path.name.casefold():
                continue
            if not resolved.is_relative_to(root):  # pragma: no cover - symlinks rejected above.
                raise SampleProtocolError(f"exclusion source escapes its root: {path}")
            discovered.add(resolved)
    return tuple(sorted(discovered, key=str))


def scan_identity_sources(
    *,
    roots: Iterable[Path],
    target: DatasetContract,
    source_git_sha: str,
    source_git_dirty: bool,
    ignored: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build an in-memory, hash-bound prior-use manifest."""

    if source_git_dirty or len(source_git_sha) != 40:
        raise SampleProtocolError("canonical exclusion scanning requires a clean full Git SHA")
    sources: list[dict[str, Any]] = []
    identities: set[SampleIdentity] = set()
    discovered = discover_identity_sources(roots, ignored=ignored)
    for path in discovered:
        before = _sha256_stream(path)
        projection = _Projection(set())
        context = _companion_context(path, target)
        suffix = path.suffix.casefold()
        if suffix == ".jsonl":
            record_count = _project_jsonl(
                path,
                target=target,
                projection=projection,
                context=context,
            )
        elif suffix == ".parquet":
            record_count = _project_parquet(
                path,
                target=target,
                projection=projection,
                context=context,
            )
        else:
            value, record_count = _load_json_or_yaml(path)
            _project_value(
                value,
                target=target,
                projection=projection,
                context=context,
                source=path,
            )
        after = _sha256_stream(path)
        if before != after:
            raise SampleProtocolError(f"exclusion source changed during scan: {path}")
        ordered = sorted(projection.identities, key=SampleIdentity.canonical_key)
        sources.append(
            {
                "path": str(path),
                "sha256": before,
                "format": suffix.removeprefix("."),
                "record_count": record_count,
                "target_mention_count": projection.target_mentions,
                "matching_identity_count": len(ordered),
                "matching_identities_sha256": hashlib.sha256(
                    canonical_bytes([identity.as_dict() for identity in ordered])
                ).hexdigest(),
            }
        )
        identities.update(ordered)
    ordered_identities = sorted(identities, key=SampleIdentity.canonical_key)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "classification": "label_blind_identity_and_provenance_exclusion_manifest",
        "source_git_sha": source_git_sha,
        "source_git_dirty": False,
        "scan_roots": [str(path.resolve()) for path in sorted(roots, key=str)],
        "dataset_contract": target.as_dict(),
        "source_count": len(sources),
        "sources": sources,
        "excluded_unique_sample_count": len(ordered_identities),
        "excluded_identities": [identity.as_dict() for identity in ordered_identities],
        "forbidden_content_fields_accessed": False,
        "dataset_rows_or_media_loaded": False,
    }
    payload["manifest_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def write_once_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create a read-only artifact without an overwrite race."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "discover_identity_sources",
    "scan_identity_sources",
    "write_once_exclusive",
]
