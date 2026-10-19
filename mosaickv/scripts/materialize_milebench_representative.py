#!/usr/bin/env python3
"""Materialize a hash-bound representative MileBench comparison cohort.

The pinned LOOK-M source expects a custom multi-image LLaVA fork that is not
present in its public repository.  The unified HF lane therefore consumes the
official ``combined_1_images`` representation supplied by MileBench and labels
every compressed baseline row ``lookm_reimpl``.  It must never be described as
an official LOOK-M execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REVISION = "53c7a58051ef88bacf76541d91f03f5ba2d71e7d"
SEED = 20271014
TASKS: dict[str, str] = {
    "ActionSequence": "multi-choice",
    "CLEVR-Change": "open-ended",
    "Spot-the-Diff": "open-ended",
    "TextNeedleInAHaystack": "open-ended",
}
_IMAGE_MARKER = re.compile(r"\{image#\d+\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _under_scratch(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    scratch = Path("/scratch/djy8hg").resolve()
    try:
        resolved.relative_to(scratch)
    except ValueError as error:
        raise ValueError(f"{label} must be under /scratch/djy8hg: {resolved}") from error
    return resolved


def _sample_id(task: str, raw: dict[str, Any]) -> str:
    value = raw.get("sample_id")
    if type(value) is not int or value < 0:
        raise ValueError(f"{task} has an invalid sample_id")
    return f"{task}-{value:04d}"


def _ranked_records(
    task: str,
    records: list[dict[str, Any]],
    *,
    count: int,
    excluded_ids: frozenset[str],
) -> list[dict[str, Any]]:
    available = [record for record in records if _sample_id(task, record) not in excluded_ids]
    ranked = sorted(
        available,
        key=lambda record: hashlib.sha256(
            f"{SEED}:{task}:{_sample_id(task, record)}".encode()
        ).digest(),
    )
    if len(ranked) < count:
        raise ValueError(f"{task} has {len(ranked)} eligible records; {count} required")
    return ranked[:count]


def _clean_context(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("MileBench context must be a string")
    cleaned = _IMAGE_MARKER.sub("", value).strip()
    if not cleaned:
        raise ValueError("MileBench context is empty after removing image markers")
    return cleaned


def _task_samples(
    dataset_root: Path,
    task: str,
    *,
    count: int,
    excluded_ids: frozenset[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    annotation = dataset_root / task / f"{task}.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"invalid official annotation structure: {annotation}")
    metadata = payload.get("meta_data")
    if not isinstance(metadata, dict) or metadata.get("question_type") != TASKS[task]:
        raise ValueError(f"unexpected official question type for {task}")
    instructions = metadata.get("task_instruction")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{task} has no task instructions")
    records = _ranked_records(
        task,
        payload["data"],
        count=count,
        excluded_ids=excluded_ids,
    )
    samples: list[dict[str, object]] = []
    selected_media: dict[str, str] = {}
    for raw in records:
        sample_id = _sample_id(task, raw)
        instance = raw.get("task_instance")
        if not isinstance(instance, dict):
            raise ValueError(f"{sample_id} has no task_instance")
        instruction_index = raw.get("task_instruction_id")
        if type(instruction_index) is not int or not 0 <= instruction_index < len(instructions):
            raise ValueError(f"{sample_id} has an invalid task_instruction_id")
        combined = instance.get("combined_1_images")
        if not isinstance(combined, list) or len(combined) != 1:
            raise ValueError(f"{sample_id} must have exactly one official combined_1_images asset")
        relative_media = combined[0]
        if not isinstance(relative_media, str) or not relative_media:
            raise ValueError(f"{sample_id} has an invalid combined image path")
        media = (dataset_root / task / "combined_1_images" / relative_media).resolve()
        if not media.is_file():
            raise FileNotFoundError(f"missing official combined image: {media}")
        media_sha = sha256(media)
        selected_media[str(media)] = media_sha
        choices = instance.get("choice_list")
        question_type = TASKS[task]
        if question_type == "multi-choice":
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"{sample_id} requires a non-empty choice list")
            normalized_choices = [str(choice) for choice in choices]
        else:
            if choices not in (None, []):
                raise ValueError(f"{sample_id} is open-ended but contains choices")
            normalized_choices = []
        reference = raw.get("response")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"{sample_id} has no reference answer")
        prompt = (
            f"{instructions[instruction_index].strip()}\n{_clean_context(instance.get('context'))}"
        )
        if normalized_choices:
            prompt += f"\nChoice List: {json.dumps(normalized_choices, ensure_ascii=False)}"
        sample: dict[str, object] = {
            "sample_id": sample_id,
            "prompt": prompt,
            "references": [reference.strip()],
            "media": [{"kind": "image", "path": str(media), "sha256": media_sha}],
            "milebench_dataset": task,
            "milebench_question_type": question_type,
            "media_representation": "official_combined_1_images",
        }
        if normalized_choices:
            sample["choices"] = normalized_choices
        samples.append(sample)
    provenance = {
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation),
        "selected_media": selected_media,
        "selected_media_count": len(selected_media),
    }
    return samples, provenance


def materialize(
    dataset_root: Path,
    output_root: Path,
    *,
    count_per_task: int,
    excluded_ids: frozenset[str] = frozenset(),
    source_revision: str = SOURCE_REVISION,
) -> dict[str, object]:
    if count_per_task < 1:
        raise ValueError("count_per_task must be positive")
    dataset_root = _under_scratch(dataset_root, "dataset root")
    output_root = _under_scratch(output_root, "output root")
    if output_root.exists():
        raise FileExistsError(f"new immutable output root required: {output_root}")
    task_payloads: dict[str, list[dict[str, object]]] = {}
    dependencies: dict[str, object] = {}
    for task in TASKS:
        samples, provenance = _task_samples(
            dataset_root,
            task,
            count=count_per_task,
            excluded_ids=excluded_ids,
        )
        task_payloads[task] = samples
        dependencies[task] = provenance
    revision_payload = {
        "upstream_dataset": "FreedomIntelligence/MileBench",
        "upstream_revision": source_revision,
        "representation": "official_combined_1_images",
        "selection_seed": SEED,
        "count_per_task": count_per_task,
        "excluded_ids": sorted(excluded_ids),
        "dependencies": dependencies,
    }
    materialization_content_sha256 = _canonical_sha(revision_payload)
    ledger_paths: dict[str, str] = {}
    ledger_hashes: dict[str, str] = {}
    for task, samples in task_payloads.items():
        path = output_root / "ledgers" / f"{task}.json"
        write_once(
            path,
            {
                "schema_version": 1,
                "task": "milebench",
                "dataset_id": "MileBench/MileBench",
                "dataset_revision": source_revision,
                "split": "official_evaluation",
                "samples": samples,
            },
        )
        ledger_paths[task] = str(path)
        ledger_hashes[task] = sha256(path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "classification": "representative_lookm_reimpl_unified_comparison_input",
        "official_lookm_execution": False,
        "reason_not_official": "pinned LOOK-M omits its custom multi-image LLaVA source",
        "dataset_id": "FreedomIntelligence/MileBench",
        "upstream_revision": source_revision,
        "dataset_revision": source_revision,
        "materialization_content_sha256": materialization_content_sha256,
        "representation": "official_combined_1_images",
        "selection_rule": "sha256_rank(seed,task,sample_id)",
        "selection_seed": SEED,
        "count_per_task": count_per_task,
        "total_samples": count_per_task * len(TASKS),
        "metrics": {
            "ActionSequence": "accuracy",
            "TextNeedleInAHaystack": "accuracy",
            "CLEVR-Change": "rouge_l_f1",
            "Spot-the-Diff": "rouge_l_f1",
        },
        "excluded_ids": sorted(excluded_ids),
        "dependencies": dependencies,
        "ledgers": ledger_paths,
        "ledger_sha256": ledger_hashes,
    }
    write_once(output_root / "materialization.json", manifest)
    return manifest


def _load_exclusions(paths: list[Path]) -> frozenset[str]:
    identities: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"exclusion ledger has no samples: {path}")
        for sample in samples:
            if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
                raise ValueError(f"invalid exclusion sample in {path}")
            identities.add(sample["sample_id"])
    return frozenset(identities)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count-per-task", type=int, default=72)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--exclude-ledger", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = materialize(
        args.dataset_root,
        args.output_root,
        count_per_task=args.count_per_task,
        excluded_ids=_load_exclusions(args.exclude_ledger),
        source_revision=args.source_revision,
    )
    print(
        json.dumps(
            {
                "materialization": str(args.output_root / "materialization.json"),
                "dataset_revision": manifest["dataset_revision"],
                "materialization_content_sha256": manifest["materialization_content_sha256"],
                "total_samples": manifest["total_samples"],
                "ledger_sha256": manifest["ledger_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
