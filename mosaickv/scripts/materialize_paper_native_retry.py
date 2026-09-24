#!/usr/bin/env python3
"""Materialize the preregistered PrefixKV and MileBench retry ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: object) -> None:
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
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _external_scratch(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    scratch = Path("/scratch/djy8hg").resolve()
    if scratch not in resolved.parents:
        raise ValueError(f"{label} must be rooted under /scratch/djy8hg: {resolved}")
    return resolved


def _prefix_ledger(
    source: Path,
    image_root: Path,
    *,
    count: int,
    revision: str,
) -> dict[str, object]:
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < count:
        raise ValueError("PrefixKV source does not contain the preregistered sample count")
    samples: list[dict[str, object]] = []
    for index, raw in enumerate(records[:count]):
        if not isinstance(raw, dict):
            raise ValueError(f"PrefixKV record {index} is not an object")
        conversations = raw.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"PrefixKV record {index} lacks the expected conversation")
        prompt = str(conversations[0].get("value", "")).replace("<image>", "").strip()
        target = str(conversations[1].get("value", "")).strip()
        sample_id = str(raw.get("id", "")).strip()
        relative_image = str(raw.get("image", "")).strip()
        image = (image_root / relative_image).resolve()
        if not sample_id or not prompt or not target or not image.is_file():
            raise ValueError(f"PrefixKV record {index} is incomplete")
        samples.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "references": [target],
                "teacher_forcing_target": target,
                "media": [
                    {
                        "kind": "image",
                        "path": str(image),
                        "sha256": _sha256(image),
                    }
                ],
            }
        )
    return {
        "schema_version": 1,
        "task": "llava_description",
        "dataset_id": "PrefixKV/LLaVA-Description",
        "dataset_revision": revision,
        "split": "official_1000",
        "samples": samples,
    }


def _image_bytes(value: object, index: int) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f"MileBench row {index} has no embedded combined image")
    raw = value.get("bytes")
    if not isinstance(raw, bytes):
        raise ValueError(f"MileBench row {index} has no embedded combined image")
    return raw


def _milebench_ledger(
    parquet: Path,
    media_root: Path,
    *,
    count: int,
    revision: str,
) -> dict[str, object]:
    import pyarrow.parquet as pq

    columns = (
        "questionId",
        "task_instruction",
        "answers",
        "question_combined_image",
        "choice_list",
        "image_combined",
    )
    table = pq.read_table(parquet, columns=list(columns))
    if table.num_rows < count:
        raise ValueError("MileBench source does not contain the preregistered sample count")
    rows = table.slice(0, count).to_pylist()
    media_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        question_id = int(row["questionId"])
        sample_id = f"ActionSequence-{question_id:04d}"
        choices = [str(item) for item in row["choice_list"]]
        reference = str(row["answers"]).strip()
        question = str(row["question_combined_image"]).replace("<image>", "").strip()
        instruction = str(row["task_instruction"]).strip()
        if reference not in choices:
            raise ValueError(f"MileBench row {index} reference is absent from its choices")
        image = media_root / f"{sample_id}.jpg"
        if image.exists():
            raise FileExistsError(f"refusing to overwrite immutable media: {image}")
        image.write_bytes(_image_bytes(row["image_combined"], index))
        prompt = f"{instruction}\n{question}\nChoice List: {json.dumps(choices)}"
        samples.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "references": [reference],
                "media": [
                    {
                        "kind": "image",
                        "path": str(image.resolve()),
                        "sha256": _sha256(image),
                    }
                ],
                "milebench_dataset": "ActionSequence",
                "milebench_question_type": "multi-choice",
                "choices": choices,
            }
        )
    return {
        "schema_version": 1,
        "task": "milebench",
        "dataset_id": "MileBench/MileBench",
        "dataset_revision": revision,
        "split": "official_evaluation",
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument(
        "--experiment-id",
        choices=("paper_native_retry_v2", "paper_native_latency_v3"),
        default="paper_native_retry_v2",
    )
    parser.add_argument(
        "--prefix-source",
        type=Path,
        default=Path("/scratch/djy8hg/data/mosaickv/prefixkv/detail_1k.json"),
    )
    parser.add_argument(
        "--prefix-image-root",
        type=Path,
        default=Path("/scratch/djy8hg/data/mosaickv/prefixkv"),
    )
    parser.add_argument("--prefix-revision", required=True)
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--milebench-parquet", type=Path, required=True)
    parser.add_argument("--milebench-revision", required=True)
    parser.add_argument("--milebench-count", type=int, default=20)
    args = parser.parse_args()
    if len(args.source_git_sha) != 40:
        raise ValueError("--source-git-sha must be an immutable 40-character commit")
    if args.prefix_count != 4 or args.milebench_count != 20:
        raise ValueError(
            f"{args.experiment_id} preregisters exactly 4 PrefixKV and 20 MileBench samples"
        )

    root = _external_scratch(args.output_root, "output root")
    prefix_source = _external_scratch(args.prefix_source, "PrefixKV source")
    prefix_image_root = _external_scratch(args.prefix_image_root, "PrefixKV image root")
    milebench_parquet = _external_scratch(args.milebench_parquet, "MileBench source")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    prefix_path = root / "prefixkv/ledger.json"
    milebench_path = root / "milebench/ledger.json"
    _write_once(
        prefix_path,
        _prefix_ledger(
            prefix_source,
            prefix_image_root,
            count=args.prefix_count,
            revision=args.prefix_revision,
        ),
    )
    _write_once(
        milebench_path,
        _milebench_ledger(
            milebench_parquet,
            root / "milebench/media",
            count=args.milebench_count,
            revision=args.milebench_revision,
        ),
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "classification": "preregistered_development_validation",
        "source_git_sha": args.source_git_sha,
        "prefixkv": {
            "selection_rule": "first_4_records_in_source_order",
            "source": str(prefix_source),
            "source_sha256": _sha256(prefix_source),
            "ledger": str(prefix_path),
            "ledger_sha256": _sha256(prefix_path),
        },
        "milebench": {
            "dataset": "ActionSequence",
            "selection_rule": "first_20_records_in_source_order",
            "source": str(milebench_parquet),
            "source_sha256": _sha256(milebench_parquet),
            "ledger": str(milebench_path),
            "ledger_sha256": _sha256(milebench_path),
        },
    }
    _write_once(root / "materialization.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
