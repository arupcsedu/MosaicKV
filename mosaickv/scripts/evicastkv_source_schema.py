#!/usr/bin/env python3
"""Inspect pinned Parquet schemas only; never decode prompts or response values."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.evaluation.fox_milebench_scope import (
    ORIGINAL_TASKS,
    PROPOSED_TASKS,
    REPOSITORY,
    REVISION,
)

SOURCE = Path(__file__).resolve().parents[2]
IDENTITY_NAMES = frozenset(
    (
        "video_id",
        "source_video_id",
        "episode_id",
        "source_episode_id",
        "trajectory_id",
        "document_id",
        "source_document_id",
        "source_id",
        "original_id",
        "source_dataset",
    )
)


def schema_paths(schema, prefix=""):
    import pyarrow as pa

    paths = []
    for field in schema:
        path = prefix + field.name
        paths.append(path)
        dtype = field.type
        if pa.types.is_struct(dtype):
            paths.extend(schema_paths(dtype, path + "."))
        elif (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)) and pa.types.is_struct(
            dtype.value_type
        ):
            paths.extend(schema_paths(dtype.value_type, path + "[]."))
    return paths


def audit(output):
    from huggingface_hub import HfFileSystem
    from pyarrow.parquet import ParquetFile

    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("schema audit output must be new")
    fs, cases = HfFileSystem(), []
    aliases = {"CLEVR-Change": "CLEVR_Change", "Spot-the-Diff": "Spot_the_Diff"}
    for task in (*ORIGINAL_TASKS, *PROPOSED_TASKS, "TextNeedleInAHaystack"):
        uri = (
            f"datasets/{REPOSITORY}@{REVISION}/preview/"
            f"{aliases.get(task, task)}_test-00000-of-00001.parquet"
        )
        case = dict(task=task, uri=uri)
        try:
            with fs.open(uri, "rb", cache_type="none") as stream:
                schema = ParquetFile(stream).schema_arrow
            paths = schema_paths(schema)
            case.update(
                status="schema_read",
                field_paths=paths,
                schema_sha256=hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
                candidate_source_identity_fields=[
                    p for p in paths if p.rsplit(".", 1)[-1] in IDENTITY_NAMES
                ],
            )
        except Exception as error:
            case.update(status="unavailable", error=f"{type(error).__name__}: {error}")
        cases.append(case)
    write_once(
        output / "report.json",
        dict(
            source_sha=revision,
            dataset=dict(id=REPOSITORY, revision=REVISION),
            status="schema_audit_complete_not_source_alias_certificate",
            cases=cases,
            metadata_errors=sum(c["status"] != "schema_read" for c in cases),
            sample_rows_decoded=0,
            responses_decoded=0,
            images_downloaded=0,
            models_loaded=0,
            source_identity_mapping_certified=False,
            confirmation_authorized=False,
            limitation=(
                "schema absence describes the pinned preview, "
                "not proof no external mapping exists"
            ),
            official_release="https://huggingface.co/datasets/FreedomIntelligence/MileBench/tree/"
            + REVISION,
            official_code="https://github.com/MileBench/MileBench",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only metadata audit")
    audit(args.output)
