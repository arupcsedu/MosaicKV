#!/usr/bin/env python3
"""Prepare ID/media catalogs only; no ledger/model/experiment entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.fox_milebench_scope import (
    PROPOSED_TASKS,
    REPOSITORY,
    REVISION,
    catalog_fingerprint,
    read_projected_catalog,
    summarize_availability,
    validate_protocol,
)

SOURCE = Path(__file__).resolve().parents[2]
SCRATCH = Path("/scratch/djy8hg").resolve()


def scratch_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(SCRATCH):
        raise ValueError("audit paths must remain under /scratch/djy8hg")
    return path


def audit(protocol_path: Path, inventory_path: Path, output: Path) -> dict:
    from huggingface_hub import HfFileSystem
    from pyarrow.parquet import ParquetFile

    source_sha = clean_source(SOURCE)
    protocol_bytes_sha = sha(protocol_path)
    inventory_bytes_sha = sha(inventory_path)
    protocol = json.loads(protocol_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    validate_protocol(protocol)
    if output.exists():
        raise FileExistsError("refusing to reuse an audit output root")
    output.mkdir(parents=True)
    write_once(output / "protocol.json", protocol)
    fs = HfFileSystem()
    catalogs = []
    artifacts = {}
    for task, question_type in PROPOSED_TASKS.items():
        remote = f"datasets/{REPOSITORY}@{REVISION}/preview/{task}_test-00000-of-00001.parquet"
        with fs.open(remote, "rb", cache_type="none") as stream:
            catalog = read_projected_catalog(
                ParquetFile(stream), task=task, question_type=question_type
            )
        catalog["source"]["remote_uri"] = remote
        file = output / "catalogs" / f"{task}.json"
        write_once(file, catalog)
        artifacts[task] = dict(
            path=str(file), sha256=sha(file), content_sha256=catalog_fingerprint(catalog)
        )
        catalogs.append(catalog)
    if sha(protocol_path) != protocol_bytes_sha or sha(inventory_path) != inventory_bytes_sha:
        raise ValueError("registered inputs changed during audit")
    clean_source(SOURCE, source_sha)
    summary = summarize_availability(catalogs, inventory, protocol)
    summary.update(
        source_sha=source_sha,
        protocol_file_sha256=protocol_bytes_sha,
        history_inventory_path=str(inventory_path),
        history_inventory_sha256=inventory_bytes_sha,
        catalogs=artifacts,
        measurement_type="input_metadata_availability_audit_not_quality_or_timing",
        no_model_loaded=True,
        no_reference_or_prompt_values_decoded=True,
        no_media_download_or_confirmation_selection=True,
    )
    write_once(output / "availability.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=scratch_path)
    parser.add_argument("--history-inventory", required=True, type=scratch_path)
    parser.add_argument("--output", required=True, type=scratch_path)
    args = parser.parse_args()
    summary = audit(args.protocol, args.history_inventory, args.output)
    print(
        json.dumps(dict(status=summary["status"], tasks=summary["tasks"], output=str(args.output)))
    )


if __name__ == "__main__":
    main()
