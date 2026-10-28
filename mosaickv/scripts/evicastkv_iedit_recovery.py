#!/usr/bin/env python3
"""Recover one pinned missing archive for input/media audit only, never scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import traceback
from pathlib import Path

from evicastkv_media_audit import image_identity, question_type_only

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.fox_milebench_scope import REPOSITORY, REVISION

SOURCE = Path(__file__).resolve().parents[2]
FILENAME = "MileBench_part2.tar.gz"
SIZE = 2018154102
SHA = "0e84fef4ae8bba6760499bf72159c4f8c62c82d9ae70112bf1a72aac949c9505"
SCOPE = Path("/scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-additional-v1")
HISTORY = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-media-content-v1/media_identities.json"
)
MAX_PREFIX_BYTES = 512 * 2**20


def bounded_prefix_audit(fs, required):
    """IEdit is the first pinned task: stop at its last required input asset."""
    remote = f"datasets/{REPOSITORY}@{REVISION}/{FILENAME}"
    if fs.info(remote)["size"] != SIZE:
        raise ValueError("pinned archive size changed")
    identities, metadata = {}, {}
    with fs.open(remote, "rb", block_size=2**20, cache_type="readahead") as stream:

        class BudgetReader:
            consumed = 0

            def read(self, size):
                if size < 0 or self.consumed + size > MAX_PREFIX_BYTES:
                    raise RuntimeError("registered prefix byte budget exhausted")
                raw = stream.read(size)
                self.consumed += len(raw)
                return raw

        reader = BudgetReader()
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if member.isdir() and name == "IEdit":
                    continue
                if not name.startswith("IEdit/"):
                    raise ValueError("required IEdit assets incomplete before next task")
                if name not in required and name != "IEdit/IEdit.json":
                    continue
                if not member.isfile() or member.size > 32 * 2**20:
                    raise ValueError("unsupported requested archive member")
                file = archive.extractfile(member)
                if file is None:
                    raise ValueError("requested asset missing")
                raw = file.read()
                if len(raw) != member.size:
                    raise ValueError("truncated requested asset")
                if name.endswith(".json"):
                    metadata["IEdit"] = dict(
                        question_type=question_type_only(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                else:
                    identities[name] = image_identity(raw)
                if required <= identities.keys() and "IEdit" in metadata:
                    break
        consumed = reader.consumed
    return dict(
        remote=remote,
        expected_full_archive_sha256=SHA,
        full_archive_sha256_status="not_verified_prefix_only",
        archive_sha256_not_claimed_measured=True,
        compressed_prefix_bytes_read=consumed,
        maximum_prefix_bytes=MAX_PREFIX_BYTES,
        identities=identities,
        metadata=metadata,
        errors=[],
    )


def worker(root):
    from huggingface_hub import HfFileSystem

    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    try:
        cache = Path(os.environ["HF_HOME"]).resolve()
        if not cache.is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("all model/data caches must remain under scratch")
        if shutil.disk_usage(cache).free < 64 * 2**20:
            raise RuntimeError("insufficient conservative filesystem headroom")
        if sha(SCOPE / "catalogs/IEdit.json") != protocol["catalog_sha256"]:
            raise ValueError("input-only IEdit catalog changed")
        if sha(HISTORY) != protocol["historical_media_sha256"]:
            raise ValueError("historical media evidence changed")
        rows = json.loads((SCOPE / "catalogs/IEdit.json").read_text())["rows"]
        required = {f"IEdit/images/{p}" for r in rows for p in r["source_image_paths"]}
        required.update(f"IEdit/combined_1_images/{r['combined_1_images']}" for r in rows)
        # Pinned first-task prefix only: no multi-GB download or extraction.
        audit = bounded_prefix_audit(HfFileSystem(), required)
        write_once(root / "media_identities.json", audit)
        old = json.loads(HISTORY.read_text())["identities"]
        fingerprints = {
            fp[k]
            for name, fp in old.items()
            if name.split("/")[0]
            in {"ActionPrediction", "ActionSequence", "CLEVR-Change", "Spot-the-Diff"}
            for k in ("encoded_sha256", "pixel_sha256")
        }
        samples = []
        for row in rows:
            names = [f"IEdit/images/{p}" for p in row["source_image_paths"]]
            names.append(f"IEdit/combined_1_images/{row['combined_1_images']}")
            missing = [n for n in names if n not in audit["identities"]]
            overlap = [
                n
                for n in names
                if n in audit["identities"]
                and any(
                    audit["identities"][n][k] in fingerprints
                    for k in ("encoded_sha256", "pixel_sha256")
                )
            ]
            samples.append(
                dict(sample_id=row["sample_id"], missing_assets=missing, historical_overlap=overlap)
            )
        write_once(root / "sample_audit.json", samples)
        write_once(
            root / "report.json",
            dict(
                status="bounded_media_recovery_complete",
                source_sha=protocol["source_sha"],
                archive_filename=FILENAME,
                expected_full_archive_bytes=SIZE,
                expected_full_archive_sha256=SHA,
                archive_sha256_not_claimed_measured=True,
                compressed_prefix_bytes_read=audit["compressed_prefix_bytes_read"],
                no_archive_persisted=True,
                complete_media_samples=sum(not s["missing_assets"] for s in samples),
                historical_overlap_samples=sum(bool(s["historical_overlap"]) for s in samples),
                official_question_type_verified=audit["metadata"]
                .get("IEdit", {})
                .get("question_type")
                == "open-ended",
                errors=audit["errors"],
                archive_extracted=False,
                responses_decoded=False,
                global_video_document_history_complete=False,
                confirmation_authorized=False,
                transport_bytes_not_measured=True,
                maximum_compressed_prefix_bytes=MAX_PREFIX_BYTES,
            ),
        )
    except Exception as error:
        write_once(
            root / "failure.json",
            dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()),
        )
        raise


def analyze(root):
    path = root / "report.json"
    report = (
        json.loads(path.read_text())
        if path.exists()
        else dict(status="missing", complete_media_samples=0)
    )
    passed = (
        report.get("complete_media_samples") == 200
        and report.get("official_question_type_verified")
        and not report.get("errors")
    )
    write_once(
        root / "analysis.json",
        dict(
            asset_recovery_passed=bool(passed),
            report=report,
            missing_global_source_identity_still_blocks_confirmation=True,
            confirmation_authorized=False,
        ),
    )


def submit(root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("immutable acquisition root exists")
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            repository=REPOSITORY,
            revision=REVISION,
            filename=FILENAME,
            size=SIZE,
            sha256=SHA,
            maximum_archives=1,
            acquisition_mode="stream_only_first_IEdit_task_prefix",
            maximum_compressed_prefix_bytes=MAX_PREFIX_BYTES,
            no_archive_persisted=True,
            catalog_sha256=sha(SCOPE / "catalogs/IEdit.json"),
            historical_media_sha256=sha(HISTORY),
            acquisition_scope="bounded_prefix_for_media_audit_only",
            no_extraction=True,
            no_quality_or_materialization=True,
            confirmation_authorized=False,
        ),
    )
    base = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    script = str(SOURCE / "mosaickv/slurm/evicastkv_iedit_recovery.sbatch")
    cmd = [
        *base,
        f"--output={root}/worker-%j.out",
        f"--error={root}/worker-%j.err",
        script,
        str(SOURCE),
        str(root),
        revision,
        "worker",
    ]
    job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    callback = [
        *base,
        "--cpus-per-task=1",
        "--time=00:05:00",
        f"--dependency=afterany:{job}",
        f"--output={root}/analysis-%j.out",
        f"--error={root}/analysis-%j.err",
        script,
        str(SOURCE),
        str(root),
        revision,
        "analyze",
    ]
    cpu = subprocess.check_output(callback, text=True).strip().split(";")[0]
    write_once(
        root / "submission.json", dict(worker=job, analysis=cpu, command=cmd, callback=callback)
    )
    print(json.dumps(dict(worker=job, analysis=cpu)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("submit", "worker", "analyze"))
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    if not a.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only recovery")
    {"submit": submit, "worker": worker, "analyze": analyze}[a.mode](a.root)
