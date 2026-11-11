#!/usr/bin/env python3
"""Download only four public, pinned historical InfographicVQA cache files.

No Parquet, image, question, answer or dataset loader is instantiated. The
download children alone disable HF offline mode, explicitly refuse tokens,
and are killed at the finite parent deadline. Complete corrupt cache bodies
are preserved and fail closed rather than being overwritten or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import evicastkv_historical_provider_superset as providers
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import signature

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
REPO, REVISION = "lmms-lab/DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e"
CARD_OID = "67f107df5895fe445cccf053d7c299d90add0610"
HUB = Path("/scratch/djy8hg/cache/mosaickv/huggingface/hub")
ENDPOINT, SECONDS, HUB_VERSION = "https://huggingface.co", 600, "0.36.2"
PINS = [
    dict(
        path="InfographicVQA/validation-00000-of-00004.parquet",
        size=63414360,
        git_blob_id="59d00a4ed21a6865b3ee865cdb74f7e8167c8572",
        lfs_sha256="a71b63eca8d3cedc5bda7f7d2a4c959b041dca8766592675f6bd0ac49500775a",
    ),
    dict(
        path="InfographicVQA/validation-00001-of-00004.parquet",
        size=72073153,
        git_blob_id="e8755286f8b701363740b310380a019ded6a20fe",
        lfs_sha256="12cbc18ad5a499deebca691fcdf9a852c3d3917e448b30dc604bfdfb4559e756",
    ),
    dict(
        path="InfographicVQA/validation-00002-of-00004.parquet",
        size=57944291,
        git_blob_id="7a0c6d857c50e6067779a19c73b78c37b2d0a09f",
        lfs_sha256="0f4b8f8633105395d8acb5fd341311c614033f35d99cc25337e43c7fd905b3c1",
    ),
    dict(
        path="InfographicVQA/validation-00003-of-00004.parquet",
        size=73025844,
        git_blob_id="621a2d2c8012e9fd79b2af4a2ba2b194646bdc31",
        lfs_sha256="254fdbd3f69499aca70292167f61d89430fb485b3f6a2fce939ab84f02ae5776",
    ),
]


def read(path, expected=None):
    return read_json(read_bytes(path, bound=2**20, expected=expected)[0])


def io_environment():
    environment = dict(os.environ)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    environment.update(
        HF_HUB_OFFLINE="0",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_HUB_ETAG_TIMEOUT="10",
        HF_HUB_DOWNLOAD_TIMEOUT="20",
        HF_HOME=str(HUB.parent.resolve(strict=True)),
        HF_HUB_CACHE=str(HUB.resolve(strict=True)),
        HF_XET_CACHE=str(HUB.parent.resolve(strict=True) / "xet"),
        CUDA_VISIBLE_DEVICES="",
    )
    return environment


def remote_metadata():
    # This mode is read-only network metadata, in an isolated public-IO child.
    import huggingface_hub

    if huggingface_hub.__version__ != HUB_VERSION:
        raise ValueError("registered HF Hub version required")
    api = huggingface_hub.HfApi(endpoint=ENDPOINT, token=False)
    info = api.repo_info(REPO, repo_type="dataset", revision=REVISION, token=False)
    if info.sha != REVISION or info.private is not False or info.gated not in (False, None):
        raise ValueError("exact public ungated dataset revision required")
    found = []
    for number, item in enumerate(
        api.list_repo_tree(
            REPO, "InfographicVQA", repo_type="dataset", revision=REVISION, expand=True, token=False
        )
    ):
        if number >= 32:
            raise ValueError("finite pinned metadata listing limit exceeded")
        if not item.path.startswith("InfographicVQA/validation-"):
            continue
        lfs = getattr(item, "lfs", None)
        if lfs is None or lfs.size != item.size:
            raise ValueError("exact Git LFS size/OID metadata required")
        found.append(
            dict(path=item.path, size=item.size, git_blob_id=item.blob_id, lfs_sha256=lfs.sha256)
        )
    if sorted(found, key=lambda r: r["path"]) != PINS:
        raise ValueError("public file names, sizes or immutable Git/LFS OIDs differ")
    return dict(
        repository=REPO,
        revision=REVISION,
        private=False,
        gated=False,
        files=found,
        total_bytes=sum(row["size"] for row in found),
    )


def call_child(mode, *, root=None, index=None, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), mode]
    if root is not None:
        command.extend(("--root", str(root)))
    if index is not None:
        command.extend(("--index", str(index)))
    result = subprocess.run(
        command, cwd=SOURCE, env=io_environment(), timeout=timeout, capture_output=True, check=False
    )
    if result.returncode:
        # Never reproduce a signed URL, exception payload or inherited token.
        raise RuntimeError("public IO child failed; preserved cache is not replaced")
    if len(result.stdout) > 2**20:
        raise ValueError("public IO metadata response exceeds finite bound")
    return read_json(result.stdout)


def prepare(root, *, cpu_gate_root):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch download-only root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    card = HUB / "datasets--lmms-lab--DocVQA" / "snapshots" / REVISION / "README.md"
    projected, card_sha = providers.card(card, CARD_OID)
    patterns, rows = providers.pool_description(projected, "InfographicVQA", "validation")
    if patterns != ["InfographicVQA/validation-*"] or rows != 2801:
        raise ValueError("pinned card's declared validation pool differs")
    packet = call_child("metadata", timeout=60)
    if packet != dict(
        repository=REPO,
        revision=REVISION,
        private=False,
        gated=False,
        files=PINS,
        total_bytes=sum(row["size"] for row in PINS),
    ):
        raise ValueError("isolated public metadata does not match immutable download plan")
    write_once(root / "public_metadata.json", packet)
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        repository=REPO,
        revision=REVISION,
        card_git_oid=CARD_OID,
        bindings={str(card): card_sha},
        files=PINS,
        expected_rows=rows,
        public_metadata_sha256=sha(root / "public_metadata.json"),
        hub_cache=str(HUB.resolve(strict=True)),
        seconds_per_worker=SECONDS,
        total_download_bytes=packet["total_bytes"],
        download_only=True,
        benchmark_columns_decoded=False,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        video_exposure_resolved=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_public_pinned_download_only",
        files=4,
        maximum_download_bytes=packet["total_bytes"],
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    root = Path(root)
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        read_bytes(name, bound=256 * 2**10, expected=expected)
    read(root / "public_metadata.json", plan["public_metadata_sha256"])
    if (
        plan["files"] != PINS
        or plan["repository"] != REPO
        or plan["revision"] != REVISION
        or Path(plan["hub_cache"]) != HUB.resolve(strict=True)
    ):
        raise ValueError("registered exact public download/cache scope differs")
    return plan


def stable_body(path, pin, deadline):
    before = signature(path)
    if before[2] != pin["size"]:
        raise ValueError("existing complete cache body has wrong size; preserved without overwrite")
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("finite public body-verification deadline")
            raw = stream.read(2**20)
            if not raw:
                break
            hashed.update(raw)
    if hashed.hexdigest() != pin["lfs_sha256"] or signature(path) != before:
        raise ValueError(
            "complete cache body SHA/immutability differs; preserved without overwrite"
        )
    return dict(path=str(path), sha256=hashed.hexdigest(), signature=before, bytes=pin["size"])


def fetch_one(root, index):
    # Run ONLY in the deadline-controlled child, before HF import.
    plan = load(root)
    if type(index) is not int or not 0 <= index < len(PINS):
        raise ValueError("one of four registered download file indices required")
    pin = PINS[index]
    repository = Path(plan["hub_cache"]) / "datasets--lmms-lab--DocVQA"
    alias = repository / "snapshots" / REVISION / pin["path"]
    blob = repository / "blobs" / pin["lfs_sha256"]
    deadline = time.monotonic() + SECONDS
    snapshot = repository / "snapshots" / REVISION
    if repository.resolve(strict=True) != repository or snapshot.resolve(strict=True) != snapshot:
        raise ValueError("exact dataset/revision cache directory must not redirect")
    for directory in (blob.parent, alias.parent):
        if (directory.exists() or directory.is_symlink()) and directory.resolve(
            strict=True
        ) != directory:
            raise ValueError("cache directory redirects outside exact public-file scope")
    if (alias.exists() or alias.is_symlink()) and alias.resolve(strict=True) != blob:
        raise ValueError("existing snapshot alias has unexpected repository/LFS target")
    reused = blob.exists() or blob.is_symlink()
    if reused:
        if blob.resolve(strict=True) != blob:
            raise ValueError("existing cache blob redirects outside exact repository")
        stable_body(blob, pin, deadline)  # Corrupt complete data is never replaced.
    import huggingface_hub

    if huggingface_hub.__version__ != HUB_VERSION:
        raise ValueError("registered HF Hub version required")
    obtained = huggingface_hub.hf_hub_download(
        REPO,
        pin["path"],
        repo_type="dataset",
        revision=REVISION,
        endpoint=ENDPOINT,
        cache_dir=plan["hub_cache"],
        token=False,
        force_download=False,
        local_files_only=False,
        etag_timeout=10,
    )
    if Path(obtained) != alias or alias.resolve(strict=True) != blob:
        raise ValueError("download returned a nonregistered cache alias or blob")
    observed = stable_body(blob, pin, deadline)
    return dict(
        index=index,
        pin=pin,
        artifact=observed,
        reused_complete_blob=reused,
        canonical_snapshot_alias=str(alias),
        benchmark_columns_decoded=False,
    )


def worker(root, index=0):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or index != 0:
        raise ValueError("one bounded CPU download worker only")
    write_once(root / "worker_lock.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    completed, unresolved = [], []
    for number, pin in enumerate(PINS):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            record = call_child("fetch-one", root=root, index=number, timeout=remaining)
            if record["index"] != number or record["pin"] != pin:
                raise ValueError("download child scope differs")
            completed.append(record)
        except (
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            subprocess.SubprocessError,
        ) as error:
            unresolved.append(dict(index=number, pin=pin, reason=type(error).__name__))
            # No automatic retries: later files remain explicitly unattempted.
            unresolved.extend(
                dict(index=i, pin=PINS[i], reason="not_attempted_after_failure")
                for i in range(number + 1, len(PINS))
            )
            break
    output = dict(
        source_sha=plan["source_sha"],
        completed=completed,
        unresolved=unresolved,
        benchmark_columns_decoded=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "worker.json", output)
    return dict(completed=len(completed), unresolved=len(unresolved))


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    path = root / "worker.json"
    result = read(path) if path.exists() else None
    completed = [] if result is None else result["completed"]
    unresolved = [] if result is None else result["unresolved"]
    if result is not None:
        indices = [r["index"] for r in [*completed, *unresolved]]
        if result["source_sha"] != plan["source_sha"] or sorted(indices) != list(range(4)):
            raise ValueError("worker source differs or exact four registered files omitted")
        deadline = time.monotonic() + 60
        for record in completed:
            if record["pin"] != PINS[record["index"]] or record["benchmark_columns_decoded"]:
                raise ValueError("download worker changed scope")
            pin = PINS[record["index"]]
            repository = Path(plan["hub_cache"]) / "datasets--lmms-lab--DocVQA"
            blob = repository / "blobs" / pin["lfs_sha256"]
            alias = repository / "snapshots" / REVISION / pin["path"]
            if (
                Path(record["artifact"]["path"]) != blob
                or blob.resolve(strict=True) != blob
                or Path(record["canonical_snapshot_alias"]) != alias
                or alias.resolve(strict=True) != blob
                or stable_body(blob, pin, deadline) != record["artifact"]
            ):
                raise ValueError("downloaded file changed after worker verification")
        if any(record["pin"] != PINS[record["index"]] for record in unresolved):
            raise ValueError("unresolved public-file queue changed")
    report = dict(
        status="pinned_download_complete" if len(completed) == 4 else "pinned_download_incomplete",
        source_sha=plan["source_sha"],
        complete_files=len(completed),
        unresolved_files=len(unresolved),
        missing_worker=result is None,
        worker_sha256=sha(path) if result is not None else None,
        completed_bytes=sum(r["pin"]["size"] for r in completed),
        files=completed,
        benchmark_columns_decoded=False,
        provider_pixel_coverage_certified=False,
        video_exposure_resolved=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_historical_provider_fetch.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "metadata", "fetch-one", "worker", "analyze"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            result = prepare(args.root, cpu_gate_root=args.cpu_gate_root)
        elif args.mode == "metadata":
            result = remote_metadata()
        elif args.mode == "fetch-one":
            result = fetch_one(args.root, args.index)
        elif args.mode == "worker":
            result = worker(args.root, args.index)
        else:
            result = analyze(args.root)
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps(dict(status="blocked", reason=type(error).__name__)))
        raise SystemExit(1) from None
