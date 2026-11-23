#!/usr/bin/env python3
"""Offline CPU processor-metadata capture, with weights/data physically absent.

The optional real capture is isolated with bubblewrap. Only the pinned source,
common environment, system runtime libraries, an allowlisted staged processor
asset set, and a new output directory are mounted. Home, original Hub caches,
datasets, media, model weights and credentials are not mounted. This is an
operational dependency capture, NOT numerical/freshness or historical proof.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import resource
import signal
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath

from mosaickv.evaluation.fox_campaign import write_once
from mosaickv.evaluation.fox_confirmation_inputs import (
    MODEL_ID,
    MODEL_REVISION,
    PROCESSOR_DESCRIPTOR_SCHEMA,
    processor_fingerprints,
)
from mosaickv.evaluation.paper_readiness_v63 import AUTHORIZATION_KEYS

SOURCE = Path(__file__).resolve().parents[2]
ENVIRONMENT = Path("/sfs/weka/scratch/djy8hg/env/mosaickv")
WORKDIR = Path("/sfs/weka/scratch/djy8hg/workdir")
SCRATCH = Path("/sfs/weka/scratch/djy8hg")
HUB = SCRATCH / "cache/mosaickv/huggingface/hub"
REPOSITORY = HUB / "models--llava-hf--llava-v1.6-mistral-7b-hf"
SNAPSHOT = REPOSITORY / "snapshots" / MODEL_REVISION
SANDBOX_SNAPSHOT = Path("/cache/huggingface/hub") / REPOSITORY.name / "snapshots" / MODEL_REVISION
ASSET_NAMES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
)
REQUIRED_NAMES = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
MAX_SECONDS = 210
MAX_INPUT = 2 * 1024**3
MAX_FILE = 512 * 1024**2
MAX_STAGED = 64 * 1024**2  # stricter operational allowance, not a larger old limit
MAX_METADATA = 8 * 1024**2
MAX_RAM = 32 * 1024**3


def canonical_scratch(path):
    value = str(path)
    if value == "/scratch/djy8hg" or value.startswith("/scratch/djy8hg/"):
        value = "/sfs/weka/scratch/djy8hg" + value[len("/scratch/djy8hg") :]
    parts = PurePosixPath(value).parts
    if (
        not value.startswith(str(SCRATCH) + "/")
        or ".." in parts
        or "." in value.split("/")
        or any(p.lower() in {".codex", ".ssh", ".env", "auth.json"} for p in parts)
    ):
        raise ValueError("explicit non-home scratch path required")
    return Path(value)


def open_directory(path):
    """Do not follow hidden ancestor symlinks or arbitrary symlink targets."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in Path(path).parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def metadata_file(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ValueError("unchanged 8 MiB metadata limit exceeded")
    return raw


def stage_assets(snapshot, staging):
    """Read ONLY nominated Hub links and regular content-addressed blobs."""
    snapshot, staging = Path(snapshot), Path(staging)
    source_fd = open_directory(snapshot)
    blob_fd = open_directory(snapshot.parent.parent / "blobs")
    records, consumed, total = [], 0, 0
    try:
        staging.mkdir(mode=0o700)
        for name in ASSET_NAMES:
            try:
                info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISLNK(info.st_mode):
                raise ValueError("pinned Hub processor asset must be a content-addressed link")
            target = PurePosixPath(os.readlink(name, dir_fd=source_fd))
            if (
                target.parts[:3] != ("..", "..", "blobs")
                or len(target.parts) != 4
                or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", target.name) is None
            ):
                raise ValueError("processor link escapes the nominated Hub blob directory")
            fd = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=blob_fd)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or not 0 < before.st_size <= MAX_FILE
                    or total + before.st_size > MAX_STAGED
                    or consumed + 2 * before.st_size > MAX_INPUT
                ):
                    raise ValueError("unchanged processor input/file limits exceeded")
                total += before.st_size
                sha = hashlib.sha256()
                git_blob = hashlib.sha1(b"blob " + str(before.st_size).encode() + b"\0")
                destination = staging / name
                with destination.open("xb") as out:
                    os.chmod(destination, 0o600)
                    for chunk in iter(lambda: stream.read(64 * 1024), b""):
                        consumed += len(chunk)
                        if consumed > MAX_INPUT:
                            raise ValueError("unchanged combined processor input limit exceeded")
                        sha.update(chunk)
                        git_blob.update(chunk)
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                after = os.fstat(stream.fileno())

            def signature(info):
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

            if signature(before) != signature(after):
                raise ValueError("processor blob changed during capture")
            address = git_blob.hexdigest() if len(target.name) == 40 else sha.hexdigest()
            if address != target.name:
                raise ValueError("processor asset content-address mismatch")
            verify = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(64 * 1024), b""):
                    consumed += len(chunk)
                    verify.update(chunk)
            if verify.hexdigest() != sha.hexdigest() or consumed > MAX_INPUT:
                raise ValueError("staged processor readback mismatch or input limit exceeded")
            records.append(
                dict(name=name, blob_id=target.name, sha256=sha.hexdigest(), bytes=before.st_size)
            )
        if not {r["name"] for r in records} >= REQUIRED_NAMES:
            raise ValueError("required processor assets missing")
        if not {"tokenizer.json", "tokenizer.model"} & {r["name"] for r in records}:
            raise ValueError("actual tokenizer asset missing")
        return dict(assets=records, staged_bytes=total, source_plus_readback_bytes=consumed)
    finally:
        os.close(source_fd)
        os.close(blob_fd)


def sandbox_command(root, *, source=SOURCE, environment=ENVIRONMENT):
    """No home, original Hub cache, data/media or model-weight mounts."""
    root, source, environment = map(Path, (root, source, environment))
    command = [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    for path in ("/usr", "/lib", "/lib64"):
        command += ["--ro-bind", path, path]
    command += [
        "--ro-bind",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    command += ["--ro-bind", str(environment), "/runtime", "--ro-bind", str(source), "/code"]
    command += ["--tmpfs", "/cache", "--tmpfs", "/tmp"]
    command += [
        "--ro-bind",
        str(root / "assets"),
        str(SANDBOX_SNAPSHOT),
        "--bind",
        str(root / "child"),
        "/output",
        "--chdir",
        "/output",
    ]
    values = {
        "HOME": "/nonexistent",
        "USER": "processor_capture",
        "LOGNAME": "processor_capture",
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "/code/mosaickv/src",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/cache/datasets",
        "XDG_CACHE_HOME": "/cache/xdg",
        "TORCH_HOME": "/cache/torch",
        "TMPDIR": "/tmp",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in values.items():
        command += ["--setenv", key, value]
    return [
        *command,
        "/runtime/bin/python",
        "/code/mosaickv/scripts/" + Path(__file__).name,
        "--child",
    ]


def sandbox_environment():
    """Version-independent clear environment before bubblewrap itself starts."""
    return {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


def capture_child(*, loader=None):
    """Heavy imports occur only in the offline mount namespace."""
    required = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HOME": "/nonexistent",
        "OMP_NUM_THREADS": "1",
    }
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise RuntimeError("offline CPU processor sandbox environment required")
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_METADATA, MAX_METADATA))
    signal.alarm(MAX_SECONDS)
    if loader is None:
        from transformers import AutoProcessor

        loader = AutoProcessor.from_pretrained
    processor = loader(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        token=False,
        trust_remote_code=False,
    )
    # The existing runtime signature is reproduced, but never persisted.
    signature = dict(
        processor_class=type(processor).__qualname__,
        image_preprocessor=processor.image_processor.to_dict(),
        chat_template=getattr(processor, "chat_template", None),
        tokenizer_chat_template=getattr(processor.tokenizer, "chat_template", None),
        special_tokens=processor.tokenizer.special_tokens_map,
        vocabulary=processor.tokenizer.get_vocab(),
    )
    result = dict(
        status="captured_unreviewed",
        processor_class=signature["processor_class"],
        **processor_fingerprints(signature),
        packages={
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "torch")
        },
        model_weights_loaded=False,
        dataset_accessed=False,
        GPU_used=False,
    )
    if len(json.dumps(result).encode()) > MAX_METADATA:
        raise ValueError("unchanged compact metadata limit exceeded")
    return result


def _read_derived(path, expected=None):
    raw = metadata_file(path)
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("derived processor capture changed")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("derived processor object required")
    return value, actual


def processor_descriptor(child, manifest_sha, source_sha, *, reviewed=False):
    identity = dict(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        processor_id=MODEL_ID,
        processor_revision=MODEL_REVISION,
    )
    return dict(
        schema=PROCESSOR_DESCRIPTOR_SCHEMA,
        reviewed=reviewed,
        local_files_only=True,
        **identity,
        processor_class=child["processor_class"],
        **{
            k: child[k]
            for k in (
                "processor_signature_sha256",
                "preprocessing_sha256",
                "tokenization_sha256",
                "chat_template_sha256",
            )
        },
        provenance=dict(**identity, source_sha=source_sha, capture_manifest_sha256=manifest_sha),
    )


def promote_reviewed_capture(root, *, expected_manifest_sha256, reviewer, output):
    """Separate explicit Codex operational review, never fabricated human review."""
    if reviewer != "codex":
        raise ValueError("only explicit Codex operational review is implemented")
    root = Path(root)
    manifest, manifest_sha = _read_derived(root / "capture_manifest.json", expected_manifest_sha256)
    if (
        manifest.get("status") != "captured_unreviewed"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("successful pinned capture required")
    child, _ = _read_derived(root / "child/processor_fingerprints.json", manifest["child_sha256"])
    if child.get("status") != "captured_unreviewed" or any(
        child.get(k) is not False for k in ("model_weights_loaded", "dataset_accessed", "GPU_used")
    ):
        raise ValueError("processor-only capture required")
    result = processor_descriptor(
        child, manifest_sha, manifest["source"]["source_sha"], reviewed=True
    )
    result["review"] = dict(
        actor="codex", type="operational_metadata_review_not_human_or_empirical_certification"
    )
    write_once(Path(output), result)
    return result


def capture(root, *, snapshot=SNAPSHOT, source=SOURCE):
    start = time.monotonic()
    root, snapshot = canonical_scratch(root), canonical_scratch(snapshot)
    if snapshot != SNAPSHOT or not root.is_relative_to(WORKDIR) or root.is_relative_to(source):
        raise ValueError("new external workdir and exact pinned snapshot required")
    if (
        any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("set offline flags and CUDA_VISIBLE_DEVICES before Python starts")
    if not os.environ.get("HF_HOME", "").startswith("/scratch/djy8hg/cache/mosaickv/"):
        raise ValueError("source the scratch-only common cache environment first")
    signal.alarm(MAX_SECONDS)
    root.mkdir(mode=0o700, exist_ok=False)
    (root / "child").mkdir(mode=0o700)
    git_env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, env=git_env, text=True, timeout=10
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, env=git_env, timeout=10
    )
    source_info = dict(
        source_sha=revision,
        worktree_dirty=bool(dirty),
        status_sha256=hashlib.sha256(dirty).hexdigest(),
    )
    lock = metadata_file(source / "mosaickv/env/common/requirements.lock")
    source_info["common_lock_sha256"] = hashlib.sha256(lock).hexdigest()
    source_info["capture_implementation_sha256"] = {
        name: hashlib.sha256(metadata_file(source / name)).hexdigest()
        for name in (
            "mosaickv/scripts/evicastkv_capture_confirmation_processor.py",
            "mosaickv/src/mosaickv/evaluation/fox_confirmation_inputs.py",
            "mosaickv/src/mosaickv/evaluation/fox_campaign.py",
            "mosaickv/src/mosaickv/evaluation/paper_readiness_v63.py",
        )
    }
    source_info["content_binding_scope"] = (
        "capture_and_dependency_implementation_only_not_whole_dirty_tree"
    )
    assets = stage_assets(snapshot, root / "assets")
    command = sandbox_command(root, source=source)
    with (
        (root / "child_stdout.txt").open("xb") as out,
        (root / "child_stderr.txt").open("xb") as err,
    ):
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            timeout=max(1, MAX_SECONDS - (time.monotonic() - start)),
            env=sandbox_environment(),
        )
    if result.returncode:
        raise RuntimeError(
            f"sandboxed processor capture failed; code={result.returncode}; "
            f"inspect {root / 'child_stderr.txt'}"
        )
    child, child_sha = _read_derived(root / "child/processor_fingerprints.json")
    manifest = dict(
        schema="evicastkv_offline_processor_capture_v1",
        status="captured_unreviewed",
        classification="operational_processor_metadata_not_empirical_or_historical_proof",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        source=source_info,
        **assets,
        child_sha256=child_sha,
        elapsed_seconds=time.monotonic() - start,
        limits=dict(
            seconds=MAX_SECONDS,
            RAM_bytes=MAX_RAM,
            input_bytes=MAX_INPUT,
            file_bytes=MAX_FILE,
            metadata_bytes=MAX_METADATA,
            stricter_staged_asset_bytes=MAX_STAGED,
            threads=1,
        ),
        input_accounting_scope="allowlisted_unique_assets_and_source_copy_readback_not_exhaustive_native_cumulative_IO_certificate",
        authorization={key: False for key in AUTHORIZATION_KEYS},
    )
    write_once(root / "capture_manifest.json", manifest)
    _, manifest_sha = _read_derived(root / "capture_manifest.json")
    write_once(
        root / "captured_unreviewed_descriptor.json",
        processor_descriptor(child, manifest_sha, revision),
    )
    return dict(
        status="captured_unreviewed", root=str(root), manifest_sha256=manifest_sha, GPU_used=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.child:
        write_once(Path("/output/processor_fingerprints.json"), capture_child())
        return
    if args.root is None:
        parser.error("--root is required")
    print(json.dumps(capture(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
