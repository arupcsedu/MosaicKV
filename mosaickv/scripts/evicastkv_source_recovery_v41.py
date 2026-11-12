"""Narrow source-alias proof and recovery of exactly named historical commits.

Source objects, configuration trees and backup filenames are not substitutes
for original model/media execution journals. No result here licenses freshness.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path

PREFIX_COMMIT = "597f1ab032704951550f93bcc8a23f1454b80aa4"
PREFIX_REPO = Path("/scratch/djy8hg/workdir/MosaicKV/third_party/PrefixKV")
ALIASES = {
    "/sfs/weka/scratch/djy8hg/runs/mosaickv/paper-admissibility-8f20d0e/prefixkv-ppl/work/confs",
    "/sfs/weka/scratch/djy8hg/runs/mosaickv/paper-admissibility-8f20d0e/prefixkv-ppl/retry-1/work/confs",
}
CONFIG_NAME = re.compile(r"confs/prefixkv_llava-v1\.5-(?:7b|13b)_0\.[0-9]\.json")
CATALOG = Path(
    "/scratch/djy8hg/workdir/fox-review-v36/run-20261111-1/history/review/media-gap.json"
)
CATALOG_SHA = "ac552b7dabf6a7f7f57a89ee7f8b55c3b5eaf51c7f52b6b4e68a7d337eed9e68"
REVISIONS = (
    "da103097e2bdcea85a1d104b251c55b42dd3c20d",
    "45d03a6cc41d6ac79f82b4f8f1cbf6f88334b5a0",
    "a05b3932380edb573913b7a670373651afa1613e",
    "0be3133461c1f523a2921940300576ea4efbaf73",
    "4d0a456b3c47686dcca821d42f1aa6b51677fdfe",
    "da3223a97a370874601347e21abffe7ad87b91c5",
    "01c45dd3d37f29535c7d61d844551c515f8f5cdc",
    "02a6ea1b2d2237f8cb04dfbbbaf400ab2c734e74",
)


def signature(path):
    value = path.lstat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def read_regular(path, bound):
    before = signature(path)
    if not stat.S_ISREG(path.lstat().st_mode) or before[2] > bound:
        raise ValueError("registered source metadata exceeds bound or is not regular")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        first = os.fstat(stream.fileno())

        def geometry(value):
            return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]

        if geometry(first) != before or not stat.S_ISREG(first.st_mode):
            raise ValueError("registered source descriptor changed")
        raw = stream.read(bound + 1)
        if len(raw) != before[2] or geometry(os.fstat(stream.fileno())) != before:
            raise ValueError("registered source changed during bounded read")
    if signature(path) != before:
        raise ValueError("registered source changed after bounded read")
    return raw


def git(repo, args, deadline, *, stdin=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("bounded source-recovery deadline")
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        timeout=min(10, remaining),
        check=False,
    )


def prove_alias(row, roots, deadline, *, aliases=ALIASES, repo=PREFIX_REPO, revision=PREFIX_COMMIT):
    path = Path(row["path"])
    if str(path) not in aliases or not any(
        path.parent.resolve(strict=True).is_relative_to(Path(root)) for root in roots
    ):
        raise ValueError("only two explicitly registered source aliases are permitted")
    if not stat.S_ISLNK(path.lstat().st_mode) or signature(path) != row["signature"]:
        raise ValueError("registered source link signature changed")
    first = os.readlink(path)
    target = (repo / "confs").resolve(strict=True)
    if path.resolve(strict=True) != target or not target.is_dir():
        raise ValueError("source alias does not name the pinned configuration tree")
    head = git(repo, ["rev-parse", "HEAD"], deadline)
    tree = git(repo, ["ls-tree", "-rz", revision, "--", "confs"], deadline)
    if head.returncode or head.stdout.decode().strip() != revision or tree.returncode:
        raise ValueError("exact upstream PrefixKV revision required")
    entries, bindings = [], {}
    for raw in tree.stdout.split(b"\0"):
        if not raw:
            continue
        header, name = raw.decode("ascii").split("\t", 1)
        mode, kind, oid = header.split()
        if mode != "100644" or kind != "blob" or CONFIG_NAME.fullmatch(name) is None:
            raise ValueError("unregistered configuration-tree entry")
        file = repo / name
        before = signature(file)
        if not stat.S_ISREG(file.lstat().st_mode) or before[2] > 16384:
            raise ValueError("regular bounded source configuration required")
        raw = read_regular(file, 16384)
        if signature(file) != before:
            raise ValueError("source configuration changed")
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != oid:
            raise ValueError("configuration bytes differ from the pinned source blob")
        values = json.loads(raw)
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in values)
        ):
            raise ValueError("pinned source is not exclusively a numeric layer-budget profile")
        bindings[str(file.resolve())] = hashlib.sha256(raw).hexdigest()
        entries.append(name)
    actual_entries = {"confs/" + x.name for x in target.iterdir()}
    if len(entries) != 20 or set(entries) != actual_entries:
        raise ValueError("unexpected or untracked source-tree files")
    second = os.readlink(path)
    if first != second or signature(path) != row["signature"]:
        raise ValueError("source alias changed during verification")
    hashed = hashlib.sha256(first.encode("utf-8")).hexdigest()
    return dict(
        path=str(path),
        signature=row["signature"],
        proof_kind="pinned_upstream_source_alias",
        whole_file_sha256_before=hashed,
        whole_file_sha256_after=hashed,
        source_hash_kind="symlink_target_utf8",
        bindings=bindings,
        source_revision=revision,
        source_tree_listing_sha256=hashlib.sha256(tree.stdout).hexdigest(),
        configuration_files_verified=len(entries),
        experimental_result_target=False,
        old_unbound_historical_version_byte_identity_claimed=False,
        historical_media_exposure_resolved=False,
        fresh_confirmation_authorized=False,
    )


def recover_sources(source, *, deadline):
    """Reuse the saved backup catalog; inspect Git metadata, not experiment bodies."""
    raw = read_regular(CATALOG, 2**20)
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA:
        raise ValueError("saved backup catalog changed or exceeds bound")
    prior = json.loads(raw)
    candidates = prior["bounded_project_backup_catalog"]["candidates"]
    roots = [Path(root) for root in prior["bounded_project_backup_catalog"]["root_paths"]]
    repositories = {str(Path(source).resolve())}
    for item in candidates:
        path = Path(item["canonical_path"])
        if path.name == ".git" and any(path.is_relative_to(root) for root in roots):
            repositories.add(str(path.parent))
    checked, found, common_seen, errors = [], {}, set(), []
    for name in sorted(repositories):
        if time.monotonic() >= deadline:
            errors.append(dict(reason="source_probe_deadline", unvisited=True))
            break
        try:
            actual = Path(name).resolve(strict=True)
            if not any(actual.is_relative_to(root) for root in roots):
                raise ValueError("repository alias escapes registered project roots")
            result = git(
                name, ["rev-parse", "--path-format=absolute", "--git-common-dir"], deadline
            )
            if result.returncode:
                continue
            common = str(Path(result.stdout.decode().strip()).resolve(strict=True))
            if common in common_seen:
                continue
            common_seen.add(common)
            if not Path(common).is_relative_to(Path("/scratch/djy8hg").resolve()):
                raise ValueError("Git database outside project scratch scope")
            batch = git(
                name,
                ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
                deadline,
                stdin="".join(revision + "\n" for revision in REVISIONS).encode(),
            )
            if batch.returncode:
                raise ValueError("Git object metadata unavailable")
            present = [
                line.split()[0]
                for line in batch.stdout.decode().splitlines()
                if line.endswith(" commit") and line.split()[0] in REVISIONS
            ]
            checked.append(
                dict(
                    repository=name,
                    common_git_dir=common,
                    object_status_sha256=hashlib.sha256(batch.stdout).hexdigest(),
                    exact_commits_available=present,
                )
            )
            for revision in present:
                found.setdefault(revision, []).append(name)
        except (ValueError, OSError, subprocess.TimeoutExpired):
            errors.append(dict(repository=name, reason="bounded_Git_metadata_unavailable"))
    return dict(
        saved_catalog_path=str(CATALOG),
        saved_catalog_sha256=CATALOG_SHA,
        original_catalog_replayed=False,
        inspected_databases=checked,
        errors=errors,
        available_exact_source_commits=found,
        missing_exact_source_commits=[r for r in REVISIONS if r not in found],
        original_execution_journals_recovered=False,
        original_media_bindings_recovered=False,
        original_dirty_snapshot_recovered=False,
        cached_media_is_not_exposure_proof=True,
        globally_fresh=False,
        historical_media_exposure_resolved=False,
        source_group_disjointness_verified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )


def remote_source_attempt(root, revisions, *, deadline):
    """Try exact public Git objects in a NEW bare repository, never user refs.

    No credentials or prompts; no branches, FETCH_HEAD or existing repository
    are changed. Shallow filtered transport avoids downloading model/data blobs.
    Authentication/absence is retained as a blocker, not bypassed.
    """
    if any(revision not in REVISIONS for revision in revisions):
        raise ValueError("only the eight exact registered source revisions may be requested")
    if not revisions:
        return dict(attempted=False, reason="all_exact_sources_already_available")
    repo = Path(root) / "remote-source.git"
    if repo.exists() or not repo.parent.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("new project-scratch bare recovery repository required")
    environment = dict(
        os.environ,
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/bin/false",
        SSH_ASKPASS="/bin/false",
        GIT_LFS_SKIP_SMUDGE="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    begun = time.monotonic()
    init = subprocess.run(
        ["git", "init", "--bare", str(repo)],
        capture_output=True,
        env=environment,
        timeout=10,
        check=False,
    )
    if init.returncode:
        return dict(attempted=False, reason="isolated_recovery_repository_unavailable")
    command = [
        "git",
        "-C",
        str(repo),
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=/bin/false",
        "fetch",
        "--depth=1",
        "--filter=blob:none",
        "--no-tags",
        "--no-write-fetch-head",
        "https://github.com/arupcsedu/MosaicKV.git",
    ]
    attempts = []
    try:
        for revision in revisions:
            remaining = min(12, deadline - time.monotonic(), 90 - (time.monotonic() - begun))
            if remaining <= 0:
                raise TimeoutError("registered source recovery deadline")
            result = subprocess.run(
                [*command, revision],
                capture_output=True,
                env=environment,
                timeout=remaining,
                check=False,
            )
            attempts.append(
                dict(
                    revision=revision,
                    returncode=result.returncode,
                    stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
                    stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
                )
            )
            if result.returncode and re.search(
                rb"Authentication failed|could not read Username|Repository not found",
                result.stderr,
                re.IGNORECASE,
            ):
                break  # Shared repository/access failure; never request credentials.
        check = git(
            repo,
            ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
            deadline,
            stdin="".join(r + "\n" for r in revisions).encode(),
        )
        status = dict(
            attempted=True,
            attempts=attempts,
            available_exact_source_commits=[
                line.split()[0]
                for line in check.stdout.decode().splitlines()
                if line.endswith(" commit") and line.split()[0] in revisions
            ],
        )
    except (subprocess.TimeoutExpired, TimeoutError):
        status = dict(attempted=True, attempts=attempts, timed_out=True)
    return dict(
        status,
        repository=str(repo),
        requested_exact_revisions=revisions,
        wall_seconds=time.monotonic() - begun,
        existing_repository_unchanged=True,
        credentials_requested=False,
        original_media_exposure_proven=False,
    )
