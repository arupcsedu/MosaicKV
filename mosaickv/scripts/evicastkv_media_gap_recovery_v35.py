#!/usr/bin/env python3
"""Write-once historical media-gap evidence; never an exemption or freshness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, write_once

SOURCE = Path(__file__).resolve().parents[2]
RUNS = Path("/scratch/djy8hg/runs/mosaickv")
AUDIT = Path("/scratch/djy8hg/workdir/fox-video-accounting-audit-20261111-1")
VIDEO_MANIFEST = (
    RUNS / "component-ablations-v17/exp3-eaeae8414d-videomme-a-mosaickv_full/manifest.json"
)
VIDEO_TRACE = (
    RUNS
    / "component-ablations-v15/traces/exp3-1d98c49a6d-videomme-a-mosaickv_full"
    / "78beb10e419e52ba.json"
)
DIRTY_SOURCE = "01c45dd3d37f29535c7d61d844551c515f8f5cdc"
DIRTY_FINGERPRINT = "a3b87873ea6fa9828c3d634785d37ba61a9c21ee0bf46fdef08678fa25bbf1f9"
VIDEO_MANIFEST_SHA = "b155703507d1c164c8e6e9cdda9b3c0f450f8a13211291524dc5b95e88439391"
PILOTS = (
    ("17220009", "preflight-qwen-video-v5-da10309", "da103097e2bdcea85a1d104b251c55b42dd3c20d"),
    ("17220227", "preflight-video-full-v6-45d03a6", "45d03a6cc41d6ac79f82b4f8f1cbf6f88334b5a0"),
    ("17220593", "preflight-video-full-v7-a05b393", "a05b3932380edb573913b7a670373651afa1613e"),
    ("17220752", "preflight-video-full-v8-0be3133", "0be3133461c1f523a2921940300576ea4efbaf73"),
    ("17221038", "preflight-qwen-video-v9-4d0a456", "4d0a456b3c47686dcca821d42f1aa6b51677fdfe"),
    ("17228767", "preflight-qwen3-video-v11-da3223a", "da3223a97a370874601347e21abffe7ad87b91c5"),
)
KEY = re.compile(rb'(?<!\\)"([A-Za-z_][A-Za-z0-9_]{0,127})"\s*:')
INTEREST = re.compile(r"media|video|frame|pixel|hash|path|sha256|image", re.I)
FORBIDDEN = re.compile(
    r"question|answer|reference|prediction|caption|prompt|score|logit|token|completion|response|text|content",
    re.I,
)
MEDIA_SUFFIX = re.compile(
    r"(?:\.mp4|\.mkv|\.mov|\.avi|\.webm|\.jpe?g|\.png|\.webp)(?:$|[?#])", re.I
)
SCALAR_FIELD = re.compile(r"path|hash|sha256|recipe|decoder|sampl|fps|filename|file_name", re.I)


def signature(path):
    item = path.stat()
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns


def bounded_json(path, maximum=2**20):
    before = signature(path)
    if before[2] > maximum:
        raise ValueError(f"registered metadata bound exceeded: {path}")
    data = path.read_bytes()
    if signature(path) != before:
        raise ValueError(f"metadata changed during read: {path}")
    return json.loads(data), hashlib.sha256(data).hexdigest()


def project_trace(path, *, max_bytes=256 * 2**20, max_seconds=120, chunk_bytes=2**20):
    """Stream/hash all bytes; decode only bounded eligible scalar metadata.

    This is a lexical evidence projection, not arbitrary-JSON validation. No
    arrays, tensor bodies, captions, questions, answers, or scores are decoded
    or exported. Its output can document a remaining gap, never certify the
    absence of historical exposure or authorize a cohort.
    """
    path = Path(path)
    before = signature(path)
    if before[2] > max_bytes or chunk_bytes < 64:
        raise ValueError("trace exceeds registered bound or chunk is too small")
    begun, total, processed = time.monotonic(), 0, 0
    hashed, counts, scalar_values = hashlib.sha256(), Counter(), {}
    tail = b""
    overlap = 16384
    incomplete_scalars = Counter()

    def consume(data, base, limit):
        for match in KEY.finditer(data):
            absolute = base + match.start()
            if absolute < processed or absolute >= limit:
                continue
            name = match.group(1).decode("ascii")
            if not INTEREST.search(name):
                continue
            counts[name] += 1
            if FORBIDDEN.search(name) or not SCALAR_FIELD.search(name):
                continue
            candidate = data[match.end() : match.end() + 4096].lstrip()
            if not candidate or candidate[:1] in (b"[", b"{"):
                incomplete_scalars[name] += 1
                continue
            try:
                value, _end = json.JSONDecoder().raw_decode(candidate.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                incomplete_scalars[name] += 1
                continue
            if not isinstance(value, (str, int, float, bool, type(None))):
                incomplete_scalars[name] += 1
                continue
            if isinstance(value, str) and len(value) > 2048:
                incomplete_scalars[name] += 1
                continue
            encoded = json.dumps(value, sort_keys=True, allow_nan=False)
            entries = scalar_values.setdefault(name, {})
            if encoded not in entries and len(entries) >= 64:
                incomplete_scalars[name] += 1
                continue
            entries[encoded] = value

    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            if time.monotonic() - begun > max_seconds:
                raise TimeoutError("bounded trace projection deadline reached")
            hashed.update(chunk)
            base = total - len(tail)
            total += len(chunk)
            data = tail + chunk
            limit = max(0, total - overlap)
            consume(data, base, limit)
            processed = limit
            tail = data[-overlap:]
        consume(tail, total - len(tail), total)
    if total != before[2] or signature(path) != before:
        raise ValueError("trace changed during complete metadata projection")
    values = {name: list(entries.values()) for name, entries in sorted(scalar_values.items())}
    media_paths = [
        dict(field=name, value=value)
        for name, entries in values.items()
        for value in entries
        if isinstance(value, str) and MEDIA_SUFFIX.search(value)
    ]
    return dict(
        path=str(path.resolve(strict=True)),
        sha256=hashed.hexdigest(),
        bytes=total,
        complete_byte_stream_read=True,
        lexical_metadata_projection_only=True,
        matching_field_names=sorted(counts),
        matching_field_counts=dict(sorted(counts.items())),
        bounded_scalar_metadata=values,
        skipped_or_over_bound_scalar_counts=dict(incomplete_scalars),
        candidate_media_path_fields=media_paths,
        tensor_arrays_decoded=False,
        benchmark_values_exported=False,
        actual_media_or_video_decoded=False,
        historical_media_exposure_resolved=False,
        maximum_field_name_length=128,
        maximum_scalar_string_length=2048,
    )


def git_available(revision):
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=SOURCE,
            capture_output=True,
            timeout=20,
        ).returncode
        == 0
    )


def project_log(path, *, maximum=2**20):
    """Only source SHAs/error classes/stack locations; never worker prose."""
    path = Path(path)
    report = dict(
        path=str(path),
        lexists=os.path.lexists(path),
        exists=path.is_file(),
        before_media_failure_certified=False,
    )
    if not path.is_file():
        return report
    canonical = path.resolve(strict=True)
    if not canonical.is_relative_to(RUNS.resolve(strict=True)):
        raise ValueError("runtime-log alias escapes registered experiment root")
    before = signature(path)
    if before[2] > maximum:
        report["content_projection_blocked_by_bound"] = True
        return report
    data = path.read_bytes()
    if signature(path) != before:
        raise ValueError("runtime log changed during bounded projection")
    text = data.decode("utf-8", errors="replace")
    report.update(
        canonical_path=str(canonical),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        source_SHA_lines=sorted(set(re.findall(r"(?m)^([a-f0-9]{40})\s*$", text))),
        exception_class_names=sorted(
            set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b", text))
        ),
        stack_locations=[
            dict(file=name, line=int(line))
            for name, line in re.findall(r'File "([^"\r\n]{1,2048})", line (\d+)', text)
        ],
        source_archive_paths_declared=sorted(
            set(re.findall(r"(/[^\s\"'|]{1,2048}\.(?:bundle|patch|diff|tar\.gz|tar\.xz))", text))
        ),
    )
    return report


def runtime_logs():
    command = [
        "sacct",
        "-j",
        ",".join(item[0] for item in PILOTS),
        "-X",
        "-nP",
        "--format=JobID,JobIDRaw,State,StdOut%300,StdErr%300",
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    records = []
    if done.returncode == 0:
        for line in done.stdout.splitlines():
            fields = line.split("|")
            if len(fields) == 6 and fields[-1] == "":
                fields = fields[:-1]
            if len(fields) != 5:
                continue
            identity, raw, state, stdout, stderr = fields
            array = re.fullmatch(r"(\d+)_(\d+)", identity)
            logs = {}
            for kind, template in (("stdout", stdout), ("stderr", stderr)):
                if not template.startswith("/"):
                    logs[kind] = dict(declared_path=template, resolvable=False)
                    continue
                expanded = template.replace("%j", raw)
                if array:
                    expanded = expanded.replace("%A", array[1]).replace("%a", array[2])
                if "%" in expanded:
                    logs[kind] = dict(
                        declared_path=template,
                        resolvable=False,
                        reason="unexpanded_scheduler_pattern",
                    )
                else:
                    logs[kind] = project_log(expanded)
            records.append(dict(job_id=identity, job_id_raw=raw, state=state, logs=logs))
    return dict(
        command=command,
        returncode=done.returncode,
        raw_accounting_sha256=hashlib.sha256(done.stdout.encode()).hexdigest(),
        records=records,
        absence_not_nonexecution_proof=True,
    )


def recover():
    revision = clean_source(SOURCE)
    manifest, manifest_sha = bounded_json(VIDEO_MANIFEST)
    if (
        manifest_sha != VIDEO_MANIFEST_SHA
        or manifest["source"]["git_sha"] != DIRTY_SOURCE
        or manifest["source"]["patch_sha"] != DIRTY_FINGERPRINT
    ):
        raise ValueError("registered dirty video provenance changed")
    bindings = {str(VIDEO_MANIFEST.resolve()): manifest_sha}
    for path in (
        AUDIT / "VIDEO_PROVENANCE_FINDINGS.md",
        AUDIT / "array-identities.json",
        AUDIT / "accounting.json",
        AUDIT / "main-array.json",
    ):
        bindings[str(path.resolve(strict=True))] = hashlib.sha256(path.read_bytes()).hexdigest()
    pilots = []
    for job, name, source in PILOTS:
        directory = RUNS / "main-quality-campaigns" / name
        logs = [project_log(path) for path in sorted(RUNS.glob(f"slurm-{job}_*.submit.log"))]
        pilots.append(
            dict(
                job=job,
                directory=str(directory),
                directory_lexists=os.path.lexists(directory),
                directory_exists=directory.exists(),
                jobs_index_exists=(directory / "jobs.jsonl").is_file(),
                matrix_manifest_exists=(directory / "matrix_manifest.json").is_file(),
                recorded_clean_source_SHA=source,
                recorded_source_commit_available=git_available(source),
                submit_logs=logs,
                original_media_bindings_recovered=False,
            )
        )
    trace = project_trace(VIDEO_TRACE)
    if trace["bytes"] != 207348619:
        raise ValueError("registered previously skipped video trace size changed")
    bindings[trace["path"]] = trace["sha256"]
    old = subprocess.run(
        ["git", "show", f"{DIRTY_SOURCE}:mosaickv/src/mosaickv/manifest.py"],
        cwd=SOURCE,
        capture_output=True,
        timeout=20,
    )
    return dict(
        source_sha=revision,
        status="bounded_actual_media_gap_recovery_not_freshness_certification",
        complete=True,
        original_report_bindings=bindings,
        pilot_roots=pilots,
        runtime_accounting=runtime_logs(),
        previously_skipped_trace=trace,
        dirty_video=dict(
            manifest_path=str(VIDEO_MANIFEST),
            manifest_sha256=manifest_sha,
            base_SHA=DIRTY_SOURCE,
            base_commit_available=git_available(DIRTY_SOURCE),
            recorded_fingerprint=DIRTY_FINGERPRINT,
            fingerprint_scheme=(
                "porcelain-z status + HEAD binary diff + staged binary diff + "
                "every untracked filename/content/link target; NOT patch-file SHA"
            ),
            source_manifest_helper_available=old.returncode == 0,
            source_manifest_helper_sha256=hashlib.sha256(old.stdout).hexdigest()
            if old.returncode == 0
            else None,
            archive_path_declared_in_manifest=False,
            original_dirty_worktree_state_recovered=False,
        ),
        missing_evidence=[
            "six started pilot job indices and model/media execution journals",
            "actual source MP4/decoded-frame hashes and exact sampling recipe",
            "original dirty status/staging/untracked contents matching recorded fingerprint",
        ],
        cached_video_availability_is_not_exposure_proof=True,
        dataset_namespace_difference_is_not_source_group_disjointness=True,
        globally_fresh=False,
        media_exposure_resolved=False,
        confirmation_blocked=True,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
        GPU_jobs_submitted=0,
        models_loaded=False,
    )


def publish(output, report):
    for key in (
        "globally_fresh",
        "media_exposure_resolved",
        "materialization_authorized",
        "fresh_confirmation_authorized",
        "paper_campaign_authorized",
    ):
        if report.get(key) is not False:
            raise ValueError("gap recovery cannot authorize or waive downstream freshness")
    write_once(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = Path("/scratch/djy8hg").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error("new immutable scratch output file required; never home")
    report = recover()
    publish(output, report)
    print(
        json.dumps(
            dict(
                output=str(output),
                report_sha256=digest(report),
                status=report["status"],
                globally_fresh=False,
            )
        )
    )


if __name__ == "__main__":
    main()
