"""File-bound out-of-core identity projection; never a freshness permission."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path

import evicastkv_coverage_resolution_v36 as common
import evicastkv_identity_sink_v44 as sink
import evicastkv_spill_projection_v44 as projection

MAXIMUM_BYTES = 2 * 2**30


class ProjectionFailure(ValueError):
    def __init__(self, category, stats):
        super().__init__(category)
        self.statistics = stats


def geometry(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def project_file(row, roots, deadline, output_root, *, maximum_bytes=MAXIMUM_BYTES):
    path = common.canonical(row["path"], roots)
    expected = row["signature"]
    if (
        str(path) != row["path"]
        or geometry(path.lstat()) != expected
        or not stat.S_ISREG(path.lstat().st_mode)
        or expected[2] > maximum_bytes
    ):
        raise ProjectionFailure("registered_source_changed_or_exceeds_remaining_budget", {})
    output_root = Path(output_root).resolve(strict=True)
    scratch = Path("/scratch/djy8hg").resolve(strict=True)
    temporary = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not output_root.is_relative_to(scratch) or not temporary.is_relative_to(scratch):
        raise ValueError("source proof workspace and outputs must stay on scratch")
    reader = None
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="evicastkv-json-v44-", dir=temporary) as folder:
            workspace = Path(folder)
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(descriptor, "rb") as stream:
                first = os.fstat(stream.fileno())
                if geometry(first) != expected or not stat.S_ISREG(first.st_mode):
                    raise ValueError("registered_source_descriptor_changed")
                reader = projection.ProjectionReader(
                    stream,
                    deadline=deadline,
                    maximum_bytes=maximum_bytes,
                    chunk_bytes=256 * 2**10,
                    workspace=workspace,
                )
                with reader:
                    node = reader.value()
                    node = None if node is projection.base.DROP else node
                    reader.spaces()
                    if reader.peek() is not None:
                        raise ValueError("trailing_JSON_payload")
                    root_digest = reader.descriptor_digest(node)
                    result = sink.collect_to_shards(node, reader, workspace, output_root)
                    reader.assert_spills_verified()
                    reader.check_deadline()
                    final = os.fstat(stream.fileno())
                    if geometry(final) != expected or geometry(path.lstat()) != expected:
                        raise ValueError("source_changed_during_complete_projection")
                    if reader.stats["bytes_read"] != expected[2]:
                        raise ValueError("complete_source_length_differs")
                    return dict(
                        result,
                        path=str(path),
                        signature_before=expected,
                        signature_after=geometry(path.lstat()),
                        fd_signature_before=geometry(first),
                        fd_signature_after=geometry(final),
                        whole_file_sha256=reader.hash.hexdigest(),
                        bytes_read=reader.stats["bytes_read"],
                        total_input_bytes=reader.budget.used,
                        projected_metadata_sha256=root_digest,
                        projection_storage_boundary="redacted_tree_and_hashed_external_arrays",
                        statistics=dict(reader.stats),
                        hash_passes=1,
                        independent_second_pass_verified=False,
                        wall_seconds=time.monotonic() - started,
                        parser_schema="evicastkv_bounded_external_JSON_v44",
                        complete_document_validated=True,
                        all_spill_replays_verified=True,
                        forbidden_value_scalars_JSON_decoded=False,
                        fresh_confirmation_authorized=False,
                        materialization_authorized=False,
                    )
    except Exception as error:
        stats = dict(reader.stats) if reader is not None else {}
        if reader is not None:
            stats["total_input_bytes"] = reader.budget.used
        category = type(error).__name__
        # Reader/sink errors carry no original scalar snippets. Only the class
        # is published; unknown parser/SQLite exceptions stay explicit negatives.
        raise ProjectionFailure(category, stats) from error
