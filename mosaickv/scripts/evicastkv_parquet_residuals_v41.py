#!/usr/bin/env python3
"""Conservative residual-Parquet proofs; never decode row groups or answers.

The fast path proves an entirely empty physical file with closed Arrow/pandas
footer metadata. Nonempty aggregates require a separately registered exact
historical producer/artifact chain; current code or a filename is not a pin.
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path

import evicastkv_coverage_resolution_v36 as common

from mosaickv.evaluation.fox_campaign import digest, sha

MAX_BYTES = 2 * 2**20
MAX_PANDAS_BYTES = 64 * 2**10
VERSION = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\Z")
PANDAS_KEYS = {
    "index_columns",
    "column_indexes",
    "columns",
    "attributes",
    "creator",
    "pandas_version",
}
# This class is intentionally narrower than arbitrary JSON. Empty Arrow tables
# emit this fixed metadata shape; accepting a new shape needs a separate proof.
# Values are checked as bytes BEFORE any scalar/object decoding.
_W = rb"[ \t\r\n]*"
_VERSION = rb"([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})"
EMPTY_PANDAS = re.compile(
    rb"\{"
    + _W
    + rb'"index_columns"'
    + _W
    + rb":"
    + _W
    + rb"\["
    + _W
    + rb"\]"
    + _W
    + rb","
    + _W
    + rb'"column_indexes"'
    + _W
    + rb":"
    + _W
    + rb"\["
    + _W
    + rb"\]"
    + _W
    + rb","
    + _W
    + rb'"columns"'
    + _W
    + rb":"
    + _W
    + rb"\["
    + _W
    + rb"\]"
    + _W
    + rb","
    + _W
    + rb'"attributes"'
    + _W
    + rb":"
    + _W
    + rb"\{"
    + _W
    + rb"\}"
    + _W
    + rb","
    + _W
    + rb'"creator"'
    + _W
    + rb":"
    + _W
    + rb"\{"
    + _W
    + rb'"library"'
    + _W
    + rb":"
    + _W
    + rb'"pyarrow"'
    + _W
    + rb","
    + _W
    + rb'"version"'
    + _W
    + rb":"
    + _W
    + rb'"'
    + _VERSION
    + rb'"'
    + _W
    + rb"\}"
    + _W
    + rb","
    + _W
    + rb'"pandas_version"'
    + _W
    + rb":"
    + _W
    + rb'"'
    + _VERSION
    + rb'"'
    + _W
    + rb"\}"
    + _W
    + rb"\Z"
)
IDENTITIES = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("finite residual-Parquet proof deadline reached")


def _pandas(raw, deadline):
    if not isinstance(raw, bytes) or len(raw) > MAX_PANDAS_BYTES:
        raise ValueError("bounded standard pandas footer metadata required")
    _check(deadline)
    match = EMPTY_PANDAS.fullmatch(raw)
    if match is None:
        raise ValueError("closed empty pandas metadata grammar required")
    _check(deadline)
    return digest(
        dict(
            index_columns=[],
            column_indexes=[],
            columns=[],
            attributes={},
            creator=dict(library="pyarrow", version=match.group(1).decode("ascii")),
            pandas_version=match.group(2).decode("ascii"),
        )
    )


def prove(row, roots, deadline):
    """Return a current-file proof or a fail-closed unsupported exception."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _check(deadline)
    path = common.canonical(row["path"], roots)
    if ".codex" in path.parts or path.name in {"auth.json", "credentials.json"}:
        raise ValueError("credential paths cannot supply residual proofs")
    signature = common.scanner.signature(path)
    if signature != row["signature"] or path.suffix != ".parquet":
        raise ValueError("registered actual Parquet signature/path required")
    file = pq.ParquetFile(path)
    metadata, schema = file.metadata, file.schema_arrow
    if metadata.num_rows != 0 or metadata.num_columns != 0 or len(schema) != 0:
        raise ValueError(
            "unbound nonempty historical Parquet producer; schema alone is insufficient"
        )
    if any(
        metadata.row_group(i).num_rows or metadata.row_group(i).num_columns
        for i in range(metadata.num_row_groups)
    ):
        raise ValueError("row-group payload exists despite an empty file header")
    if signature[2] != metadata.serialized_size + 12 or signature[2] > MAX_BYTES:
        raise ValueError("unreferenced payload or unsupported empty-footer byte geometry")
    if (
        re.fullmatch(
            r"parquet-cpp-arrow version [0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
            metadata.created_by or "",
        )
        is None
    ):
        raise ValueError("unknown or identity-bearing Parquet creator string")
    values = metadata.metadata or {}
    if set(values) != {b"ARROW:schema", b"pandas"}:
        raise ValueError("unreviewed footer key/name requires explicit identity projection")
    pandas_sha = _pandas(values[b"pandas"], deadline)
    arrow_bytes = base64.b64decode(values[b"ARROW:schema"], validate=True)
    reader = pa.BufferReader(arrow_bytes)
    restored = pa.ipc.read_schema(reader)
    if (
        reader.tell() != len(arrow_bytes)
        or len(restored) != 0
        or restored.metadata != {b"pandas": values[b"pandas"]}
        or schema.metadata != restored.metadata
    ):
        raise ValueError("opaque or dynamically named Arrow schema metadata")
    _check(deadline)
    expected = row.get("file_sha256", row.get("current_file_sha256"))
    raw, before = common.read_bytes(path, bound=MAX_BYTES, expected=expected)
    if (
        raw[:4] != b"PAR1"
        or raw[-4:] != b"PAR1"
        or int.from_bytes(raw[-8:-4], "little") != metadata.serialized_size
    ):
        raise ValueError("exact footer-only physical layout required")
    _check(deadline)
    _after_raw, after = common.read_bytes(path, bound=MAX_BYTES, expected=before)
    if common.scanner.signature(path) != signature:
        raise ValueError("source changed during footer transaction")
    errors = row.get("original_error_sha256s", [row.get("original_error_sha256")])
    if (
        not isinstance(errors, list)
        or not errors
        or any(
            not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
            for value in errors
        )
        or len(set(errors)) != len(errors)
    ):
        raise ValueError("exact original error bindings required")
    _check(deadline)
    return dict(
        path=str(path),
        signature=signature,
        kind="closed_parquet",
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        original_error_sha256s=errors,
        identities={name: [] for name in IDENTITIES},
        resolved_as="closed_current_empty_Arrow_footer_no_data_pages",
        source_proof=dict(
            proof_kind="complete_footer_only_physical_content",
            historical_producer_inferred=False,
            row_groups_decoded=0,
            rows=0,
            columns=0,
            pandas_metadata_sha256=pandas_sha,
            Arrow_schema_sha256=digest(list(arrow_bytes)),
            producer_code_sha256=sha(Path(__file__)),
        ),
        bindings={str(path): before, str(Path(__file__).resolve()): sha(Path(__file__))},
        old_unbound_historical_version_byte_identity_claimed=False,
        raw_or_answer_values_exported=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
