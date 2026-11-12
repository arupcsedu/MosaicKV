"""Synthetic Parquet footer fixtures only; no historical producer execution."""

import json
import time
from pathlib import Path

import evicastkv_parquet_residuals_v41 as helper
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def fixture(tmp_path, *, metadata_change=None, frame=None):
    frame = pd.DataFrame() if frame is None else frame
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if metadata_change is not None:
        metadata = dict(table.schema.metadata)
        metadata_change(metadata)
        table = table.replace_schema_metadata(metadata)
    path = tmp_path / "aggregate.parquet"
    pq.write_table(table, path)
    row = dict(
        path=str(path),
        signature=helper.common.scanner.signature(path),
        original_error_sha256="a" * 64,
    )
    return row, [tmp_path.resolve()], time.monotonic() + 10


def test_complete_empty_footer_has_no_row_values_or_identity_fields(tmp_path, monkeypatch):
    row, roots, deadline = fixture(tmp_path)
    monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: pytest.fail("no row decoding"))
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("no pandas rows"))
    result = helper.prove(row, roots, deadline)
    assert result["source_proof"]["rows"] == result["source_proof"]["columns"] == 0
    assert result["source_proof"]["row_groups_decoded"] == 0
    assert all(not value for value in result["identities"].values())
    assert result["whole_file_sha256_before"] == result["whole_file_sha256_after"]
    assert result["source_proof"]["historical_producer_inferred"] is False
    assert result["old_unbound_historical_version_byte_identity_claimed"] is False
    assert result["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"method": ["Top-K"]}),
        pd.DataFrame(columns=["sample_123456"]),
        pd.DataFrame({"score": [1.0]}),
    ],
)
def test_nonempty_or_dynamically_named_columns_remain_unsupported(tmp_path, frame):
    row, roots, deadline = fixture(tmp_path, frame=frame)
    with pytest.raises(ValueError, match="unbound nonempty"):
        helper.prove(row, roots, deadline)


def test_unknown_footer_key_is_not_silently_treated_as_identity_free(tmp_path):
    row, roots, deadline = fixture(
        tmp_path,
        metadata_change=lambda values: values.update({b"sample_id": b"synthetic-untrusted"}),
    )
    with pytest.raises(ValueError, match="footer key"):
        helper.prove(row, roots, deadline)


@pytest.mark.parametrize("change", ["index", "column", "attribute", "unknown-key", "version"])
def test_untrusted_pandas_footer_names_values_and_attributes_rejected(tmp_path, change):
    def mutate(values):
        metadata = json.loads(values[b"pandas"])
        if change == "index":
            metadata["index_columns"] = ["synthetic-source-group"]
        elif change == "column":
            metadata["column_indexes"] = [{"name": "123456"}]
        elif change == "attribute":
            metadata["attributes"] = {"sample_id": "123456"}
        elif change == "unknown-key":
            metadata["123456"] = "untrusted"
        else:
            metadata["creator"]["version"] = "untrusted-identifier"
        values[b"pandas"] = json.dumps(metadata).encode()

    row, roots, deadline = fixture(tmp_path, metadata_change=mutate)
    with pytest.raises(ValueError):
        helper.prove(row, roots, deadline)


def test_unreferenced_hidden_bytes_fail_exact_empty_physical_geometry(tmp_path):
    row, roots, deadline = fixture(tmp_path)
    path = Path(row["path"])
    raw = path.read_bytes()
    path.write_bytes(raw[:4] + b"SYNTHETIC-HIDDEN-PAYLOAD" + raw[4:])
    row["signature"] = helper.common.scanner.signature(path)
    with pytest.raises(ValueError, match="unreferenced"):
        helper.prove(row, roots, deadline)


def test_registered_signature_and_expected_current_hash_are_fail_closed(tmp_path):
    row, roots, deadline = fixture(tmp_path)
    row["signature"][3] += 1
    with pytest.raises(ValueError, match="signature"):
        helper.prove(row, roots, deadline)
    row["signature"] = helper.common.scanner.signature(Path(row["path"]))
    row["file_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        helper.prove(row, roots, deadline)


def test_no_unregistered_root_or_expired_transaction(tmp_path):
    row, roots, _deadline = fixture(tmp_path)
    with pytest.raises(TimeoutError):
        helper.prove(row, roots, time.monotonic() - 1)
    with pytest.raises(ValueError):
        helper.prove(row, [tmp_path / "outside"], time.monotonic() + 10)


@pytest.mark.parametrize("field", ["library", "version", "pandas_version"])
def test_closed_footer_never_decodes_untrusted_nested_values(field, monkeypatch):
    metadata = dict(
        index_columns=[],
        column_indexes=[],
        columns=[],
        attributes={},
        creator=dict(library="pyarrow", version="25.0.0"),
        pandas_version="2.3.3",
    )
    if field == "pandas_version":
        metadata[field] = {"answer": "PRIVATE"}
    else:
        metadata["creator"][field] = {"answer": "PRIVATE"}
    raw = json.dumps(metadata).encode()
    monkeypatch.setattr(json, "loads", lambda *a, **k: pytest.fail("untrusted scalar decoded"))
    with pytest.raises(ValueError, match="closed empty pandas"):
        helper._pandas(raw, time.monotonic() + 10)


def test_closed_footer_requires_no_json_value_decoder(monkeypatch):
    metadata = dict(
        index_columns=[],
        column_indexes=[],
        columns=[],
        attributes={},
        creator=dict(library="pyarrow", version="25.0.0"),
        pandas_version="2.3.3",
    )
    raw = json.dumps(metadata).encode()
    monkeypatch.setattr(json, "loads", lambda *a, **k: pytest.fail("footer values decoded"))
    assert len(helper._pandas(raw, time.monotonic() + 10)) == 64
