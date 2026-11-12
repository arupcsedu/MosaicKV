"""CPU synthetic media projections; no historic scans, pixels or datasets."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_media_gap_recovery_v35.py"
    spec = importlib.util.spec_from_file_location("media_gap_v35_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_streams_hashes_whole_minified_trace_without_exporting_answers(
    protocol, tmp_path
):
    path = tmp_path / "trace.json"
    original = dict(
        question="SECRET QUESTION",
        answer="SECRET ANSWER",
        pixel_values=list(range(10000)),
        media_path="/scratch/example.mp4",
        frame_recipe="uniform_8",
        trace_path="/scratch/result.json",
        image_video_encoder=0.3,
        answer_media_path="DO_NOT_EXPORT",
        embedded='escaped \\"media_path\\": \\"not a field\\"',
    )
    data = json.dumps(original, separators=(",", ":")).encode()
    path.write_bytes(data)
    report = protocol.project_trace(path, chunk_bytes=8192)
    assert report["complete_byte_stream_read"] is True
    assert report["sha256"] == hashlib.sha256(data).hexdigest()
    assert report["candidate_media_path_fields"] == [
        dict(field="media_path", value="/scratch/example.mp4")
    ]
    assert report["bounded_scalar_metadata"]["frame_recipe"] == ["uniform_8"]
    assert "answer_media_path" not in report["bounded_scalar_metadata"]
    serialized = json.dumps(report)
    assert "SECRET QUESTION" not in serialized and "SECRET ANSWER" not in serialized
    assert "DO_NOT_EXPORT" not in serialized
    assert report["tensor_arrays_decoded"] is False
    assert report["historical_media_exposure_resolved"] is False


@pytest.mark.parametrize("chunk", (64, 128, 8192, 16384, 32768))
def test_metadata_key_and_string_value_split_across_chunk_boundaries(protocol, tmp_path, chunk):
    path = tmp_path / "split.json"
    value = dict(
        prefix=list(range(1000)),
        media_path="/scratch/video.mp4",
        frame_hash="a" * 64,
        trace_path="/scratch/result.json",
    )
    path.write_text(json.dumps(value, separators=(",", ":")))
    report = protocol.project_trace(path, chunk_bytes=chunk)
    assert report["matching_field_counts"]["media_path"] == 1
    assert report["matching_field_counts"]["frame_hash"] == 1
    assert report["bounded_scalar_metadata"]["media_path"] == ["/scratch/video.mp4"]


def test_large_scalar_arrays_and_overlong_strings_are_not_decoded(protocol, tmp_path):
    path = tmp_path / "bounded.json"
    path.write_text(
        json.dumps(
            dict(
                frame_paths=["/scratch/a.png"] * 5000,
                video_path="x" * 5000,
                trace_path="/scratch/no-media.json",
            )
        )
    )
    report = protocol.project_trace(path)
    assert report["candidate_media_path_fields"] == []
    assert report["skipped_or_over_bound_scalar_counts"]["frame_paths"] == 1
    assert report["skipped_or_over_bound_scalar_counts"]["video_path"] == 1


def test_byte_and_time_limits_fail_closed(protocol, tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_text('{"media_path":"/scratch/a.mp4"}')
    with pytest.raises(ValueError, match="bound"):
        protocol.project_trace(path, max_bytes=1)
    times = iter((0, 2))
    monkeypatch.setattr(protocol.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        protocol.project_trace(path, max_seconds=1)


def test_gap_report_is_write_once_and_never_activates_freshness(protocol, tmp_path):
    report = dict(
        globally_fresh=False,
        media_exposure_resolved=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
    )
    path = tmp_path / "report.json"
    protocol.publish(path, report)
    with pytest.raises(FileExistsError):
        protocol.publish(path, report)
    with pytest.raises(ValueError, match="cannot authorize"):
        protocol.publish(tmp_path / "invalid.json", dict(report, globally_fresh=True))


def test_json_artifact_path_is_not_mistaken_for_source_media(protocol, tmp_path):
    path = tmp_path / "trace.json"
    path.write_text('{"image_video_encoder":0.25,"trace_path":"/scratch/trace.json"}')
    report = protocol.project_trace(path, chunk_bytes=64)
    assert report["matching_field_names"] == ["image_video_encoder", "trace_path"]
    assert report["candidate_media_path_fields"] == []
    assert report["bounded_scalar_metadata"] == {"trace_path": ["/scratch/trace.json"]}


def test_runtime_log_projection_does_not_export_worker_prose_or_certify_stage(
    protocol, tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol, "RUNS", tmp_path)
    path = tmp_path / "stderr.log"
    path.write_text(
        'PRIVATE QUESTION AND ANSWER\nFile "/scratch/module.py", line 12\n'
        + "a" * 40
        + "\nValueError: PRIVATE CONTENT\n"
    )
    report = protocol.project_log(path)
    assert report["source_SHA_lines"] == ["a" * 40]
    assert report["exception_class_names"] == ["ValueError"]
    assert report["stack_locations"] == [dict(file="/scratch/module.py", line=12)]
    assert report["before_media_failure_certified"] is False
    assert "PRIVATE" not in json.dumps(report)


def test_accounting_keeps_empty_stderr_and_expands_only_recorded_array_identity(
    protocol, tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol, "RUNS", tmp_path)
    path = tmp_path / "slurm-17220009_0.submit.log"
    path.write_text("a" * 40 + "\n")
    line = f"17220009_0|17220056|FAILED|{tmp_path}/slurm-%A_%a.submit.log||\n"
    monkeypatch.setattr(
        protocol.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=line),
    )
    report = protocol.runtime_logs()
    assert len(report["records"]) == 1
    item = report["records"][0]
    assert item["logs"]["stdout"]["path"] == str(path)
    assert item["logs"]["stderr"]["resolvable"] is False
    assert item["logs"]["stdout"]["source_SHA_lines"] == ["a" * 40]
