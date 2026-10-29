"""Reporting must preserve scientific stopping and freshness controls."""

import pytest
from evicastkv_review_closure_v2 import (
    coco_identity,
    freshness_gate,
    projected_ids,
    strict_positive,
    summarize_activity,
    worker_bindings,
)


@pytest.mark.parametrize("low,passes", [(-1.0, False), (0.0, False), (0.001, True)])
def test_zero_touching_interval_is_not_superiority(low, passes):
    assert strict_positive(low) is passes


def test_id_availability_does_not_certify_freshness():
    assert not freshness_gate(available_ids=36408, unresolved_roots=[], global_media_complete=False)
    assert not freshness_gate(
        available_ids=36408, unresolved_roots=["unknown"], global_media_complete=True
    )
    assert not freshness_gate(available_ids=255, unresolved_roots=[], global_media_complete=True)
    assert freshness_gate(available_ids=256, unresolved_roots=[], global_media_complete=True)


def test_source_image_identity_must_agree_with_filename():
    row = dict(
        question_id="COCO_val2014_000000000123.jpg",
        file_name="COCO_val2014_000000000123.jpg",
        id=123,
    )
    assert coco_identity(row) == row["file_name"]
    assert coco_identity(dict(row, id=37)) == row["file_name"]
    with pytest.raises(ValueError):
        coco_identity(dict(row, question_id="alias"))


def test_reanalysis_retains_original_gpu_import_paths():
    spec = {"imported_workers": {"0": {"parent_root": "original", "source_sha": "producer"}}}
    result = worker_bindings(spec, "derived-analysis", "producer")
    assert result["0"]["parent_root"] == "original"
    assert result["1"]["parent_root"] == "derived-analysis"


def test_activity_does_not_turn_gqa_pooling_into_individual_head_evidence():
    data = [
        dict(
            cap=40,
            variant="both",
            policy=dict(
                grouped_query_heads_averaged=True,
                individual_prompt_positions=True,
                evidence_preparation_seconds=1.0,
                query_guard=dict(routes=[dict(rolled_back=True), dict(rolled_back=False)]),
            ),
        )
    ]
    row = summarize_activity(data)[0]
    assert row["grouped_query_heads_averaged"] is True
    assert row["rolled_back_routes"] == 1
    assert row["guarded_routes"] == 2


def test_arrow_projection_ignores_media_and_reference_payloads(tmp_path):
    import pyarrow as pa

    table = pa.table(
        dict(
            question_id=["COCO_val2014_000000000123.jpg"],
            file_name=["COCO_val2014_000000000123.jpg"],
            id=[37],
            image=[b"deliberately not a decodable image"],
            answer=[["must not be read"]],
        )
    )
    path = tmp_path / "catalog.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    rows = projected_ids(path)
    assert rows == [
        dict(
            question_id=table["question_id"][0].as_py(),
            file_name=table["file_name"][0].as_py(),
            id=37,
        )
    ]
    assert "answer" not in rows[0] and "image" not in rows[0]


def test_finite_cpu_submission_analyzes_all_producers_afterany(tmp_path, monkeypatch):
    import json

    import evicastkv_review_closure_v2 as closure

    monkeypatch.setattr(closure, "clean_source", lambda *_: "a" * 40)
    commands = []

    def submitted(command, **kwargs):
        commands.append(command)
        return f"{100 + len(commands)}\n"

    monkeypatch.setattr(closure.subprocess, "check_output", submitted)
    root = tmp_path / "new-output"
    closure.submit(root)
    assert len(commands) == 6
    assert "--dependency=afterany:101:102:103:104:105" in commands[-1]
    assert all(not any("gres" in x or "gpus" in x for x in c) for c in commands)
    assert json.loads((root / "submission.json").read_text())["gpu_jobs"] == 0
    with pytest.raises(FileExistsError):
        closure.submit(root)
