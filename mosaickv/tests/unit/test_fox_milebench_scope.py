"""Replacement-scope preparation never turns unused IDs into fresh evidence."""

import copy
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mosaickv.evaluation.fox_milebench_scope import (
    DISABLED_ACTIONS,
    PROJECTED_COLUMNS,
    PROPOSED_TASKS,
    read_projected_catalog,
    safe_relative,
    summarize_availability,
    validate_protocol,
)


def protocol():
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (
            root / "configs/experiments/evicastkv_fox/milebench_additional_subtasks_v1.json"
        ).read_text()
    )


def source_records(task, count=2):
    return [
        dict(
            sample_id=i,
            task_instruction="FORBIDDEN PROMPT VALUE",
            response="FORBIDDEN REFERENCE VALUE",
            task_instance=dict(
                images_path=[f"video_{i}/frame.jpg"],
                combined_1_images=[f"{i}.jpg"],
                choice_list=["A", "B"] if PROPOSED_TASKS[task] == "multi-choice" else [],
                context="FORBIDDEN CONTEXT VALUE",
            ),
        )
        for i in range(count)
    ]


def catalog(task, records=None):
    stream = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records or source_records(task)), stream)
    stream.seek(0)
    underlying = pq.ParquetFile(stream)

    class AuditedParquet:
        def read(self, *, columns):
            assert columns == list(PROJECTED_COLUMNS)
            return underlying.read(columns=columns)

    return read_projected_catalog(AuditedParquet(), task=task, question_type=PROPOSED_TASKS[task])


@pytest.mark.parametrize("task", PROPOSED_TASKS)
def test_actual_nested_parquet_projects_only_inputs(task):
    out = catalog(task)
    encoded = json.dumps(out)
    assert "FORBIDDEN" not in encoded
    assert len(out["rows"]) == 2
    assert out["labels_decoded"] is False
    assert out["fresh_confirmation_certified"] is False
    assert out["rows"][0]["source_media_verified"] is False
    assert out["metric"] == ("accuracy" if PROPOSED_TASKS[task] == "multi-choice" else "rouge_l_f1")
    assert "official_annotation_meta_pending" in out["question_type_evidence"]


@pytest.mark.parametrize("value", ["../x", "a/../x", "/x", "a\\x", "a//x", "./a", "", "a\x00b"])
def test_unsafe_media_paths_fail(value):
    with pytest.raises(ValueError, match="media path"):
        safe_relative(value)


@pytest.mark.parametrize("mutation", ["duplicate", "bad_id", "choices", "montage", "path"])
def test_malformed_real_source_rows_fail_closed(mutation):
    rows = source_records("CharacterOrder")
    if mutation == "duplicate":
        rows[1]["sample_id"] = 0
    elif mutation == "bad_id":
        rows[0]["sample_id"] = -1
    elif mutation == "choices":
        rows[0]["task_instance"]["choice_list"] = []
    elif mutation == "montage":
        rows[0]["task_instance"]["combined_1_images"] = ["0.jpg", "1.jpg"]
    else:
        rows[0]["task_instance"]["images_path"] = ["../../escape.jpg"]
    with pytest.raises(ValueError):
        catalog("CharacterOrder", rows)


@pytest.mark.parametrize("action", DISABLED_ACTIONS)
def test_audit_rejects_downstream_authorization(action):
    p = protocol()
    validate_protocol(p)
    p[action] = True
    with pytest.raises(ValueError, match="disabled"):
        validate_protocol(p)


def test_unknown_scope_and_output_columns_rejected():
    p = protocol()
    p["proposed_tasks"]["ActionSequence"] = "multi-choice"
    with pytest.raises(ValueError, match="subtask mix"):
        validate_protocol(p)
    p = protocol()
    p["projected_columns"].append("response")
    with pytest.raises(ValueError, match="input columns"):
        validate_protocol(p)


def test_all_ids_unused_and_even_complete_id_inventory_never_certify_media():
    catalogs = [catalog(t) for t in PROPOSED_TASKS]
    inventory = dict(MileBench_consumed_ids=[], complete_scan=True, errors=[])
    result = summarize_availability(catalogs, inventory, protocol())
    assert all(t["id_only_available_upper_bound"] == 2 for t in result["tasks"])
    assert all(t["certified_fresh_media_groups"] == 0 for t in result["tasks"])
    assert not result["confirmation_authorized"]
    assert not result["materialization_authorized"]
    assert not result["gpu_execution_authorized"]
    assert result["original_four_task_result_preserved"]


def test_known_used_ids_removed_and_unresolved_history_is_not_hidden():
    catalogs = [catalog(t) for t in PROPOSED_TASKS]
    inventory = dict(
        MileBench_consumed_ids=["CharacterOrder-0000"],
        complete_scan=False,
        errors=[],
        unclassified_workload_roots=["unclassified-run"],
    )
    result = summarize_availability(catalogs, inventory, protocol())
    row = next(t for t in result["tasks"] if t["task"] == "CharacterOrder")
    assert row["id_only_available_upper_bound"] == 1 and row["known_consumed_ids"] == 1
    assert not result["history_inventory_complete"]
    assert result["unresolved_history_roots"] == ["unclassified-run"]
    changed = copy.deepcopy(catalogs)
    changed[0]["rows"][0]["sample_id"] = "changed"
    with pytest.raises(ValueError, match="content changed"):
        summarize_availability(changed, inventory, protocol())
    with pytest.raises(ValueError, match="exactly once"):
        summarize_availability(catalogs + catalogs[:1], inventory, protocol())
