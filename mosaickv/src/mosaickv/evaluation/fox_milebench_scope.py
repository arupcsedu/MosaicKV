"""Input-only availability audit for a separately registered MileBench scope.

An unused task ID is not evidence of untouched video/document/image content.
This module deliberately cannot authorize materialization or experiment jobs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.milebench import route_milebench_dataset

REPOSITORY = "FreedomIntelligence/MileBench"
REVISION = "53c7a58051ef88bacf76541d91f03f5ba2d71e7d"
PROJECTED_COLUMNS = (
    "sample_id",
    "task_instance.images_path",
    "task_instance.combined_1_images",
    "task_instance.choice_list",
)
ORIGINAL_TASKS = ("ActionPrediction", "ActionSequence", "CLEVR-Change", "Spot-the-Diff")
PROPOSED_TASKS = {
    "ActionLocalization": "multi-choice",
    "CharacterOrder": "multi-choice",
    "ALFRED": "open-ended",
    "IEdit": "open-ended",
}
DISABLED_ACTIONS = (
    "materialization_authorized",
    "gpu_execution_authorized",
    "confirmation_authorized",
    "automatic_task_replacement_authorized",
)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Do not reinterpret an old four-task protocol or widen this audit's scope."""
    if protocol.get("protocol_id") != "fox-milebench-additional-subtasks-v1":
        raise ValueError("unexpected replacement protocol identity")
    if protocol.get("source_dataset") != {"id": REPOSITORY, "revision": REVISION}:
        raise ValueError("official dataset must be pinned to the reviewed revision")
    if protocol.get("proposed_tasks") != PROPOSED_TASKS:
        raise ValueError("proposed subtask mix differs from the input-only registration")
    if protocol.get("preserved_original_tasks") != list(ORIGINAL_TASKS):
        raise ValueError("original failed/exhausted scope must remain explicit")
    if protocol.get("projected_columns") != list(PROJECTED_COLUMNS):
        raise ValueError("only the registered input columns may be decoded")
    if protocol.get("metadata_audit_authorized") is not True:
        raise ValueError("metadata-only audit is not authorized")
    if any(protocol.get(action) is not False for action in DISABLED_ACTIONS):
        raise ValueError("downstream actions must remain explicitly disabled")
    if protocol.get("confirmation_per_task") != 64 or protocol.get("correctness_per_task") != 8:
        raise ValueError("registered input counts changed")


def safe_relative(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("unsafe or noncanonical official media path")
    return PurePosixPath(value).as_posix()


def read_projected_catalog(parquet_file: Any, *, task: str, question_type: str) -> dict:
    """Decode only IDs/media paths/choice lists; never read prompts or answers.

    HTTP range reads may transfer adjacent encoded bytes. The guarantee is column
    projection and no decoding/use/persistence of response values, not that no
    response byte ever traverses the transport.
    """
    if task not in PROPOSED_TASKS or question_type != PROPOSED_TASKS[task]:
        raise ValueError("unregistered subtask or question type")
    route = route_milebench_dataset(task, question_type)
    table = parquet_file.read(columns=list(PROJECTED_COLUMNS))
    records = table.to_pylist()
    if not records:
        raise ValueError("empty official preview")
    rows = []
    seen = set()
    for record in records:
        if set(record) != {"sample_id", "task_instance"}:
            raise ValueError("unexpected decoded columns")
        raw_id = record["sample_id"]
        if type(raw_id) is not int or raw_id < 0:
            raise ValueError("sample_id must be a nonnegative integer")
        sid = f"{task}-{raw_id:04d}"
        if sid in seen:
            raise ValueError("duplicate official sample identity")
        seen.add(sid)
        instance = record["task_instance"]
        if not isinstance(instance, dict) or set(instance) != {
            "images_path",
            "combined_1_images",
            "choice_list",
        }:
            raise ValueError("unexpected projected task_instance schema")
        images = instance["images_path"]
        montage = instance["combined_1_images"]
        choices = instance["choice_list"]
        if not isinstance(images, list) or not images:
            raise ValueError("missing source image paths")
        images = [safe_relative(p) for p in images]
        if not isinstance(montage, list) or len(montage) != 1:
            raise ValueError("exactly one official combined_1_images asset required")
        montage = safe_relative(montage[0])
        if choices is not None and (
            not isinstance(choices, list) or any(not isinstance(c, str) for c in choices)
        ):
            raise ValueError("invalid choice-list structure")
        count = len(choices or [])
        if (question_type == "multi-choice" and count < 2) or (
            question_type == "open-ended" and count != 0
        ):
            raise ValueError("input choice structure disagrees with proposed question type")
        rows.append(
            dict(
                sample_id=sid,
                source_image_paths=images,
                combined_1_images=montage,
                choice_count=count,
                source_media_verified=False,
            )
        )
    rows.sort(key=lambda row: row["sample_id"])
    return dict(
        task=task,
        metric=route.metric.value,
        official_group=route.group,
        proposed_question_type=question_type,
        question_type_evidence="input_choice_structure_only; official_annotation_meta_pending",
        source=dict(
            repository=REPOSITORY,
            revision=REVISION,
            file=f"preview/{task}_test-00000-of-00001.parquet",
            decoded_columns=list(PROJECTED_COLUMNS),
            full_file_sha256=None,
            full_file_sha256_status="not_downloaded_or_hashed; immutable_repository_revision_bound",
        ),
        rows=rows,
        projection_sha256=digest(rows),
        fresh_confirmation_certified=False,
        labels_decoded=False,
    )


def summarize_availability(catalogs: list[dict], inventory: dict, protocol: dict) -> dict:
    validate_protocol(protocol)
    if len(catalogs) != len(PROPOSED_TASKS) or {c["task"] for c in catalogs} != set(PROPOSED_TASKS):
        raise ValueError("all four registered subtasks must be audited exactly once")
    used = inventory.get("MileBench_consumed_ids")
    if not isinstance(used, list) or any(not isinstance(s, str) for s in used):
        raise ValueError("historical identity inventory is missing")
    used = set(used)
    summaries = []
    for catalog in sorted(catalogs, key=lambda c: c["task"]):
        if catalog["projection_sha256"] != digest(catalog["rows"]):
            raise ValueError("projected catalog content changed")
        unseen = [r for r in catalog["rows"] if r["sample_id"] not in used]
        summaries.append(
            dict(
                task=catalog["task"],
                metric=catalog["metric"],
                preview_ids=len(catalog["rows"]),
                known_consumed_ids=len(catalog["rows"]) - len(unseen),
                id_only_available_upper_bound=len(unseen),
                requested_correctness_and_confirmation=72,
                id_only_count_sufficient=len(unseen) >= 72,
                certified_fresh_media_groups=0,
                official_question_type_verified=False,
            )
        )
    return dict(
        status="scope_prepared_not_freshness_certified",
        protocol_sha256=digest(protocol),
        tasks=summaries,
        history_inventory_complete=inventory.get("complete_scan") is True,
        history_errors=inventory.get("errors", ["unknown"]),
        unresolved_history_roots=inventory.get("unclassified_workload_roots", []),
        required_before_confirmation=[
            "classify unresolved historical roots and hash-bound ID/media exclusions",
            "verify official annotation question_type without benchmark-outcome selection",
            "check original frames, montage pixels and video/document source groups across history",
            "split media groups into disjoint correctness and confirmation cohorts before scoring",
            "pass common-arithmetic retention-one and padding controls for every method",
            "freeze reviewed common configuration and preregister new-scope statistical family",
        ],
        numerical_result_not_overridden=True,
        materialization_authorized=False,
        gpu_execution_authorized=False,
        confirmation_authorized=False,
        original_four_task_result_preserved=True,
        changed_task_mix_is_not_original_confirmation=True,
    )


def catalog_fingerprint(catalog: dict) -> str:
    """Convenient stable artifact hash, not a claim about uncached image content."""
    return hashlib.sha256(
        json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
