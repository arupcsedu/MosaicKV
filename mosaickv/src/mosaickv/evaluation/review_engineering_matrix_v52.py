"""Pure current engineering-gate readout; never scientific claim admission.

The original V33 executions and V34 independent replay remain authoritative.
This bounded adapter verifies their already-qualified, immutable gate capsules;
it does not replay trajectories, allocator histories, or historical identities.
Earlier lossy-source and native-boundary failures remain separate negatives.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest

MAXIMUM_BYTES = 8 * 1024**2
MAXIMUM_NODES = 100000
PRODUCER = "422a67994d06ffb0b69943727063a7dfc885b6d2"
NUMERICAL_PRODUCER = "8b8db8cb7ef91ad957dbab85948f02613e7dc5fa"
NATIVE_PRODUCER = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
SOURCE_PLAN_SHA = "3bbd8e053073bf6c878e1618923c45f02ea2f5325b04a582d1ee648e2aa28745"
COCO_PLAN_SHA = "0e5351a279139c5687087be1ecc5fffd31cab359728f5557c938e16101145a9a"
SOURCE_ROOT = "/sfs/weka/scratch/djy8hg/workdir/fox-review-v33/run-20261111-1"
KNOWN_BINDINGS = {
    "native": {
        "path": "/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109/"
        "analysis/correctness/GATE_correctness.json",
        "sha256": "ca593e0522d3a9ee996fac0ce60a58b5802d35c4ff4404cdbb5799643b3439d9",
    },
    "source": {
        "path": "/scratch/djy8hg/workdir/fox-review-v33/run-20261111-1/"
        "analysis/GATE_bounded_reader_v28.json",
        "sha256": "c05c0626c56f91ffb89c0b73f793ab9faceda6fc1d7ccac68f1543aa78d46ac6",
    },
    "whole": {
        "path": "/scratch/djy8hg/workdir/fox-review-v34/run-20261111-1/"
        "whole-replay/GATE_bounded_reader_v27.json",
        "sha256": "49a2762f961a744162182fa892f5db7eb28ddefeb97d6ef255941ecfef6ee7d0",
    },
    "coco": {
        "path": "/scratch/djy8hg/workdir/fox-review-v34/run-20261111-1/"
        "coco/analysis/GATE_coco_boundary_v28.json",
        "sha256": "b3562f04514157619ce2f0162a2ff3b5f78e8e732028b5d2f8cabfa83d7afabf",
    },
}
SOURCE_SAMPLES = ("000000532947", "v1_197")
COCO_SAMPLES = (
    "coco_caption:COCO_val2014_000000260440.jpg",
    "coco_caption:COCO_val2014_000000512467.jpg",
)
STORAGE_FIELDS = tuple(
    f"{boundary}_{kind}_bytes"
    for boundary in ("initial", "final", "peak")
    for kind in ("tensor_backing", "allocator_block")
)
KV_SCOPE = "published_KV_and_all_cache_required_storage_prefill_construction_decode"
BRIDGE_FIELDS = {
    "actual_GPU_execution_base_sha",
    "new_control_source_sha",
    "all_original_producer_files_byte_identical",
    "original_V9_transport_sha256",
    "source_change_policy",
    "added_paths",
}
SOURCE_BRIDGE_SHA = "b5028428d68b924b9ab708b985040a8008357068279eb5eb2692ea05dfc3ba30"
TRANSPORT_SHA = "3722b96e4d88b76c80268af9256cb1be4a0c9526469064002b0b1f9ddab75a7b"


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate gate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("nonfinite JSON gate constant")


def _bounded(value):
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAXIMUM_NODES or depth > 64:
            raise ValueError("unchanged gate node/depth bound exceeded")
        if isinstance(item, dict):
            nodes += len(item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite gate observation")
        elif isinstance(item, str) and len(item.encode()) > MAXIMUM_BYTES:
            raise ValueError("unchanged projected metadata byte bound exceeded")


def _read(raw, expected):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAXIMUM_BYTES:
        raise ValueError("bounded raw immutable gate bytes required")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original engineering gate hash differs")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
    except (UnicodeError, RecursionError):
        raise ValueError("bounded UTF-8 gate object required") from None
    if not isinstance(value, dict):
        raise ValueError("immutable gate object required")
    _bounded(value)
    return value


def _require(record, fields, label):
    if not isinstance(record, dict) or any(record.get(name) is not True for name in fields):
        raise ValueError(f"complete passed registered {label} required")


def _parity(value):
    _require(value, ("bitwise_logit_parity", "pass_128"), "source parity")
    if (
        type(value.get("predictions")) is not int
        or value["predictions"] != 128
        or type(value.get("prediction_agreement")) is not int
        or value["prediction_agreement"] != 128
        or type(value.get("maximum_logit_difference")) not in (int, float)
        or value["maximum_logit_difference"] != 0
    ):
        raise ValueError("registered 128-prediction bitwise source fidelity failed")


def _observations(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("two complete consumed-input byte-matching observations required")
    rows = {}
    for row in value:
        sample = row.get("sample_id") if isinstance(row, dict) else None
        if (
            sample not in SOURCE_SAMPLES
            or sample in rows
            or row.get("exact_bytes_matched") is not True
        ):
            raise ValueError("unique registered paired sample and exact bytes required")
        left, right = row.get("source"), row.get("raw")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError("actual constructed source/raw storage observations required")
        if any(
            type(left.get(field)) is not int
            or left[field] <= 0
            or type(right.get(field)) is not int
            or left[field] != right[field]
            for field in STORAGE_FIELDS
        ):
            raise ValueError("six actual physical storage quantities must match exactly")
        for kind in ("tensor_backing", "allocator_block"):
            if left[f"peak_{kind}_bytes"] < max(
                left[f"initial_{kind}_bytes"], left[f"final_{kind}_bytes"]
            ):
                raise ValueError("peak storage cannot be below initial or final storage")
        rows[sample] = {field: left[field] for field in STORAGE_FIELDS}
    return rows


def _source_gate(value):
    _require(
        value,
        (
            "complete",
            "source_fidelity_128_pass",
            "raw_same_selected_128_pass",
            "whole_request_KV_byte_match_pass",
        ),
        "V33 source",
    )
    if (
        value.get("source_sha") != PRODUCER
        or value.get("plan_sha256") != SOURCE_PLAN_SHA
        or value.get("failures") != []
        or value.get("independent_quality_confirmation") is not False
        or value.get("paper_campaign_authorized") is not False
        or value.get("whole_request_GPU_peak_certified") is not False
    ):
        raise ValueError("source execution provenance or claim scope differs")
    arms = value.get("completed_arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise ValueError("four actually constructed source/raw arms required")
    for index, row in enumerate(arms):
        if not isinstance(row, dict):
            raise ValueError("explicit source worker record required")
        if (
            row.get("worker") != index
            or type(row.get("worker")) is not int
            or row.get("arm") != ("source" if index % 2 == 0 else "raw")
            or row.get("sample_id") != SOURCE_SAMPLES[index // 2]
        ):
            raise ValueError("source worker/sample/arm registration differs")
        _parity(row.get("parity"))
    return _observations(value.get("observations"))


def _whole_gate(value, source_observations):
    _require(
        value,
        (
            "complete",
            "source_fidelity_128_pass",
            "whole_request_KV_byte_match_pass",
            "whole_request_KV_peak_certified",
            "independent_replay_from_actual_GPU_artifacts",
        ),
        "independent whole-KV replay",
    )
    if (
        value.get("source_sha") != PRODUCER
        or value.get("actual_execution_source_sha") != PRODUCER
        or value.get("validation_source_sha") != NUMERICAL_PRODUCER
        or value.get("gate_origin") != "derived_validation_not_new_GPU_execution"
        or value.get("actual_execution_root") != SOURCE_ROOT
        or value.get("actual_execution_root_sha256") != digest(SOURCE_ROOT)
        or value.get("actual_plan_path") != SOURCE_ROOT + "/plan.json"
        or value.get("actual_plan_sha256") != SOURCE_PLAN_SHA
        or value.get("actual_gate_path") != SOURCE_ROOT + "/analysis/GATE_bounded_reader_v28.json"
        or value.get("actual_gate_sha256") != KNOWN_BINDINGS["source"]["sha256"]
        or value.get("input_gate_sha256") != KNOWN_BINDINGS["source"]["sha256"]
        or value.get("input_plan_sha256") != SOURCE_PLAN_SHA
        or value.get("failures") != []
        or value.get("whole_request_GPU_peak_certified") is not False
        or value.get("independent_quality_confirmation") is not False
        or value.get("fresh_confirmation_authorized") is not False
        or value.get("paper_campaign_authorized") is not False
    ):
        raise ValueError("whole-KV actual execution/replay provenance or scope differs")
    if _observations(value.get("observations")) != source_observations:
        raise ValueError("independent replay storage differs from original observations")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("four completed independently replayed ownership records required")
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("explicit replayed ownership record required")
        sample, arm = SOURCE_SAMPLES[index // 2], "source" if index % 2 == 0 else "raw"
        if (
            row.get("sample_id") != sample
            or row.get("arm") != arm
            or type(row.get("worker")) is not int
            or row.get("worker") != index
        ):
            raise ValueError("ownership record sample/arm registration differs")
        _parity(row.get("parity"))
        boundaries, crosslink = row.get("boundaries"), row.get("native_role_crosslink")
        _require(boundaries, ("whole_request_KV_peak_certified",), "published-KV storage")
        _require(
            crosslink,
            ("published_native_owner_ranges_bound", "retired_native_owner_ranges_absent"),
            "actual native-owner retirement",
        )
        if (
            type(crosslink.get("verified_layers")) is not int
            or crosslink["verified_layers"] != 32
            or boundaries.get("scope") != KV_SCOPE
            or boundaries.get("decode_updates") != 127
            or boundaries.get("whole_request_GPU_peak_certified") is not False
            or any(
                boundaries.get(field) != source_observations[sample][field]
                for field in STORAGE_FIELDS
            )
            or not isinstance(row.get("ownership_artifact"), str)
            or row["ownership_artifact"]
            != SOURCE_ROOT + f"/workers/{index}/traces/whole_allocator_ownership.json"
            or type(row.get("observation_count")) is not int
            or row["observation_count"] <= 0
        ):
            raise ValueError("complete time-aligned ownership/storage crosslinks required")
        contract._hash(row.get("ownership_sha256"), "already-qualified ownership sidecar")


def _coco_gate(value):
    _require(
        value,
        ("complete", "corrected_boundary_128_pass", "historical_native_boundary_failure_preserved"),
        "corrected COCO boundary",
    )
    if (
        value.get("source_sha") != NUMERICAL_PRODUCER
        or value.get("plan_sha256") != COCO_PLAN_SHA
        or value.get("source_scope") != "FP16_GQA_COCO_FullKV_evaluation_only_boundary_control"
        or value.get("failures") != []
        or any(
            value.get(field) is not False
            for field in (
                "fresh_confirmation_authorized",
                "paper_campaign_authorized",
                "production_defaults_changed",
                "quality_or_latency_claim_allowed",
                "source_reconstruction_validated",
            )
        )
    ):
        raise ValueError("COCO numerical controls cannot broaden model/source/quality scope")
    reports = value.get("reports")
    if not isinstance(reports, list) or len(reports) != 2:
        raise ValueError("both registered consumed COCO numerical cases required")
    for index, row in enumerate(reports):
        _require(
            row,
            (
                "pass_128",
                "bitwise_logits",
                "initial_cache_bitwise_equal",
                "final_cache_bitwise_equal",
            ),
            "COCO bitwise numerical",
        )
        errors = row.get("per_prediction_maximum_logit_difference")
        if (
            type(row.get("index")) is not int
            or row.get("index") != index
            or row.get("sample_id") != COCO_SAMPLES[index]
            or row.get("prediction_agreement") != 128
            or row.get("predictions") != 128
            or row.get("ordinary_continuation_steps") != 127
            or row.get("unchanged_absolute_tolerance") != 1e-4
            or row.get("maximum_logit_difference") != 0
            or not isinstance(errors, list)
            or len(errors) != 128
            or any(type(error) not in (int, float) or error != 0 for error in errors)
        ):
            raise ValueError("128 actual bitwise COCO predictions and unchanged tolerance required")


def validate_engineering_gates(
    gates: dict[str, bytes], *, source_bridge: dict[str, Any]
) -> dict[str, Any]:
    """Verify saved compact certificates, not their raw observations again.

    The caller independently checks the original V33-to-V34 Git difference and
    supplies that exact registered proof from the original V34 plan. A current
    auditor SHA is recorded separately by the I/O caller, never substituted for
    either original execution or original validation source.
    """
    if not isinstance(gates, dict) or set(gates) != set(KNOWN_BINDINGS):
        raise ValueError("all four registered immutable engineering gates required")
    if sum(len(raw) for raw in gates.values() if isinstance(raw, bytes)) > MAXIMUM_BYTES:
        raise ValueError("unchanged combined projected gate-byte bound exceeded")
    records = {name: _read(raw, KNOWN_BINDINGS[name]["sha256"]) for name, raw in gates.items()}
    _bounded(records)
    if not isinstance(source_bridge, dict) or set(source_bridge) != BRIDGE_FIELDS:
        raise ValueError("explicit bounded CPU source-compatibility proof required")
    _bounded(source_bridge)
    if (
        source_bridge["actual_GPU_execution_base_sha"] != PRODUCER
        or source_bridge["new_control_source_sha"] != NUMERICAL_PRODUCER
        or source_bridge["all_original_producer_files_byte_identical"] is not True
        or source_bridge["original_V9_transport_sha256"] != TRANSPORT_SHA
        or source_bridge["source_change_policy"]
        != "additions_only_all_original_files_byte_identical"
        or not isinstance(source_bridge["added_paths"], list)
        or digest(source_bridge) != SOURCE_BRIDGE_SHA
    ):
        raise ValueError("CPU proof must preserve the original V33-to-V34 registered source bridge")
    added = source_bridge["added_paths"]
    if (
        any(not isinstance(path, str) for path in added)
        or len(added) != len(set(added))
        or any(
            not isinstance(path, str)
            or not path.startswith("mosaickv/")
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            for path in added
        )
    ):
        raise ValueError("original CPU-only bridge requires canonical unique added paths")
    native = records["native"]
    if (
        native.get("status") != "passed"
        or native.get("stage") != "correctness"
        or native.get("source_sha") != NATIVE_PRODUCER
        or native.get("blockers") != []
        or native.get("tests_actually_executed")
        != dict(
            actual=128,
            completed=128,
            expected=128,
            pairing_count=64,
            workers_completed=4,
            workers_expected=4,
        )
    ):
        raise ValueError("native COCO gate is 128 checks on 32 inputs, not 128-token evidence")
    observations = _source_gate(records["source"])
    _whole_gate(records["whole"], observations)
    _coco_gate(records["coco"])
    return dict(
        schema="fox_current_engineering_matrix_v52",
        classification="bounded_saved_gate_readout_not_new_experiment_or_paper_admission",
        evidence={name: dict(binding) for name, binding in KNOWN_BINDINGS.items()},
        execution_source_sha=PRODUCER,
        numerical_source_sha=NUMERICAL_PRODUCER,
        registered_validation_source_sha=NUMERICAL_PRODUCER,
        source_compatibility_proof_sha256=SOURCE_BRIDGE_SHA,
        source_fidelity=dict(
            status="passed_in_registered_lossless_BF16_MHA_scope",
            consumed_inputs=2,
            predictions_per_input=128,
            constructed_arms=4,
            maximum_logit_difference=0.0,
        ),
        physical_byte_matching=dict(
            status="passed_constructed_source_versus_same_selected_raw",
            boundaries=list(STORAGE_FIELDS),
            observations=observations,
        ),
        whole_request_peak=dict(
            status="passed_in_registered_published_KV_plus_required_storage_scope",
            scope=KV_SCOPE,
            whole_device_GPU_peak_certified=False,
            memory_advantage_established=False,
        ),
        coco_numerical_controls=dict(
            status="passed_evaluation_only_shape_matched_boundary",
            consumed_inputs=2,
            predictions_per_input=128,
            native_subset_checks=128,
            native_subset_inputs=32,
            unchanged_absolute_tolerance=1e-4,
            source_reconstruction_on_FP16_GQA_validated=False,
        ),
        earlier_lossy_source_failures_preserved=True,
        earlier_native_boundary_failures_preserved=True,
        raw_trajectory_or_ownership_proofs_replayed=False,
        superiority_established=False,
        independent_quality_confirmation=False,
        six_component_benefit_established=False,
        latency_superiority_established=False,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
