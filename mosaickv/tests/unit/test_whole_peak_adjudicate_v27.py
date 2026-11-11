"""CPU-only evidence-producer isolation, not new GPU measurements."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_whole_peak_adjudicate_v27.py"
    spec = importlib.util.spec_from_file_location("whole_peak_adjudicate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(protocol):
    return dict(source_sha=protocol.PRODUCER), dict(
        source_sha=protocol.PRODUCER,
        plan_sha256=protocol.PARENT_PLAN_SHA,
        failures=[],
        complete=True,
        source_fidelity_128_pass=True,
        whole_request_KV_byte_match_pass=True,
    )


def test_only_registered_actual_execution_is_admissible(protocol):
    plan, gate = record(protocol)
    protocol.validate_parent(plan, gate)
    plan["source_sha"] = "f" * 40
    with pytest.raises(ValueError, match="original producer"):
        protocol.validate_parent(plan, gate)


@pytest.mark.parametrize(
    "field", ["complete", "source_fidelity_128_pass", "whole_request_KV_byte_match_pass"]
)
@pytest.mark.parametrize("bad", [False, 1, "true", None])
def test_failed_or_truthy_gates_never_authorize_reuse(protocol, field, bad):
    plan, gate = record(protocol)
    gate[field] = bad
    with pytest.raises(ValueError):
        protocol.validate_parent(plan, gate)


@pytest.mark.parametrize(
    "field,bad", [("source_sha", "f" * 40), ("plan_sha256", "0" * 64), ("failures", ["incomplete"])]
)
def test_provenance_or_failure_drift_is_not_reclassified(protocol, field, bad):
    plan, gate = record(protocol)
    gate[field] = bad
    with pytest.raises(ValueError):
        protocol.validate_parent(plan, gate)


def owner_records():
    record = dict(
        evidence=dict(
            native_CUDA_free_completed_verified=True,
            native_retirement_receipts=[
                dict(
                    label="layer_0",
                    actual_free_completed_verified=True,
                    allocations=[dict(owner_ranges=[[8192, 512]])],
                ),
                dict(
                    label="layer_1",
                    actual_free_completed_verified=True,
                    allocations=[dict(owner_ranges=[[12288, 512]])],
                ),
            ],
        )
    )
    deltas = [
        dict(
            phase="prefill_layer_0_KV_published",
            added_owners=[[4096, 128], [8192, 512]],
            removed_owners=[],
        ),
        dict(phase="prefill_layer_0_native_retired", added_owners=[], removed_owners=[[8192, 512]]),
        dict(phase="prefill_layer_1_KV_published", added_owners=[[12288, 512]], removed_owners=[]),
        dict(
            phase="prefill_layer_1_native_retired", added_owners=[], removed_owners=[[12288, 512]]
        ),
    ]
    return record, deltas


def test_native_receipts_bind_published_then_retired_storage(protocol):
    record, deltas = owner_records()
    assert protocol.crosslink_native_owners(record, deltas, layers=2)["verified_layers"] == 2


@pytest.mark.parametrize(
    "change", ["unpublished", "still_live", "missing_retirement", "missing_publication"]
)
def test_unrelated_valid_frees_do_not_certify_request_peak(protocol, change):
    record, deltas = owner_records()
    if change == "unpublished":
        deltas[0]["added_owners"] = [[4096, 128]]
    elif change == "still_live":
        deltas[1]["removed_owners"] = []
    elif change == "missing_retirement":
        deltas.pop()
    else:
        deltas.pop(0)
    with pytest.raises(ValueError):
        protocol.crosslink_native_owners(record, deltas, layers=2)


def output_fixture(protocol, tmp_path):
    out = tmp_path / "worker"
    out.mkdir()
    outputs = {}
    for name in protocol.REQUIRED_OUTPUTS:
        path = out / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("synthetic-fixture-" + name).encode())
        outputs[name] = protocol.sha(path)
    result = dict(
        source_sha=protocol.PRODUCER,
        plan_sha256=protocol.PARENT_PLAN_SHA,
        failures=[],
        actual_rows=1,
        outputs=outputs,
    )
    return out, result


def test_every_critical_read_is_bound_before_analysis(protocol, tmp_path):
    out, result = output_fixture(protocol, tmp_path)
    resolved = protocol.validate_worker_outputs(out, result)
    assert set(resolved) == protocol.REQUIRED_OUTPUTS
    assert all(path.is_relative_to(out.resolve()) for path in resolved.values())


@pytest.mark.parametrize(
    "missing",
    [
        "raw.jsonl",
        "logits.pt",
        "manifest.json",
        "prefill-lifecycle.json",
        "traces/result.json",
        "traces/whole_allocator_ownership.json",
    ],
)
def test_omitting_any_critical_hash_binding_fails(protocol, tmp_path, missing):
    out, result = output_fixture(protocol, tmp_path)
    del result["outputs"][missing]
    with pytest.raises(ValueError, match="six critical"):
        protocol.validate_worker_outputs(out, result)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.json",
        "/absolute.json",
        "traces/../escape.json",
        "./extra.json",
        "traces//extra.json",
        "traces\\extra.json",
    ],
)
def test_relative_escape_or_noncanonical_output_names_fail(protocol, tmp_path, name):
    out, result = output_fixture(protocol, tmp_path)
    result["outputs"][name] = "a" * 64
    with pytest.raises(ValueError, match="canonical relative"):
        protocol.validate_worker_outputs(out, result)


def test_symlink_escape_is_rejected_even_when_its_hash_matches(protocol, tmp_path):
    out, result = output_fixture(protocol, tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("synthetic-outside")
    path = out / "extra.json"
    path.symlink_to(outside)
    result["outputs"]["extra.json"] = protocol.sha(outside)
    with pytest.raises(ValueError, match="escapes"):
        protocol.validate_worker_outputs(out, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "f" * 40),
        ("plan_sha256", "0" * 64),
        ("actual_rows", True),
        ("failures", ["failed"]),
    ],
)
def test_worker_source_plan_rows_and_failure_contracts_fail_closed(
    protocol, tmp_path, field, value
):
    out, result = output_fixture(protocol, tmp_path)
    result[field] = value
    with pytest.raises(ValueError, match="source/plan-matched"):
        protocol.validate_worker_outputs(out, result)


def test_changed_or_malformed_output_hash_fails(protocol, tmp_path):
    out, result = output_fixture(protocol, tmp_path)
    result["outputs"]["raw.jsonl"] = "f" * 64
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.validate_worker_outputs(out, result)
    result["outputs"]["raw.jsonl"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        protocol.validate_worker_outputs(out, result)


def test_distinct_critical_files_cannot_be_symlink_aliases(protocol, tmp_path):
    out, result = output_fixture(protocol, tmp_path)
    # An extra harmless alias is allowed; two required records cannot alias.
    (out / "manifest.json").unlink()
    (out / "manifest.json").symlink_to(out / "raw.jsonl")
    result["outputs"]["manifest.json"] = result["outputs"]["raw.jsonl"]
    with pytest.raises(ValueError, match="distinct critical"):
        protocol.validate_worker_outputs(out, result)


def sidecar_fixture(protocol, tmp_path):
    out, result = output_fixture(protocol, tmp_path)
    outputs = protocol.validate_worker_outputs(out, result)
    proof, _deltas = owner_records()
    proof["allocator"] = dict(
        constructor_snapshot_reuse=True,
        snapshot_cache=dict(every_observation_has_proof=True),
        ownership_delta_artifact="traces/whole_allocator_ownership.json",
        ownership_delta_sha256=result["outputs"]["traces/whole_allocator_ownership.json"],
    )
    return dict(whole_KV=proof), outputs, result


def test_required_ownership_sidecar_is_bound_to_both_row_and_manifest(protocol, tmp_path):
    row, outputs, result = sidecar_fixture(protocol, tmp_path)
    assert (
        protocol.validate_ownership_sidecar(row, outputs, result)
        == outputs["traces/whole_allocator_ownership.json"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_receipts",
        "false_free",
        "missing_allocator",
        "missing_snapshot_proofs",
        "wrong_sidecar",
        "wrong_sidecar_hash",
    ],
)
def test_missing_or_drifted_critical_owner_proofs_fail(protocol, tmp_path, change):
    row, outputs, result = sidecar_fixture(protocol, tmp_path)
    if change == "missing_receipts":
        del row["whole_KV"]["evidence"]["native_retirement_receipts"]
    elif change == "false_free":
        row["whole_KV"]["evidence"]["native_CUDA_free_completed_verified"] = 1
    elif change == "missing_allocator":
        del row["whole_KV"]["allocator"]
    elif change == "missing_snapshot_proofs":
        row["whole_KV"]["allocator"]["snapshot_cache"] = {}
    elif change == "wrong_sidecar":
        row["whole_KV"]["allocator"]["ownership_delta_artifact"] = "../other.json"
    else:
        row["whole_KV"]["allocator"]["ownership_delta_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        protocol.validate_ownership_sidecar(row, outputs, result)


@pytest.mark.parametrize(
    "change",
    [
        "missing_flag",
        "empty_allocations",
        "empty_ranges",
        "bool_pointer",
        "wrong_layer_label",
        "false_completed",
    ],
)
def test_native_owner_crosslinks_require_complete_ordered_strong_proof(protocol, change):
    record, deltas = owner_records()
    record = deepcopy(record)
    evidence = record["evidence"]
    receipt = evidence["native_retirement_receipts"][0]
    if change == "missing_flag":
        del evidence["native_CUDA_free_completed_verified"]
    elif change == "empty_allocations":
        receipt["allocations"] = []
    elif change == "empty_ranges":
        receipt["allocations"][0]["owner_ranges"] = []
    elif change == "bool_pointer":
        receipt["allocations"][0]["owner_ranges"][0][0] = True
    elif change == "wrong_layer_label":
        receipt["label"] = "layer_9"
    else:
        receipt["actual_free_completed_verified"] = 1
    with pytest.raises(ValueError):
        protocol.crosslink_native_owners(record, deltas, layers=2)
