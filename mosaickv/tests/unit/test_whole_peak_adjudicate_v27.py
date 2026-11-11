"""CPU-only evidence-producer isolation, not new GPU measurements."""

import importlib.util
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
            native_retirement_receipts=[
                dict(allocations=[dict(owner_ranges=[[8192, 512]])]),
                dict(allocations=[dict(owner_ranges=[[12288, 512]])]),
            ]
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
