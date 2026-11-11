"""Complete early archives are CPU-recoverable, never another GPU retry."""

import gzip
import json
from copy import deepcopy

import evicastkv_source_ownership_v12 as producer
import evicastkv_source_trace_adjudication as study
import pytest
from test_source_trace_adjudication import fixture


def setup(tmp_path, monkeypatch):
    import evicastkv_review_followups as controls

    trace, conservative, declared, _, _ = fixture(tmp_path)
    parent = tmp_path / "parent"
    (parent / "worker").mkdir(parents=True)
    cpu = tmp_path / "cpu"
    (cpu / "full-cpu").mkdir(parents=True)
    (cpu / "full-cpu/cpu-tests.json").write_text("{}")
    policy = dict(
        observer_variant=study.TRACE_FIRST_VARIANT,
        arm="source3_attention_optional",
        max_allocator_entries=1000000,
    )
    protocol = dict(source_sha="a" * 40, source_root=str(tmp_path), bindings={}, policy=policy)
    (parent / "protocol.json").write_text(json.dumps(protocol))
    (parent / "worker/manifest.json").write_text(
        json.dumps(
            dict(source_sha="a" * 40, source_dirty=False, policy_sha256=study.digest(policy))
        )
    )
    (parent / "worker/raw.json").write_text(
        json.dumps(
            dict(
                arm=policy["arm"],
                counters=dict(updates=127),
                record=dict(output_tokens=128, storage=dict(history_updates_per_layer=[127] * 32)),
            )
        )
    )
    (parent / "worker/logits.pt").write_bytes(b"immutable complete model tensor artifact")
    capture = dict(
        records=trace["allocator_records"],
        events=len(trace["actual_events"]),
        stack_table=trace["stack_table"],
        clock_calibration=trace["clock_calibration"],
        complete_request=True,
        allocator_history_complete=True,
        blocks_exact_certification=conservative["blocks_exact_certification"],
    )
    producer.archive_before_attribution(
        parent,
        capture,
        declared["observations"],
        trace["actual_events"],
        dict(segments=[], device_traces=[trace["actual_events"]]),
    )
    monkeypatch.setattr(study, "clean_source", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(study, "SCRATCH", tmp_path)
    monkeypatch.setattr(
        controls, "cpu_gate", lambda root, revision: dict(status="passed", source_sha=revision)
    )
    return parent, cpu, tmp_path / "derived"


def test_valid_complete_archive_recovers_only_CPU_inventory_and_preserves_raw(
    tmp_path, monkeypatch
):
    parent, cpu, out = setup(tmp_path, monkeypatch)
    before = {p.name: study.sha(p) for p in (parent / "worker").iterdir()}
    gate = study.recover_trace_first(out, parent=parent, cpu_gate_root=cpu)
    assert gate["status"] == "CPU_recovered_trace_inventory"
    assert gate["new_GPU_inference"] == 0 and gate["ownership_capture_complete"]
    assert not gate["GPU_worker_completed"] and not gate["numerical_guard_promoted"]
    assert not gate["exact_whole_request_KV_peak_certified"]
    assert before == {p.name: study.sha(p) for p in (parent / "worker").iterdir()}
    assert not (parent / "worker/result.json").exists()
    values, _, _ = study._inputs(out)
    assert values["CPU_recovered"] and not values["worker_complete"]
    assert values["conservative"]["unknown_allocation_generations"] > 0
    assert study.inventory(out)["rules_generated"] == 0
    with pytest.raises(ValueError, match="new immutable"):
        study.recover_trace_first(out, parent=parent, cpu_gate_root=cpu)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_receipt",
        "truncated",
        "changed_archive",
        "receipt_hash",
        "capture_hash",
        "coverage",
        "source",
        "structural",
    ],
)
def test_missing_truncated_tampered_incomplete_cannot_recover(tmp_path, monkeypatch, defect):
    parent, cpu, out = setup(tmp_path, monkeypatch)
    receipt = parent / "worker/trace_pre_attribution_receipt.json"
    archive = parent / "worker/allocator_trace_pre_attribution.json.gz"
    value = json.loads(receipt.read_text())
    if defect == "missing_receipt":
        receipt.unlink()
    elif defect in {"truncated", "changed_archive"}:
        archive.write_bytes(
            archive.read_bytes()[:-4] if defect == "truncated" else gzip.compress(b"{}")
        )
        if defect == "truncated":
            value["trace_sha256"] = study.sha(archive)
            receipt.write_text(json.dumps(value))
    elif defect in {"receipt_hash", "capture_hash"}:
        value["trace_sha256" if defect == "receipt_hash" else "capture_sha256"] = "0" * 64
        receipt.write_text(json.dumps(value))
    elif defect == "coverage":
        value["complete_request"] = False
        receipt.write_text(json.dumps(value))
    elif defect == "source":
        manifest = parent / "worker/manifest.json"
        item = json.loads(manifest.read_text())
        item["source_sha"] = "b" * 40
        manifest.write_text(json.dumps(item))
    else:
        trace = study.load(archive)
        trace["allocator_records"][1]["bytes"] += 1
        trace["capture_metadata"]["records"] = deepcopy(trace["allocator_records"])
        archive.unlink()
        count = producer._gzip_once(archive, trace)
        value.update(
            trace_sha256=study.sha(archive),
            uncompressed_bytes=count,
            allocation_records_sha256=study.digest(trace["allocator_records"]),
            capture_sha256=study.digest(trace["capture_metadata"]),
        )
        receipt.write_text(json.dumps(value))
    with pytest.raises((ValueError, EOFError, OSError)):
        study.recover_trace_first(out, parent=parent, cpu_gate_root=cpu)
    assert not out.exists()


def test_changed_CPU_derived_ledger_is_not_accepted_for_inventory(tmp_path, monkeypatch):
    parent, cpu, out = setup(tmp_path, monkeypatch)
    study.recover_trace_first(out, parent=parent, cpu_gate_root=cpu)
    with (out / "allocator_ledger_CPU_derived.json.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="bundle changed"):
        study.inventory(out)


def test_CPU_failure_blocks_all_recovery_without_GPU_action(tmp_path, monkeypatch):
    import evicastkv_review_followups as controls

    parent, cpu, out = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        controls, "cpu_gate", lambda *a: (_ for _ in ()).throw(ValueError("CPU failed"))
    )
    with pytest.raises(ValueError, match="CPU failed"):
        study.recover_trace_first(out, parent=parent, cpu_gate_root=cpu)
    assert not out.exists()
