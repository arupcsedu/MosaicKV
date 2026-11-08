import json

import pytest
import review_shared_proto_ownership as writer

from mosaickv.measurements.kv_timeline import AllocationTimeline


def test_write_once_review_keeps_mixed_peak_result_and_denominators(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    root.mkdir()
    receipts = root / "receipts"
    receipts.mkdir()
    plan = dict(source_sha="a" * 40, bound_files={}, reuse_canaries=str(receipts))
    (root / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(writer, "clean_source", lambda *_: "b" * 40)
    monkeypatch.setattr(writer, "source_bindings", lambda *_: {})
    for index in (0, 1):
        folder = root / f"workers/{index}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(dict(status="completed")))
        (folder / "manifest.json").write_text(
            json.dumps(
                dict(source_dirty=False, git_sha=plan["source_sha"], environment=dict(packages={}))
            )
        )
        records = []
        for arm in sorted(writer.ARMS):
            candidate = arm == "shared_source_prototypes"
            ledger = AllocationTimeline()
            ledger.ingest([dict(action="alloc", addr=128, size=110 if candidate else 100)])
            ledger.classify(128, "kv_workspace", "synthetic fixture only")
            proof = ledger.report(
                complete=True, history_complete=True, opaque_workspace_classified=True
            )
            proof["all_head_source_retirement_receipts"] = [dict(retired_before_decode=True)] * (
                32 if candidate else 0
            )
            records.append(
                dict(
                    arm=arm,
                    sample_id=f"fixture-{index}",
                    status="completed",
                    ownership_timeline=proof,
                    detail=dict(
                        initial_encoded_bytes=90 if candidate else 100,
                        final_encoded_bytes=95 if candidate else 105,
                    ),
                    mechanisms=dict(CPU_residual_bytes=20 if candidate else 0),
                    fidelity=dict(passed=True),
                    numerically_admissible=True,
                    request_wall_seconds=0.01,
                )
            )
            (folder / f"{arm}-audit.logits.pt").write_bytes(
                b"synthetic fixture, not model predictions"
            )
        (folder / "raw.jsonl").write_text("\n".join(json.dumps(row) for row in records))
        (receipts / f"{index}.json").write_text(
            json.dumps(
                dict(
                    producer_source_sha="c" * 40,
                    native=dict(passed=True),
                    rows=[dict(row, phase="canary") for row in records],
                )
            )
        )
    output = tmp_path / "derived"
    writer.summarize(root, output, writer.sha(root / "plan.json"))
    result = json.loads((output / "decision.json").read_text())
    assert result["complete"] and result["numerical_gate_passed"]
    assert result["whole_request_certificates_passed"]
    assert result["actual_all_head_source_retirement_passed"]
    assert not result["all_whole_request_peaks_smaller"]
    assert not result["larger_campaign_authorized"]
    assert len(result["audits"]) == len(result["reused_canaries"]) == 4
    assert result["comparisons"][0]["final_GPU_plus_CPU_residual_reduction"] < 0
    with pytest.raises(FileExistsError):
        writer.summarize(root, output, writer.sha(root / "plan.json"))
