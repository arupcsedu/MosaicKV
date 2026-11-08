import json

import shared_proto_ledger_study as study


def test_reused_canaries_are_loaded_before_completeness_and_memory_checks(tmp_path, monkeypatch):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for index in (0, 1):
        folder = tmp_path / f"workers/{index}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({"status": "completed"}))
        rows = [
            dict(
                arm=arm,
                status="completed",
                numerically_admissible=True,
                detail={
                    key: size
                    for key in (
                        "initial_encoded_bytes",
                        "final_encoded_bytes",
                        "reported_counter_peak_bytes",
                    )
                },
            )
            for arm, size in (("storage_compact", 100), ("shared_source_prototypes", 90))
        ]
        (folder / "raw.jsonl").write_text(
            "\n".join(json.dumps(dict(row, phase="ownership_audit")) for row in rows)
        )
        (receipts / f"{index}.json").write_text(
            json.dumps(
                dict(producer_source_sha="a" * 40, rows=[dict(row, phase="canary") for row in rows])
            )
        )
    monkeypatch.setattr(
        study, "check", lambda *_: dict(audit_only=True, reuse_canaries=str(receipts))
    )
    study.analyze(tmp_path, "fixture")
    result = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert result["complete"] and result["numerical_gate_passed"]
    assert len(result["rows"]) == 8 and len(result["memory_comparison"]) == 2
    assert result["all_reported_cache_boundaries_smaller"]
    assert not result["whole_request_certificate_passed"]
