from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _load_analyzer() -> ModuleType:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_choice_c_v73_clean_quality_memory.py"
    spec = importlib.util.spec_from_file_location("mosaickv_v83_method_labels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_analyzer()


def _row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "status": "completed",
        "error": None,
    }


def _write_telemetry(root: Path, candidate_run_id: str) -> None:
    sample_ids = ["docvqa:1", "docvqa:2"]
    trials = []
    for order, (protocol_method, effective_method) in enumerate(
        (
            ("full_kv", "full_kv"),
            ("prompt_attention_topk", "prompt_attention_topk"),
            ("mosaickv_full", "mosaickv_exact"),
        )
    ):
        run_id = (
            candidate_run_id
            if protocol_method == "mosaickv_full"
            else f"v83-docvqa-trial-00-{protocol_method}"
        )
        trials.append(
            {
                "repetition": 0,
                "method": effective_method,
                "run_id": run_id,
                "order": order,
                "rows": [_row(sample_id) for sample_id in sample_ids],
            }
        )
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "sample_ids": sample_ids,
        "trials": trials,
    }
    destination = root / "telemetry" / "v83-docvqa-test.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(telemetry), encoding="utf-8")


def test_load_task_records_protocol_and_effective_method(tmp_path: Path) -> None:
    _write_telemetry(tmp_path, "v83-docvqa-trial-00-mosaickv_full")

    _telemetry, rows = ANALYZER._load_task(
        tmp_path,
        "docvqa",
        expected_samples=2,
        expected_repetitions=1,
        run_prefix="v83",
        candidate_effective_method="mosaickv_exact",
    )

    candidate_rows = [row for row in rows if row["method"] == "mosaickv_full"]
    assert len(candidate_rows) == 2
    assert {row["effective_method"] for row in candidate_rows} == {"mosaickv_exact"}


def test_load_task_rejects_mislabeled_candidate_run(tmp_path: Path) -> None:
    _write_telemetry(tmp_path, "v83-docvqa-trial-00-mosaickv_exact")

    with pytest.raises(RuntimeError, match="unexpected protocol/effective method pair"):
        ANALYZER._load_task(
            tmp_path,
            "docvqa",
            expected_samples=2,
            expected_repetitions=1,
            run_prefix="v83",
            candidate_effective_method="mosaickv_exact",
        )


def test_v83_negative_seal_binds_protocol_and_analyzer() -> None:
    package = Path(__file__).resolve().parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v83_negative_heldout_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol = package / seal["protocol"]["archived_path"]
    analyzer = package / seal["artifacts"]["analyzer"]["path"]

    assert seal["overall_quality_memory_gate_pass"] is False
    assert seal["memory_gate_pass"] is True
    assert seal["quality_gate_pass"] is False
    assert seal["frozen_candidate"]["protocol_method"] == "mosaickv_full"
    assert seal["frozen_candidate"]["effective_method"] == "mosaickv_exact"
    assert seal["frozen_candidate"]["supports_full_mosaickv_claim"] is False
    assert seal["future_protocol"]["v83_candidate_must_not_be_revised_from_v83_outputs"]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == seal["protocol"]["sha256"]
    assert (
        hashlib.sha256(analyzer.read_bytes()).hexdigest() == seal["artifacts"]["analyzer"]["sha256"]
    )
