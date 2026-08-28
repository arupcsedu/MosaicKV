from __future__ import annotations

from pathlib import Path

import yaml


def test_v109_negative_seal_consumes_ids_and_preserves_complete_traces() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (
            package / "configs/ledgers/choice_c_v109_negative_preflight_seal.yaml"
        ).read_text(encoding="utf-8")
    )

    assert seal["classification"] == "sealed_negative_four_sample_gpu_preflight"
    assert seal["overall_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["gate_summary"]["candidate_samples_completed"] == 0
    assert seal["gate_summary"]["rejected_proposals_meeting_memory_gate"] == 2
    assert seal["gate_summary"]["rejected_proposals_meeting_every_fidelity_gate"] == 0
    assert seal["gate_summary"]["complete_nine_field_failure_traces"] == 2
    assert seal["gate_summary"]["complete_dual_six_field_accounting_traces"] == 2
    assert seal["scientific_boundary"]["all_v109_planned_identities_consumed"] is True
    assert seal["root_cause"]["classification"] == (
        "reserved_probes_stranded_after_all_screening_partitions_failed"
    )
