import pytest
import torch
from test_text_source_storage import setup

from mosaickv.evaluation import text_source_storage
from mosaickv.evaluation.construction_phase_timing import PhaseTimer, instrument, phase_totals


@pytest.mark.parametrize("merged", [False, True])
def test_profiled_and_unprofiled_packets_and_proof_identical(merged):
    packet, positions, text, record, attention = setup(merged)
    original_phase = text_source_storage._construction_phase
    reference, proof = text_source_storage.optimize(packet, record, positions, text, attention)
    timer = PhaseTimer()
    with instrument(timer):
        candidate, candidate_proof = text_source_storage.optimize(
            packet, record, positions, text, attention
        )
    assert text_source_storage._construction_phase is original_phase
    assert proof == candidate_proof
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(reference.tensors(), candidate.tensors(), strict=True)
    )
    totals = phase_totals(timer.resolve())
    for key in (
        "metadata_validation",
        "reference_materialize",
        "strip_reference_overlays",
        "source_encoding",
        "projection_metadata",
        "candidate_materialize",
        "sparse_patch_build",
        "text_only_patch_validation",
        "accounting_and_choice",
    ):
        assert totals["text_source." + key]["calls"] == 1
    for key in ("verification_materialize", "verification_equal"):
        assert totals["text_source." + key]["calls"] == 3


def test_invalid_metadata_still_fails_and_scope_restores():
    packet, positions, text, record, attention = setup()
    original_phase = text_source_storage._construction_phase
    timer = PhaseTimer()
    with pytest.raises(ValueError), instrument(timer):
        text_source_storage.optimize(packet, record, positions, tuple(reversed(text)), attention)
    assert text_source_storage._construction_phase is original_phase
    assert any(row["phase"] == "text_source.metadata_validation" for row in timer.resolve())
