"""Synthetic algebra and read-only layer comparison, not quality measurements."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.backends.hf_prefix_outputs import PrefixOutputRecorder
from mosaickv.evaluation.early_prefix import (
    compare_layer_outputs,
    omission_decomposition,
    position_metadata,
    retained_positions,
)


def test_omitted_output_not_attention_mass_includes_renormalization_and_history():
    p = torch.tensor([[[[0.4, 0.3, 0.2, 0.1]]]])
    v = torch.tensor([[[[1.0, 0.0], [9.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]])
    row = omission_decomposition(p, v, [[0, 2]], 3)
    kept = (0.4 * 1 + 0.2 * 2 + 0.1 * 3) / 0.7
    full = 0.4 * 1 + 0.3 * 9 + 0.2 * 2 + 0.1 * 3
    torch.testing.assert_close(row["retained_head_output"], torch.tensor([[kept, 0.0]]))
    torch.testing.assert_close(row["output_error_norm"], torch.tensor([full - kept]))
    assert row["omitted_signed_output_contribution"][0, 1] > 0
    assert row["retained_mass"][0] == pytest.approx(0.7)
    assert row["identity_max_abs_error"] < 1e-6


def test_gqa_preserves_individual_queries_and_exposes_bf16_softmax_rounding():
    p = torch.tensor([[[[0.4, 0.3, 0.3]], [[0.2, 0.5, 0.3]]]], dtype=torch.bfloat16)
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=torch.bfloat16)
    before = v.clone()
    row = omission_decomposition(p, v, [[0]], 2)
    assert row["full_head_output"].shape == (2, 2)
    assert not torch.equal(row["retained_mass"][0], row["retained_mass"][1])
    assert row["softmax_rounding_output"].abs().max() > 0
    assert torch.equal(v, before)
    assert row["identity_max_abs_error"] < 1e-5


@pytest.mark.parametrize("selected", [[], [0, 0], [9]])
def test_invalid_selection_is_rejected(selected):
    with pytest.raises(ValueError):
        omission_decomposition(torch.ones(1, 1, 1, 3) / 3, torch.ones(1, 1, 3, 2), [selected], 2)


def test_original_positions_not_compact_slot_indices_are_recorded():
    packed = SimpleNamespace(
        snapshot=SimpleNamespace(layers=[SimpleNamespace(key=torch.zeros(1, 1, 2, 3))]),
        slot_records=[
            dict(layer=0, kv_head=0, slot=1, logical_position=9, tier="exact"),
            dict(layer=0, kv_head=0, slot=0, logical_position=4, tier="exact"),
        ],
    )
    assert retained_positions(packed) == [[[4, 9]]]
    packed.slot_records[0]["tier"] = "prototype"
    with pytest.raises(ValueError, match="transformed"):
        retained_positions(packed)


def test_actual_layer_comparison_does_not_use_surrogate_scores():
    full = {2: [{k: torch.ones(3) for k in PrefixOutputRecorder.FIELDS}]}
    other = {2: [{k: torch.ones(3) * 2 for k in PrefixOutputRecorder.FIELDS}]}
    rows = compare_layer_outputs(full, other)
    assert len(rows) == 6
    assert all(r["maximum_absolute_error"] == 1 for r in rows)
    with pytest.raises(ValueError, match="paired"):
        compare_layer_outputs(full, {})


def test_only_authoritative_grid_coordinates_are_reported_not_ocr_labels():
    spans = [dict(start=2, end=8, modality="image", image_index=0, grid_shape=[2, 3])]
    assert position_metadata(6, spans) == dict(
        modality="image", image_index=0, grid_row=1, grid_column=1
    )
    assert position_metadata(0, spans)["modality"] == "unknown"
    spans[0]["end"] = 9
    assert position_metadata(6, spans)["grid_row"] is None
