from dataclasses import fields

import pytest
from test_source_row_geometry import packet

from mosaickv.evaluation.source_row_geometry import apply, geometry


@pytest.mark.parametrize("warps", [4, 8, 16])
def test_warp_override_changes_only_launch_metadata(warps):
    old = packet()
    new = geometry(old, 16, 4, num_warps=warps)
    assert new.source_num_warps == warps
    assert old.source_num_warps is None
    for field in fields(old):
        if field.name != "source_num_warps":
            assert getattr(new, field.name) is getattr(old, field.name)


@pytest.mark.parametrize("bad", [True, False, 0, 3, 32, 8.0, "8"])
def test_unaudited_warp_counts_fail_even_for_ineligible_baseline(bad):
    with pytest.raises(ValueError, match="warp"):
        geometry(packet(), 16, 4, num_warps=bad)
    with pytest.raises(ValueError, match="warp"):
        apply((object(),), (16, 4), num_warps=bad)


def test_default_uses_original_launch_and_unrelated_baseline_is_untouched():
    old = packet()
    assert geometry(old, 16, 4).source_num_warps is None
    baseline = object()
    assert apply((baseline,), (64, 4), num_warps=16)[0] is baseline


def test_phase_evidence_required_on_both_inputs_and_no_failures():
    from evicastkv_source_warp_followup import profile_justifies_tuning

    phase = "mosaickv.evaluation.shared_source_kernel.project"
    groups = [
        dict(
            method="mosaickv_exact",
            input_index=i,
            phases={phase: dict(mean_inclusive_cuda_stream_ms=500)},
        )
        for i in (0, 1)
    ]
    data = dict(complete=True, failures=[], summaries=groups)
    assert profile_justifies_tuning(data)
    assert not profile_justifies_tuning(dict(data, complete=False))
    assert not profile_justifies_tuning(dict(data, failures=["changed logits"]))
    assert not profile_justifies_tuning(dict(data, summaries=groups[:1]))
    groups[1]["phases"][phase]["mean_inclusive_cuda_stream_ms"] = 249
    assert not profile_justifies_tuning(data)
