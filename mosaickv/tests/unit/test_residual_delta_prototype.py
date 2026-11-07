import pytest
import torch

from mosaickv.evaluation.residual_delta_prototype import construct, study


def fixture():
    original = torch.zeros(8, 8).bfloat16(), torch.ones(8, 8).bfloat16()
    parent = tuple(t.clone() for t in original)
    parent[1][2:4] -= 0.25
    return dict(
        original_pair=original,
        parent_pair=parent,
        phases=(torch.ones(8, 8).bfloat16(), torch.zeros(8, 8).bfloat16()),
        prompt_queries=torch.ones(4, 8).bfloat16(),
        evidence=dict(selected_extra=[4], prototype_members=[2, 3]),
        overlay=dict(
            exact_pair=tuple(t[4:5].clone() for t in original),
            member_keys=torch.zeros(2, 8),
            prototype_value=torch.ones(1, 8),
        ),
        evaluation_only=True,
        labels_accessed=False,
    )


def test_residual_not_full_value_and_originals_unchanged():
    p = fixture()
    pair, dispersion = construct(
        p["original_pair"], p["parent_pair"], p["phases"], (2, 3), 4, (1.0, 1.0)
    )
    assert (pair[0] == 0).all() and (pair[1] == 0.25).all()
    assert dispersion == 0
    assert (p["parent_pair"][1][2:4] == 0.75).all()
    result = study(p)
    assert result["variants"]["joint"]["mean_squared_output_error"] == 0
    assert result["variants"]["parent"]["mean_squared_output_error"] > 0
    assert result["no_source_memory_savings_claim"]


def test_phase_neutral_residuals_and_normalized_weights():
    p = fixture()
    p["original_pair"][0][2, :4] = 1
    p["original_pair"][0][3, 4:] = 1
    p["phases"][0][3] = 0
    p["phases"][1][3] = 1
    pair, _ = construct(p["original_pair"], p["parent_pair"], p["phases"], (2, 3), 4, (3.0, 1.0))
    assert (pair[0][0, :4] == 1).all()
    assert (pair[0][0, 4:] == 0).all()


@pytest.mark.parametrize(
    "members,anchor,weights",
    [
        ((2, 2), 4, (1.0, 1.0)),
        ((2,), 4, (1.0,)),
        ((2, 3), 8, (1.0, 1.0)),
        ((2, 3), 4, (0.0, 1.0)),
        ((2, 3), 4, (float("nan"), 1.0)),
    ],
)
def test_fail_closed(members, anchor, weights):
    p = fixture()
    with pytest.raises(ValueError):
        construct(p["original_pair"], p["parent_pair"], p["phases"], members, anchor, weights)


def test_live_delta_reads_add_to_parent_then_promote_original_without_extra_storage():
    from test_storage_six_overlay import overlay

    from mosaickv.evaluation.storage_six_overlay import _bytes

    o = overlay()
    o.prototype_representation = "parent_residual_delta"
    before = _bytes(o.gpu_tensors())
    destination = torch.ones(1, 2, 8, 8)
    o.fill(1, destination)
    assert (destination[0, 1, 2:4] == 4).all()
    assert (destination[0, 1, 4] == 8).all()
    assert (destination[0, 0] == 1).all()
    o.promote()
    destination.fill_(1)
    o.fill(1, destination)
    assert (destination[0, 1, 2:4] == 5).all()
    assert _bytes(o.gpu_tensors()) == before


def test_proposal_policies_strict_separate_single_effects_and_same_allowance():
    from storage_six_ablation import component_policies

    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    policies = {name: OverlayPolicy(**p) for name, p in component_policies(True).items()}
    assert len(policies) == 7
    assert policies["individual_forecast"].prototype_representation == "absolute_kv"
    assert policies["delta_prototype"].forecast_distribution == "mean_query"
    assert policies["individual_delta"].forecast_distribution == "individual_queries"
    assert policies["individual_delta"].prototype_representation == "parent_residual_delta"
    assert len({p.maximum_extra_gpu_bytes for p in policies.values()}) == 1
    for bad in ({"forecast_distribution": "oracle"}, {"prototype_representation": "unknown"}):
        with pytest.raises(ValueError):
            OverlayPolicy(**bad)
