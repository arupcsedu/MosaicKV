import pytest
import torch

from mosaickv.evaluation.precision_transaction_study import attend, restoration_errors


def test_linear_cost_formula_matches_every_actual_single_row_restoration():
    torch.manual_seed(212)
    original = tuple(torch.randn(23, 16).bfloat16() for _ in range(2))
    parent = tuple((t.float() + torch.randn_like(t.float()) * 0.03).bfloat16() for t in original)
    queries = torch.randn(8, 16).bfloat16()
    ids = (0, 3, 15, 21, 22)
    observed = restoration_errors(queries, original, parent, ids)
    target = attend(queries, original, causal=True)
    for slot, i in enumerate(ids):
        actual = tuple(t.clone() for t in parent)
        for value, source in zip(actual, original, strict=True):
            value[i] = source[i]
        error = (attend(queries, actual, causal=True) - target).square().sum(-1)
        assert torch.allclose(observed[:, slot], error, atol=1e-12, rtol=1e-8)
    # Future prompt positions cannot influence earlier causal-query scores.
    baseline_error = (attend(queries, parent, causal=True) - target).square().sum(-1)
    assert torch.equal(observed[:6, -1], baseline_error[:6])


@pytest.mark.parametrize("mode", ("exact", "prototype", "joint_guarded"))
def test_live_planner_transactions_keep_capacity_and_prototype_membership(mode):
    from test_storage_error_aware import arguments

    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy, _bytes, plan_layer

    args, parent = arguments()
    before = tuple(t.clone() for t in (*args[3], *parent))
    actual = plan_layer(
        *args,
        OverlayPolicy(
            forecast_distribution="individual_queries",
            prototype_representation="parent_residual_delta",
            precision_transaction=mode,
            record_phase_timings=True,
        ),
        pin_residuals=False,
        parent_pair=parent,
    )
    proof = actual.evidence["precision_transaction_evidence"]
    assert _bytes(actual.gpu_tensors()) == 6713
    assert actual.cpu_bytes == 1024
    assert (
        len(actual.evidence["selected_extra"]) == len(actual.evidence["submodular_seed_extra"]) == 4
    )
    assert proof["fixed_prototype_members_preserved"] and proof["final_joint_guard_passed"]
    assert actual.evidence["prototype_members"] == proof["fixed_prototype_members"]
    for kind in proof["baseline"]:
        for boundary in ("early", "late", "maximum"):
            assert (
                proof["final_metrics"][kind][boundary] <= proof["baseline"][kind][boundary] + 1e-12
            )
    if mode == "prototype":
        assert actual.evidence["selected_extra"] == actual.evidence["submodular_seed_extra"]
    if mode == "exact":
        assert not proof["accepted_prototype"]
    assert all(torch.equal(a, b) for a, b in zip(before, (*args[3], *parent), strict=True))


def test_transaction_cannot_accept_benchmark_labels_or_oracle_predictions():
    from mosaickv.evaluation.precision_transaction_study import plan_precision_transaction

    with pytest.raises(ValueError, match="strict label-free"):
        plan_precision_transaction({"answer": "no", "reference": "yes", "logits": torch.ones(2)})


def test_ablations_do_not_leave_the_disabled_mechanism_in_a_hidden_refiner():
    from storage_six_ablation import component_policies

    policies = component_policies(transaction_study=True)
    assert policies["no_forecast"]["precision_transaction"] == "none"
    assert policies["no_value"]["precision_transaction"] == "prototype"
