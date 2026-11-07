from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.llava import Llava15Adapter
from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy, _bytes, plan_layer


def arguments():
    torch.manual_seed(11)
    n, h, d = 64, 4, 128
    attention = SimpleNamespace(q_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16())
    prepared = SimpleNamespace(
        modality_map=(
            ModalitySpan(0, 56, Modality.IMAGE, image_index=0),
            ModalitySpan(56, 64, Modality.TEXT),
        )
    )
    original = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    parent = tuple((t.float() + 0.03 * torch.randn_like(t.float())).bfloat16() for t in original)
    hidden = torch.randn(1, n, h * d).bfloat16()
    phases = (torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16())
    probs = torch.softmax(torch.randn(1, h, n, n), -1).bfloat16()
    args = (
        SimpleNamespace(capabilities=Llava15Adapter.capabilities),
        prepared,
        attention,
        original,
        hidden,
        phases,
        probs,
        ((*range(7), n - 1),) * h,
    )
    return args, parent


@pytest.mark.parametrize(
    "name",
    (
        "error_utility",
        "demand_graph",
        "opportunity_route",
        "all_adaptive",
        "no_forecast",
        "no_graph",
        "no_value",
        "modular_selection",
        "exact_members",
        "no_repair",
        "forced_repair",
        "individual_delta_fast_graph",
    ),
)
def test_error_aware_variants_use_same_real_storage_and_preserve_mandatory(name):
    from storage_six_ablation import component_policies

    args, parent = arguments()
    policies = component_policies(allocation_study=True)
    baseline = plan_layer(
        *args,
        OverlayPolicy(**policies["individual_delta"]),
        pin_residuals=False,
        parent_pair=parent,
    )
    actual = plan_layer(
        *args, OverlayPolicy(**policies[name]), pin_residuals=False, parent_pair=parent
    )
    assert _bytes(actual.gpu_tensors()) == _bytes(baseline.gpu_tensors()) == 6713
    assert actual.cpu_bytes == baseline.cpu_bytes == 1024
    assert not set(actual.evidence["prototype_members"]) & set(
        actual.evidence["original_protected_positions"]
    )
    if policies[name]["demand_weighted_coverage"]:
        report = actual.evidence["per_query_mass_guard"]
        assert report["minimum_query_mass_fraction"] == 0.95
        assert all(r["final_minimum_query_mass_ratio"] >= 0.95 for r in report["routes"])
    if name == "individual_delta_fast_graph":
        assert actual.evidence["selected_extra"] == baseline.evidence["selected_extra"]
        assert actual.evidence["prototype_members"] == baseline.evidence["prototype_members"]
        assert all(
            torch.equal(a, b) for a, b in zip(actual.exact_pair, baseline.exact_pair, strict=True)
        )


def test_opportunity_routes_to_feasible_unprotected_parent_error_not_large_original_values():
    args, parent = arguments()
    original = args[3]
    parent = tuple(t.clone() for t in original)
    parent[1][0, 2, 40:44].add_(1)
    policy = OverlayPolicy(
        record_phase_timings=True,
        routing_signal="parent_error_opportunity",
        prototype_representation="parent_residual_delta",
    )
    overlay = plan_layer(*args, policy, pin_residuals=False, parent_pair=parent)
    assert overlay.head == 2
    assert overlay.evidence["routing_signal"] == "parent_error_opportunity"


@pytest.mark.parametrize(
    "changes",
    (
        {"utility_signal": "unknown"},
        {"routing_signal": "unknown"},
        {"demand_weighted_coverage": 1},
        {"graph_accumulation": "approximate"},
    ),
)
def test_flags_strict(changes):
    with pytest.raises(ValueError):
        replace(OverlayPolicy(), **changes)
