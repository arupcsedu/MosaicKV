from dataclasses import asdict

import pytest

from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_legacy_default_false_policy_field_is_resolved_not_assumed_present():
    from component_admission_benchmark import check_retained_policy

    old = asdict(OverlayPolicy())
    del old["graph_direct_metadata_key"]
    assert check_retained_policy(old).graph_direct_metadata_key is False
    assert check_retained_policy(asdict(OverlayPolicy())).graph_direct_metadata_key is False
    old.update(graph_structural_templates=True, graph_direct_metadata_key=True)
    with pytest.raises(ValueError, match="retained corrected"):
        check_retained_policy(old)
    old.update(
        graph_direct_metadata_key=False,
        anchor_counterfactual="single_head",
        live_anchor_guard=True,
        inline_repair=True,
    )
    with pytest.raises(ValueError, match="retained corrected"):
        check_retained_policy(old)
