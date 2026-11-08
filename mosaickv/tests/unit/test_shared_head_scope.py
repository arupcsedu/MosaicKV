import pytest
import torch

from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_head_scope import choose_group


def test_source_groups_obey_text_quantizer_and_all_head_demand():
    torch.manual_seed(32)
    original = torch.zeros(1, 1, 256, 64).bfloat16()
    original[:, :, 8:16] = torch.randn(1, 1, 8, 64).bfloat16()
    body = pack(original, bits=3, axis=2)
    group, proof = choose_group(body, tuple(range(16, 256)), torch.ones(256), 8)
    assert group == tuple(range(8))  # zero source distortion, source-index tie
    assert proof["risk"] == 0
    assert proof["not_the_six_component_submodular_planner"]
    with pytest.raises(ValueError):
        choose_group(body, tuple(range(256)), torch.ones(256), 8)


def test_declared_source_risk_changes_with_prompt_demand_without_label_access():
    torch.manual_seed(42)
    body = pack(torch.randn(1, 1, 128, 64).bfloat16(), bits=3, axis=2)
    demand = torch.ones(128)
    demand[8:16] = 0
    group, _ = choose_group(body, (), demand, 8)
    assert group == tuple(range(8, 16))
