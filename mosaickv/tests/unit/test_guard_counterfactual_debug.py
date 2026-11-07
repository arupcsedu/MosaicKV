import importlib.util
from pathlib import Path

import pytest
import torch


def load():
    path = Path(__file__).parents[2] / "scripts/debug_guard_counterfactual.py"
    spec = importlib.util.spec_from_file_location("guard_counterfactual_debug", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_and_first_changed_vector_are_distinguished_from_token_change():
    delta = load().delta
    p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert delta(p, p)["all_vectors_bitwise"]
    other = p.clone()
    other[1, 0] += 0.01
    result = delta(p, other)
    assert result["changed_prefix_indices"] == [1]
    assert result["changed_greedy_prefix_indices"] == []
    other[1, 0] = 5
    assert delta(p, other)["changed_greedy_prefix_indices"] == [1]
    with pytest.raises(ValueError):
        delta(p, other[:1])
