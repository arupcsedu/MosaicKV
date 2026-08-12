from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _analysis_module() -> ModuleType:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "analyze_choice_c_latency_compute_test",
            scripts / "analyze_choice_c_latency_compute.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the Choice-C compute analyzer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _records(score_delta: float = 0.0, *, selected: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "node_id": 0,
            "selected": selected,
            "selection_rank": 0 if selected else None,
            "marginal_gain": 1.0 + score_delta,
            "marginal_gain_per_cost": 0.5 + score_delta / 2,
            "cost": 2,
            "reason": "lazy_greedy" if selected else "not_selected_lower_gain",
        }
    ]


def test_selection_parity_separates_identity_from_roundoff() -> None:
    module = _analysis_module()
    result = module._selection_parity(
        {"sample:0": [_records(), _records(5e-16), _records(), _records(5e-16)]},
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )

    assert result["identity_parity"]
    assert result["score_parity"]
    assert result["non_bitwise_score_count"] > 0


def test_selection_parity_rejects_identity_or_material_score_changes() -> None:
    module = _analysis_module()
    identity = module._selection_parity(
        {"sample:0": [_records(), _records(selected=False), _records(), _records()]},
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    score = module._selection_parity(
        {"sample:0": [_records(), _records(1e-6), _records(), _records()]},
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )

    assert not identity["identity_parity"]
    assert not score["score_parity"]
