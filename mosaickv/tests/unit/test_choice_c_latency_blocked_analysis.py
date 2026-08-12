from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _script_module(filename: str) -> ModuleType:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            f"{Path(filename).stem}_test",
            scripts / filename,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_mode_order_is_balanced_within_each_gpu() -> None:
    worker = _script_module("run_choice_c_latency_blocked_worker.py")

    for rank in (0, 1):
        orders = [worker.mode_order(rank, cycle) for cycle in range(4)]
        assert orders.count(("reference", "optimized")) == 2
        assert orders.count(("optimized", "reference")) == 2
    for cycle in range(4):
        assert worker.mode_order(0, cycle) == tuple(reversed(worker.mode_order(1, cycle)))


def test_blocked_effect_pairs_within_gpu_cycle_and_bootstraps_blocks() -> None:
    analyzer = _script_module("analyze_choice_c_latency_blocked.py")
    rows = []
    for rank in (0, 1):
        for cycle in range(4):
            modes = (
                ("reference", "optimized")
                if (rank + cycle) % 2 == 0
                else ("optimized", "reference")
            )
            for order_index, mode in enumerate(modes):
                for sample in range(4):
                    rows.append(
                        {
                            "rank": rank,
                            "cycle": cycle,
                            "sample_id": f"sample:{sample}",
                            "order_index": order_index,
                            "compute_mode": mode,
                            "end_to_end": 10.0 if mode == "reference" else 9.5,
                        }
                    )

    effect, blocks = analyzer._paired_effect(pd.DataFrame(rows), "end_to_end")

    assert effect["pair_count"] == 32
    assert effect["block_count"] == 8
    assert effect["relative_reduction"] == pytest.approx(0.05)
    assert effect["positive_block_count"] == 8
    assert effect["block_bootstrap_95_reduction_seconds"] == pytest.approx([0.5, 0.5])
    assert set(blocks["optimized_order"]) == {0, 1}
