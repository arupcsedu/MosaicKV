"""Synthetic CPU regression tests for post-processing, not measured results."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def tool(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prefix_campaign_analysis", scripts / "analyze_prefixkv_full_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_only_fullkv_never_enters_shared_byte_gate(monkeypatch):
    module = tool(monkeypatch)
    rows = []
    for method in ["full_kv", "prefixkv_reimpl", "prompt_attention_topk", "mosaickv_exact"]:
        rows.append(
            dict(
                task="ci",
                lane="shared_byte_40mib",
                method=method,
                sample_id="fixture",
                **{field: (400 if method == "full_kv" else 40) * 2**20 for field in module.STORAGE},
            )
        )
    frame = pd.DataFrame(rows)
    assert module.check_shared_storage(frame, {"ci": 1})
    frame.loc[frame.method == "mosaickv_exact", module.STORAGE[-1]] -= 1
    assert not module.check_shared_storage(frame, {"ci": 1})


def test_ppl_point_difference_is_not_bootstrap_replicate_mean(monkeypatch):
    module = tool(monkeypatch)
    ref = pd.DataFrame(dict(sample_id=["a", "b"], numerator=[1.0, 8.0], denominator=[1, 7]))
    cand = pd.DataFrame(dict(sample_id=["a", "b"], numerator=[2.0, 8.0], denominator=[1, 7]))
    point, low, high = module._paired_bootstrap(
        ref, cand, metric="teacher_forced_ppl", replicates=100, seed=7
    )
    assert point == pytest.approx(
        module._metric_value(cand, "teacher_forced_ppl")[0]
        - module._metric_value(ref, "teacher_forced_ppl")[0]
    )
    assert low <= point <= high
