"""Dependency-free analysis rendering and fail-closed source recovery."""

import json
import xml.etree.ElementTree as ET

import evicastkv_token_eviction_audit as module
import pytest

from mosaickv.evaluation.fox_campaign import sha
from mosaickv.evaluation.nll_svg import cumulative_nll_svg


def fixture_rows():
    return [
        dict(
            task=task,
            reference="full_kv_reference",
            role="window_topk_control",
            per_token_deltas=[0.0, 1.0, -0.5],
            first_generated_eviction_affected_prediction=3,
        )
        for task in ["llava_description", "mmvet_prefixkv"]
    ]


def test_svg_has_two_real_curves_and_observed_event_markers_without_matplotlib():
    rows = fixture_rows()
    rows[0]["role"] = "policy<&>"
    xml = cumulative_nll_svg(rows)
    svg = ET.fromstring(xml)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert len(svg.findall("s:path", ns)) == 2
    assert len(svg.findall("s:line[@data-boundary='fifo']", ns)) == 1
    assert "policy&lt;&amp;&gt;" in xml


def test_svg_rejects_nonfinite_or_missing_data():
    rows = fixture_rows()
    rows[0]["per_token_deltas"] = [float("nan")]
    with pytest.raises(ValueError, match="nonfinite"):
        cumulative_nll_svg(rows)
    with pytest.raises(ValueError, match="both task"):
        cumulative_nll_svg(fixture_rows()[:1])


@pytest.mark.parametrize(
    "path",
    ["mosaickv/src/mosaickv/evaluation/nll_svg.py", "mosaickv/src/mosaickv/backends/hf_runtime.py"],
)
def test_cpu_recovery_allows_rendering_but_rejects_runtime_changes(tmp_path, monkeypatch, path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(source_sha="producer", bound_files={})))
    monkeypatch.setattr(module, "clean_source", lambda *_: "recovery")
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_a, **_k: path + "\n")
    if "hf_runtime" in path:
        with pytest.raises(ValueError, match="allowlist"):
            module.analysis_inputs(tmp_path, sha(plan))
    else:
        assert module.analysis_inputs(tmp_path, sha(plan))[1] == "recovery"
