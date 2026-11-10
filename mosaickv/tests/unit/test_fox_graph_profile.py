"""Synthetic CPU-only observational equivalence, never model-quality evidence."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from mosaickv.cache_state import FullKVState
from mosaickv.config import GraphConfig
from mosaickv.evaluation.fox_graph_profile import GraphCPUProfiler
from mosaickv.graph import build_evidence_graph


def test_real_graph_is_exactly_unchanged_and_subphases_are_measured():
    key = np.arange(24, dtype=np.float32).reshape(1, 1, 8, 3)
    state = FullKVState.from_tensors(((key, key + 1),), block_size=1)
    before = key.copy()
    config = GraphConfig(max_neighbors=2)
    original = build_evidence_graph(state, config)
    profile = GraphCPUProfiler(maximum_graph_calls=1)
    result = profile.wrap(build_evidence_graph)(state, config)
    np.testing.assert_equal(asdict(result), asdict(original))
    assert np.array_equal(key, before)
    assert len(profile.records) == 1
    functions = profile.records[0]["functions"]
    assert {r["graph_subphase"] for r in functions} >= {
        "complete_graph",
        "descriptor_pooling",
        "cosine_edges",
        "structural_edges",
        "edge_finalization",
        "graph_diagnostics",
    }
    for row in functions:
        assert row["total_calls"] >= row["primitive_calls"] >= 1
        assert row["cumulative_thread_cpu_seconds"] >= row["exclusive_thread_cpu_seconds"] >= 0
    assert profile.report()["profiled_request_latency_eligible"] is False


def test_return_identity_and_mutable_inputs_are_not_copied_or_rewritten():
    item = SimpleNamespace(nodes=(1,), row_indices=())
    args = {"value": [1, 2]}

    def original(value):
        assert value is args["value"]
        return item

    profile = GraphCPUProfiler(maximum_graph_calls=1)
    assert profile.wrap(original)(**args) is item
    assert args == {"value": [1, 2]}


def test_failure_is_preserved_and_call_budget_is_enforced():
    def failure():
        raise ValueError("synthetic graph failure")

    profile = GraphCPUProfiler(maximum_graph_calls=1)
    wrapped = profile.wrap(failure)
    with pytest.raises(ValueError, match="synthetic graph failure"):
        wrapped()
    assert profile.records[0]["status"] == "failed"
    assert profile.records[0]["functions"]
    with pytest.raises(RuntimeError, match="budget exceeded"):
        wrapped()
    assert profile.report()["actual_graph_calls"] == 1


def test_runtime_patch_is_isolated_and_restored():
    import mosaickv.backends.hf_runtime as runtime

    original = runtime.build_evidence_graph
    profile = GraphCPUProfiler()
    with pytest.raises(RuntimeError, match="deliberate"), profile.install():
        assert runtime.build_evidence_graph is not original
        raise RuntimeError("deliberate")
    assert runtime.build_evidence_graph is original


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_bounded_scope_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        GraphCPUProfiler(maximum_graph_calls=bad)


@pytest.fixture
def consumed_plan(tmp_path, monkeypatch):
    import evicastkv_graph_profile as cli

    prior, source, parent = (tmp_path / name for name in ("prior", "source", "parent"))
    parent.mkdir()
    names = (
        "mosaickv/scripts/evicastkv_worker.py",
        "mosaickv/scripts/evicastkv_campaign.py",
        "mosaickv/src/mosaickv/evaluation/fox_numerics.py",
        "mosaickv/src/mosaickv/evaluation/fox_campaign.py",
        "mosaickv/configs/experiments/evicastkv_fox/coco/mosaickv_exact.json",
    )
    for base in (source, prior):
        for name in names:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unchanged synthetic fixture")
    ledger = parent / "consumed.json"
    ledger.write_text(json.dumps({"samples": [{"sample_id": f"id-{i}"} for i in range(8)]}))
    lock = dict(
        source_root=str(prior),
        source_sha="a" * 40,
        arithmetic={},
        cohorts={
            "coco": {
                "profiler": dict(
                    path=str(ledger),
                    sha256=cli.sha(ledger),
                    sample_ids=[f"id-{i}" for i in range(8)],
                )
            }
        },
    )
    (parent / "protocol_lock.json").write_text(json.dumps(lock))
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        gate = parent / "analysis" / stage / f"GATE_{stage}.json"
        gate.parent.mkdir(parents=True)
        gate.write_text(
            json.dumps(
                dict(
                    status="passed",
                    blockers=[],
                    source_sha="a" * 40,
                    protocol_sha256=cli.sha(parent / "protocol_lock.json"),
                )
            )
        )
    monkeypatch.setattr(cli, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(cli, "runtime_files", lambda *args: {"test-runtime": "unchanged"})
    return cli, parent, source


def test_plan_selects_two_consumed_inputs_without_reading_labels(consumed_plan):
    cli, parent, source = consumed_plan
    lock, bindings, ids = cli.registered_plan(parent, source=source)
    assert ids == ["id-0", "id-1"]
    assert len(lock["cohorts"]["coco"]["profiler"]["sample_ids"]) == 8
    assert len(bindings) == 5


def test_changed_model_worker_blocks_diagnostic(consumed_plan):
    cli, parent, source = consumed_plan
    (source / "mosaickv/scripts/evicastkv_worker.py").write_text("changed")
    with pytest.raises(ValueError, match="worker, numerical control or policy changed"):
        cli.registered_plan(parent, source=source)


def test_failed_inherited_gate_blocks_diagnostic(consumed_plan):
    cli, parent, source = consumed_plan
    gate = parent / "analysis/measurement/GATE_measurement.json"
    value = json.loads(gate.read_text())
    value["status"] = "blocked"
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="parent control not passed: measurement"):
        cli.registered_plan(parent, source=source)


def test_modified_consumed_ledger_blocks_diagnostic(consumed_plan):
    cli, parent, source = consumed_plan
    (parent / "consumed.json").write_text('{"samples": []}')
    with pytest.raises(ValueError, match="profiler ledger changed"):
        cli.registered_plan(parent, source=source)
