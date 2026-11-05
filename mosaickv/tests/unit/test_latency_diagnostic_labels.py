import ast
import importlib
import inspect
import json
import sys

import annotate_latency_diagnostic_labels as annotations
import pytest

from mosaickv.evaluation.fox_campaign import sha, write_once


@pytest.mark.parametrize(
    "module,expected",
    [
        ("evicastkv_deferred_exact_constructor", "consumed_deferred_exact_constructor"),
        ("evicastkv_graph_accounting", "consumed_graph_accounting_execution_screen"),
        ("evicastkv_event_only_reader_profile", "consumed_event_only_native_model_attribution"),
        ("evicastkv_host_dispatch_profile", "consumed_host_decode_attribution"),
    ],
)
def test_wrapper_wires_its_own_manifest_variant_without_running_jobs(module, expected):
    wrapper = importlib.import_module(module)
    assert expected == wrapper.MANIFEST_VARIANT
    tree = ast.parse(inspect.getsource(wrapper.main))
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "MANIFEST_VARIANT"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "MANIFEST_VARIANT"
        for node in ast.walk(tree)
    )


def fixture(root, *, complete=True):
    plan = root / "plan.json"
    write_once(plan, dict(stage="fixture", source_sha="a" * 40))
    write_once(
        root / "analysis/decision.json",
        dict(complete=complete, failures=[], source_sha="a" * 40, plan_sha256=sha(plan)),
    )
    for i in (0, 1):
        write_once(
            root / f"workers/{i}/manifest.json",
            dict(
                variant="old", source_sha="a" * 40, plan_sha256=sha(plan), independent_quality_N=0
            ),
        )


def test_annotation_binds_both_manifests_without_rewriting_them(tmp_path):
    fixture(tmp_path)
    files = list(tmp_path.rglob("*.json"))
    before = {p: sha(p) for p in files}
    row = annotations.annotate(tmp_path, "fixture", "old", "corrected")
    assert row["descriptive_variant_correction"] == "corrected"
    assert len(row["manifests"]) == 2
    assert {p: sha(p) for p in files} == before
    with pytest.raises(ValueError):
        annotations.annotate(tmp_path, "wrong_stage", "old", "corrected")


def test_incomplete_result_cannot_be_annotated_as_complete(tmp_path):
    fixture(tmp_path, complete=False)
    with pytest.raises(ValueError):
        annotations.annotate(tmp_path, "fixture", "old", "corrected")


@pytest.mark.parametrize("passed", [False, True])
def test_CPU_analysis_is_fail_closed_and_cannot_submit_GPU(tmp_path, monkeypatch, passed):
    import evicastkv_lt_latency

    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotator",
            "--output",
            str(output),
            "--cpu-proof-root",
            str(tmp_path),
            "--plan-sha",
            "b" * 64,
        ],
    )
    monkeypatch.setattr(annotations, "clean_source", lambda p: "a" * 40)
    monkeypatch.setattr(evicastkv_lt_latency.driver, "checked", lambda *a: {})

    def proof(*args):
        if not passed:
            raise RuntimeError("incomplete CPU XML")

    monkeypatch.setattr(evicastkv_lt_latency.driver, "cpu_proof", proof)
    annotations.main()
    row = json.loads(output.read_text())
    assert row["complete"] is passed
    assert row["no_GPU_submission"] is True
    assert bool(row["failures"]) is not passed


def test_CPU_analysis_requires_frozen_hash_before_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotator",
            "--output",
            str(tmp_path / "analysis.json"),
            "--cpu-proof-root",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(annotations, "clean_source", lambda p: "a" * 40)
    with pytest.raises(ValueError, match="immutable plan hash"):
        annotations.main()


@pytest.mark.parametrize(
    "module,stage",
    [
        ("evicastkv_event_only_reader_profile", "event_only_native_model_attribution"),
        ("evicastkv_host_dispatch_profile", "consumed_host_decode_attribution"),
        ("evicastkv_constructor_host_profile", "consumed_constructor_host_attribution"),
    ],
)
def test_each_profile_uses_its_actual_timing_scope(module, stage):
    wrapper = importlib.import_module(module)
    assert annotations.timing_scope(stage) == wrapper.MANIFEST_TIMING_PROTOCOL
    assert wrapper.MANIFEST_TIMING_PROTOCOL["not_latency_evidence"]
    tree = ast.parse(inspect.getsource(wrapper.main))
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "MANIFEST_TIMING_PROTOCOL"
            for target in node.targets
        )
        for node in ast.walk(tree)
    )
    if module == "evicastkv_constructor_host_profile":
        assert wrapper.MANIFEST_TIMING_PROTOCOL["decode_excluded"]
        assert "profiled_steps" not in wrapper.MANIFEST_TIMING_PROTOCOL


def test_unknown_timing_annotation_rejected():
    with pytest.raises(ValueError):
        annotations.timing_scope("unregistered")
