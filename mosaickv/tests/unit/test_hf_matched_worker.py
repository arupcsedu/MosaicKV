"""Synthetic worker preparation/gate tests; no real models or datasets."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.baselines.prefixkv import (
    PrefixKVCalibrationObservation,
    generate_prefixkv_byte_profile,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "matched_worker", SCRIPTS / "run_hf_matched_validation.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
finally:
    sys.path.remove(str(SCRIPTS))


def fixture_inputs(tmp_path):
    parent, prior = tmp_path / "parent", tmp_path / "prior"
    for task in worker.TASKS:
        meta = dict(
            task=task, dataset_id=f"synthetic/{task}", dataset_revision="b" * 40, split="test"
        )
        worker.write_once(
            parent / task / "00.json", {**meta, "sample_ids": [f"{task}:{i}" for i in range(400)]}
        )
        worker.write_once(
            prior / f"{task}_evaluation.json", {"sample_ids": [f"{task}:{i}" for i in range(10)]}
        )
        observation = PrefixKVCalibrationObservation(
            sample_id=f"{task}:cal",
            layer_scores=tuple((1.0,) * 200 for _ in range(32)),
            question_positions=(195, 196, 197),
            question_layer_scores=tuple((1.0,) * 200 for _ in range(32)),
            question_mapping_sha256="c" * 64,
        )
        capture = prior / "outputs" / task / "calibration/00.json"
        worker.write_once(
            capture,
            dict(
                sample_id=observation.sample_id,
                layer_scores=observation.layer_scores,
                question_positions=observation.question_positions,
                question_layer_scores=observation.question_layer_scores,
                question_mapping_sha256=observation.question_mapping_sha256,
                bytes_per_layer_position=16384,
            ),
        )
        worker.write_once(
            prior / "outputs" / task / "frozen_ablations.json",
            {
                "calibration_captures": {
                    observation.sample_id: {"path": str(capture), "sha256": worker.digest(capture)}
                }
            },
        )
        for arm, alpha in [("control", 0.0), ("c_rank", 0.5)]:
            p = generate_prefixkv_byte_profile(
                (observation,),
                byte_cap=40 * 2**20,
                bytes_per_layer_position=16384,
                model_id=worker.MODEL,
                model_revision=worker.REVISION,
                dataset_id=meta["dataset_id"],
                dataset_revision=meta["dataset_revision"],
                calibration_split=meta["split"],
                evaluation_sample_ids=(),
                seed=20271001,
                question_score_weight=alpha,
            )
            p.write(prior / "outputs" / task / "profiles" / f"{arm}.json")
    worker.write_once(prior / "plan.json", {"classification": "synthetic_fixture_not_measured"})
    return parent, prior


def test_prepare_freezes_all_caps_before_validation_and_disjoint_larger_stage(
    tmp_path, monkeypatch
):
    parent, prior = fixture_inputs(tmp_path)
    monkeypatch.setattr(worker, "_require_clean_source", lambda *a: "a" * 40)
    root = tmp_path / "new"
    worker.prepare(
        SimpleNamespace(
            root=root,
            source_sha="a" * 40,
            parent_root=parent,
            profile_root=prior,
            exclude_root=[prior],
            validation_count=64,
            larger_count=256,
            shard_size=32,
        )
    )
    p = json.loads((root / "plan.json").read_text())
    assert len(p["jobs"]) == 68
    assert sum(j["stage"] == "validation" for j in p["jobs"]) == 4
    for name, digest in p["bound_files"].items():
        assert worker.digest(name) == digest
    for task in worker.TASKS:
        ids = {}
        for stage in ["validation", "larger"]:
            ids[stage] = {
                s
                for j in p["jobs"]
                if j["task"] == task and j["stage"] == stage
                for s in json.loads(Path(j["ledger"]).read_text())["sample_ids"]
            }
        assert len(ids["validation"]) == 64 and len(ids["larger"]) == 256
        assert not ids["validation"] & ids["larger"]
        assert not (ids["validation"] | ids["larger"]) & {f"{task}:{i}" for i in range(10)}
        for arm in ["control", "c_rank"]:
            assert json.loads(Path(p["profiles"][task]["40"][arm]).read_text()) == json.loads(
                (prior / "outputs" / task / "profiles" / f"{arm}.json").read_text()
            )
    assert not p["paper_claim_authorized"] and not p["heldout_claim"]


def test_capture_tampering_fails_before_profile_regeneration(tmp_path):
    _, prior = fixture_inputs(tmp_path)
    capture = prior / "outputs/textvqa/calibration/00.json"
    capture.write_text("{}")
    with pytest.raises(RuntimeError, match="capture changed"):
        worker.freeze_cap_profiles(prior, "textvqa", tmp_path / "out", {})


def test_larger_requires_both_task_gates_and_unchanged_inputs(tmp_path):
    root = tmp_path / "run"
    path = root / "analysis/prefixkv-validation.json"
    worker.write_once(
        path,
        {
            "plan_sha256": "a" * 64,
            "all_gates_passed": True,
            "tasks": {"textvqa": {"gates_passed": True}, "docvqa": {"gates_passed": False}},
            "inputs": {},
        },
    )
    with pytest.raises(RuntimeError, match="blocked"):
        worker.require_larger_gate(root, "a" * 64)
    with pytest.raises(RuntimeError, match="blocked"):
        worker.require_larger_gate(root, "b" * 64)
