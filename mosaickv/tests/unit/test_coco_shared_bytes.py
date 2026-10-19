"""Synthetic fixtures only; not experimental evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.config import synthetic_smoke_config
from mosaickv.evaluation.coco_shared_bytes import (
    STORAGE_FIELDS,
    TokenizedCider,
    bootstrap_cider_curve,
    control_hashes,
    dense_capacity_certificate,
    sample_size_recommendation,
    validate_caption_rows,
)
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.scorers import aggregate_cider
from mosaickv.evaluation.metrics.statistics import paired_bootstrap
from mosaickv.evaluation.metrics.types import MetricSample
from mosaickv.types import Precision

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def script(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def caption_samples():
    return [
        MetricSample(
            str(i), text, (text, text + " here", text + " nearby", text + " today", text + ".")
        )
        for i, text in enumerate(
            ("a red car on the road", "two birds fly", "a cat sleeps", "a man holding a hat")
        )
    ]


@pytest.mark.parametrize("reference_count", [5, 6, 7])
def test_freeze_sample_preserves_every_coco_reference(tmp_path, reference_count):
    from PIL import Image

    materializer = script("materialize_coco_shared_bytes")
    # Retain annotation order, whitespace and duplicate strings, not just count.
    references = ["  two cats near a window  ", "a cat", "a cat"] + [
        f"caption {index}" for index in range(reference_count - 3)
    ]
    original = references.copy()
    doc = dict(answer=references, image=Image.new("RGB", (2, 2)))
    sample = materializer.freeze_sample(doc, "synthetic-coco-image", tmp_path)
    assert sample["sample_id"] == "synthetic-coco-image"
    assert references == original
    assert sample["references"] == original
    assert len(sample["references"]) == reference_count
    assert json.loads(json.dumps(sample))["references"] == original
    path = Path(sample["media"][0]["path"])
    assert path.is_file()
    assert sample["media"][0]["sha256"] == materializer.sha(path)


@pytest.mark.parametrize(
    "references",
    [
        None,
        "not a caption list",
        (),
        [],
        ["caption"] * 4,
        ["caption"] * 4 + [""],
        ["caption"] * 4 + ["   "],
        ["caption"] * 4 + [None],
    ],
)
def test_freeze_sample_rejects_incomplete_or_malformed_references(tmp_path, references):
    materializer = script("materialize_coco_shared_bytes")
    # Validation must reject before touching media or producing a partial sample.
    with pytest.raises(ValueError, match="at least five nonempty strings"):
        materializer.freeze_sample(dict(answer=references), "invalid", tmp_path)
    assert not (tmp_path / "media").exists()


def test_cider_preserves_mixed_five_six_seven_caption_annotations():
    samples = caption_samples()
    samples[1] = replace(samples[1], references=(*samples[1].references, "birds in the sky"))
    samples[2] = replace(
        samples[2], references=(*samples[2].references, "a sleeping cat", "a sleeping cat")
    )
    assert [len(sample.references) for sample in samples] == [5, 6, 7, 5]
    scorer = TokenizedCider(samples)
    assert scorer(samples) == pytest.approx(aggregate_cider(samples), abs=1e-10)
    draw = [samples[1], samples[2], samples[2], samples[0]]
    assert scorer(draw) == pytest.approx(aggregate_cider(draw), abs=1e-10)


def test_cached_tokens_preserve_official_cider_and_duplicate_draws():
    samples = caption_samples()
    scorer = TokenizedCider(samples)
    assert scorer(samples) == pytest.approx(aggregate_cider(samples), abs=1e-10)
    draw = [samples[0], samples[0], samples[2], samples[3]]
    assert scorer(draw) == pytest.approx(aggregate_cider(draw), abs=1e-10)
    # Mean of singleton scores is zero (IDF); pooled corpus is not.
    assert scorer(samples) > sum(scorer([s]) for s in samples) / len(samples)
    with pytest.raises(ValueError, match="changed"):
        scorer([replace(samples[0], prediction="unfrozen caption")])


def test_bootstrap_recomputes_corpus_for_every_occurrence(monkeypatch):
    samples = caption_samples()
    candidate = [replace(s, prediction="a " + s.prediction) for s in samples]
    scorer = TokenizedCider(samples + candidate)
    calls = []
    original = scorer._scorer

    class CountingCider:
        def compute_score(self, gts, res):
            calls.append((len(gts), len(res)))
            return original().compute_score(gts, res)

    monkeypatch.setattr(scorer, "_scorer", CountingCider)
    definition = replace(default_metric_registry().resolve("cider"), reducer=scorer)
    result = paired_bootstrap(definition, samples, candidate, resamples=7, seed=4)
    assert result.sample_count == 4
    assert calls == [(4, 4)] * 16
    assert result == paired_bootstrap(definition, samples, candidate, resamples=7, seed=4)
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(definition, samples[:-1], candidate, resamples=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_shared_draw_curve_bootstrap_matches_pairwise_cider(workers):
    reference = caption_samples()
    candidate = [replace(sample, prediction="a " + sample.prediction) for sample in reference]
    groups = {("reference", 40.0): reference, ("candidate", 25.5): candidate}
    comparison = (("reference", 40.0), ("candidate", 25.5))
    actual = bootstrap_cider_curve(
        groups,
        [comparison, comparison],
        resamples=7,
        seed=4,
        workers=workers,
    )
    scorer = TokenizedCider(reference + candidate)
    expected = paired_bootstrap(
        replace(default_metric_registry().resolve("cider"), reducer=scorer),
        reference,
        candidate,
        resamples=7,
        seed=4,
    )
    assert actual == {comparison: expected}


@pytest.mark.parametrize("cap", [10, 20, 40, 80])
def test_independent_dense_capacity_largest_feasible(cap):
    layout = [
        dict(
            key_shape=[1, 8, 1000, 128],
            value_shape=[1, 8, 1000, 128],
            key_element_size=2,
            value_element_size=2,
        )
    ] * 32
    result = dense_capacity_certificate(layout, cap * 2**20)
    assert result["largest_feasible_initial_bytes"] == cap * 2**20
    assert result["bytes_per_position"] == 131072
    tiny = dense_capacity_certificate(layout, 150000)
    assert tiny["largest_feasible_initial_bytes"] == 131072
    assert tiny["target_shortfall_bytes"] == 18928
    with pytest.raises(ValueError, match="mandatory"):
        dense_capacity_certificate(layout, 100)
    wrong = [dict(layout[0], key_element_size=4)]
    with pytest.raises(ValueError, match="FP16"):
        dense_capacity_certificate(wrong, 2**20)


def row_fixture():
    sample = dict(sample_id="one", references=["a caption"] * 5)
    row = dict(
        sample_id="one",
        answer="a caption",
        reference=json.dumps(sample["references"]),
        status="completed",
        error=None,
        generated_tokens=64,
        prefill_time=1.0,
        ttft=2.0,
        compression_time=1.0,
        decode_time=3.0,
        end_to_end_time=5.0,
        repair_count=0,
        residual_kv_bytes=0,
        **dict.fromkeys(STORAGE_FIELDS, 1024),
    )
    return sample, row


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("answer", None),
        ("reference", '["wrong"]'),
        ("generated_tokens", 63),
        ("ttft", float("nan")),
        ("prefill_time", None),
        ("repair_count", 1),
        ("initial_active_kv_storage_bytes", None),
    ],
)
def test_caption_gate_preserves_failure_denominator(field, value):
    sample, row = row_fixture()
    validate_caption_rows([row], [sample])
    row[field] = value
    with pytest.raises(ValueError):
        validate_caption_rows([row], [sample])


def test_empty_generated_caption_is_retained_not_filtered():
    sample, row = row_fixture()
    validate_caption_rows([dict(row, answer="")], [sample])
    samples = caption_samples()
    empty = [replace(s, prediction="") for s in samples]
    assert TokenizedCider(empty)(empty) == 0.0


def test_bf16_references_cannot_match_fp16_controls():
    cfg = synthetic_smoke_config()
    processor = dict(
        chat_template="a",
        tokenizer_chat_template=None,
        image_preprocessor={},
        vocabulary={},
        special_tokens={},
    )
    sample = dict(prompt="caption", media=[dict(sha256="a" * 64)], references=["caption"] * 5)
    fp16 = replace(cfg, model=replace(cfg.model, precision=Precision.FP16))
    bf16 = replace(cfg, model=replace(cfg.model, precision=Precision.BF16))
    assert control_hashes(fp16, sample, processor, {}) != control_hashes(
        bf16, sample, processor, {}
    )


def test_resume_rejects_corruption_and_duplicate_success(tmp_path):
    worker = script("run_coco_shared_bytes")
    assert worker.verified_success(tmp_path, "plan", "cfg") is None
    artifact = tmp_path / "raw.jsonl"
    artifact.write_text("fixture")
    record = dict(
        plan_sha256="plan", config_sha="cfg", bound_files={str(artifact): worker.digest(artifact)}
    )
    worker.write_once(tmp_path / "attempt-1/success.json", record)
    assert worker.verified_success(tmp_path, "plan", "cfg") == record
    with pytest.raises(ValueError, match="provenance"):
        worker.verified_success(tmp_path, "changed", "cfg")
    worker.write_once(tmp_path / "attempt-2/success.json", record)
    with pytest.raises(ValueError, match="duplicate"):
        worker.verified_success(tmp_path, "plan", "cfg")
    artifact.write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        worker.verified_success(tmp_path, "plan", "cfg")


def test_native_worker_preserves_milebench_routing_fields(tmp_path):
    worker = script("run_coco_shared_bytes")
    media = tmp_path / "combined.jpg"
    media.write_bytes(b"fixture")
    sample = SimpleNamespace(
        sample_id="ActionSequence-0001",
        prompt="Choose.",
        references=("second",),
        media=(
            SimpleNamespace(kind=SimpleNamespace(value="image"), path=media, sha256="a" * 64),
        ),
        milebench_dataset="ActionSequence",
        milebench_question_type=SimpleNamespace(value="multi-choice"),
        choices=("first", "second"),
    )
    payload = worker.selected_sample_payload(sample, "milebench")
    assert payload["milebench_dataset"] == "ActionSequence"
    assert payload["milebench_question_type"] == "multi-choice"
    assert payload["choices"] == ["first", "second"]
    with pytest.raises(ValueError, match="unsupported"):
        worker.selected_sample_payload(sample, "not-a-task")


def test_pilot_precision_is_estimate_not_claim():
    value = sample_size_recommendation(
        pilot_n=256, ci_low=-4, ci_high=4, target_half_width=2, seconds_per_example=10
    )
    assert value["recommended_samples"] == 1024
    assert value["estimated_four_gpu_hours"] == pytest.approx(1024 * 10 / 14400)
    huge = sample_size_recommendation(
        pilot_n=256, ci_low=-20, ci_high=20, target_half_width=2, seconds_per_example=10
    )
    assert huge["recommended_samples"] == 4096
    assert not huge["target_precision_feasible_estimate"]


def test_worker_script_resources_and_no_blind_submission():
    text = (SCRIPTS.parent / "slurm/coco_shared_bytes.sbatch").read_text()
    for control in (
        "--cpus-per-task=12",
        "--gres=gpu:a100:1",
        "--time=02:00:00",
        "--reservation=bi_fox_dgx",
    ):
        assert control in text
    manager = script("manage_coco_shared_bytes")
    assert manager.missing_indices(dict(jobs=[]), "a", "validation") == []


def test_real_plan_shape_and_single_fullkv_per_shard(tmp_path, monkeypatch):
    from argparse import Namespace

    from PIL import Image

    from mosaickv.evaluation.coco_shared_bytes import COCO_REVISION
    from mosaickv.evaluation.shared_byte_contract import load_contract

    prepare = script("prepare_native_metric_shared_bytes")
    source = SCRIPTS.parents[1]
    contract = load_contract(source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml")
    monkeypatch.setattr(prepare, "_require_clean_source", lambda *args: "fixture")
    monkeypatch.setattr(
        prepare, "candidate_content_digest", lambda *args: contract["candidate"]["content_sha256"]
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    pool = tmp_path / "pool.json"
    payload = dict(
        schema_version=1,
        task="coco_caption",
        dataset_id="lmms-lab/COCO-Caption",
        dataset_revision=COCO_REVISION,
        split="val",
        samples=[
            dict(
                sample_id=f"coco:{i}",
                prompt="Caption",
                references=["a cat"] * 5,
                media=[dict(kind="image", path=str(image), sha256=prepare.digest(image))],
            )
            for i in range(288)
        ],
        bound_files={str(image): prepare.digest(image)},
        historical_prior_use_disclosure="synthetic_fixture",
        processor_signature=dict(fixture=True),
    )
    prepare.write_once(pool, payload)
    root = tmp_path / "new-plan"
    prepare.prepare(
        Namespace(
            source_sha="a" * 40, root=root, task="coco_caption", ledger=[pool], exclude_ledger=[]
        )
    )
    plan = json.loads((root / "plan.json").read_text())
    small = [j for j in plan["jobs"] if j["stage"] == "validation"]
    large = [j for j in plan["jobs"] if j["stage"] == "larger"]
    assert len(small) == 8 and len(large) == 208
    assert len([j for j in large if j["method"] == "full_kv"]) == 16
    assert len({sid for j in small for sid in j["sample_ids"]}) == 32
    assert len({sid for j in large for sid in j["sample_ids"]}) == 256
    assert not {sid for j in small for sid in j["sample_ids"]} & {
        sid for j in large for sid in j["sample_ids"]
    }
    assert not plan["larger_execution_authorized"] and not plan["final_execution_authorized"]
    assert plan["bootstrap_resamples"] == 10000
    assert all("samples" in json.loads(Path(j["ledger"]).read_text()) for j in small + large)
    minimum_root = tmp_path / "minimum-plan"
    prepare.prepare(
        Namespace(
            source_sha="a" * 40,
            root=minimum_root,
            task="coco_caption",
            ledger=[pool],
            exclude_ledger=[],
            methods=["full_kv", "vl_cache_reimpl", "mosaickv_exact"],
            larger_caps=[40],
        )
    )
    minimum = json.loads((minimum_root / "plan.json").read_text())
    assert minimum["methods"] == ["full_kv", "vl_cache_reimpl", "mosaickv_exact"]
    assert minimum["caps_mib"] == [40]
    assert len([job for job in minimum["jobs"] if job["stage"] == "validation"]) == 6
    assert len([job for job in minimum["jobs"] if job["stage"] == "larger"]) == 48
    final = script("prepare_coco_final_curve")
    monkeypatch.setattr(final, "_require_clean_source", lambda *args: "fixture")
    parent_sha = prepare.digest(root / "plan.json")
    gate = root / "analysis/larger/report.json"
    prepare.write_once(
        gate,
        dict(
            status="passed",
            plan_sha256=parent_sha,
            missing_or_failed=[],
            derived_artifact_hashes={},
        ),
    )
    prepare.write_once(
        root / "final-preregistration.json",
        dict(plan_sha256=parent_sha, pilot_report_sha256=prepare.digest(gate), sample_count=288),
    )
    final_root = tmp_path / "final-plan"
    final.prepare(
        Namespace(
            parent=root, parent_sha=parent_sha, pool=pool, root=final_root, authorize_final=False
        )
    )
    final_plan = json.loads((final_root / "plan.json").read_text())
    assert len(final_plan["jobs"]) == 18 * 13
    assert not final_plan["final_execution_authorized"]
    assert final_plan["final_sample_count"] == 288
    assert final_plan["reuse_records"] == {}


def test_registered_pilot_reuse_is_hash_bound_and_does_not_repeat(tmp_path):
    worker = script("run_coco_shared_bytes")
    record = dict(plan_sha256="parent", config_sha="cfg", source_sha="source", bound_files={})
    path = tmp_path / "parent/attempt-1/success.json"
    worker.write_once(path, record)
    key = json.dumps(["full_kv", None, "sid"])
    plan = dict(
        _plan_sha="final",
        source_sha="source",
        reuse_records={key: dict(path=str(path), sha256=worker.digest(path), plan_sha256="parent")},
    )
    job = dict(output=str(tmp_path / "final"), method="full_kv", cap_mib=None, config_sha="cfg")
    assert worker.completed_sample(plan, job, "sid") == record
    assert worker.completed_sample(plan, job, "another") is None
    with pytest.raises(ValueError, match="source"):
        worker.completed_sample(dict(plan, source_sha="changed"), job, "sid")


@pytest.mark.parametrize("underfill", [False, True])
def test_complete_analyzer_reduces_captions_and_gates_exact_bytes(tmp_path, monkeypatch, underfill):
    from argparse import Namespace

    import pandas as pd

    from mosaickv.evaluation.shared_byte_contract import CONTROL_FIELDS

    analyzer = script("analyze_coco_shared_bytes")
    layout = [
        dict(
            key_shape=[1, 8, 1000, 128],
            value_shape=[1, 8, 1000, 128],
            key_element_size=2,
            value_element_size=2,
        )
    ] * 32
    rows, traces, samples = {}, {}, {}
    for i in range(32):
        sid = str(i)
        sample, template = row_fixture()
        samples[sid] = dict(sample, sample_id=sid)
        for method in ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"):
            cap = None if method == "full_kv" else 40
            initial = 131072000 if cap is None else 40 * 2**20
            if underfill and method == "mosaickv_exact":
                initial -= 131072
            row = dict(
                template,
                sample_id=sid,
                method=method,
                cap_mib=cap,
                lane="vl_cache",
                task="coco_caption",
                result_category="unified_comparison",
                metric_name="cider",
                physical_byte_cap=None if cap is None else cap * 2**20,
                initial_active_kv_storage_bytes=initial,
                final_active_kv_storage_bytes=initial + 63 * 131072,
                peak_active_kv_storage_bytes=initial + 63 * 131072,
                control_hashes=dict.fromkeys(CONTROL_FIELDS, "a" * 64),
                decode_evictions=0,
                used_fullkv_fallback=False,
                config_sha="a" * 64,
            )
            rows[method, cap, sid] = row
            traces[method, cap, sid] = dict(source_kv_layout=layout)
    plan = dict(
        model_id="synthetic",
        model_revision="a" * 40,
        dataset_id="synthetic",
        dataset_revision="b" * 40,
        output_tokens=64,
        source_sha="a" * 40,
        prior_use_disclosures=["synthetic_fixture"],
        bootstrap_resamples=2,
        bootstrap_seed=1,
    )
    path = tmp_path / "plan.json"
    analyzer.write_once(path, plan)
    monkeypatch.setattr(analyzer, "collect", lambda *args: (rows, traces, samples, []))
    monkeypatch.setattr(analyzer, "TokenizedCider", lambda samples: lambda draw: float(len(draw)))
    args = Namespace(root=tmp_path, stage="validation", plan_sha=analyzer.digest(path), output=None)
    assert analyzer.analyze(args) == int(underfill)
    gate = tmp_path / "analysis/validation/report.json"
    assert gate.exists() is not underfill
    if not underfill:
        report = json.loads(gate.read_text())
        assert len(report["aggregates"]) == 4
        assert all(a["metric_value"] == 32 for a in report["aggregates"])
        full = next(a for a in report["aggregates"] if a["method"] == "full_kv")
        assert full["end_to_end_time"] is None and full["decode_time"] is None
        latency = [r for r in report["comparisons"] if r["metric_name"] == "end_to_end_time"]
        assert len(latency) == 2 and all(r["baseline"] != "full_kv" for r in latency)
        table = next(
            p for p in report["derived_artifact_hashes"] if p.endswith("corpus_metrics.parquet")
        )
        assert len(pd.read_parquet(table)) == 4
        import xml.etree.ElementTree as ET

        plots = [Path(p) for p in report["derived_artifact_hashes"] if p.endswith(".svg")]
        assert len(plots) == 2
        for plot in plots:
            ET.parse(plot)
            if "latency" in plot.name:
                assert "FullKV" not in plot.read_text()
    else:
        # A failed attempt must not poison the later canonical report path.
        assert len(list(gate.parent.glob("attempt-*/report.json"))) == 1
        assert analyzer.analyze(args) == 1
        assert len(list(gate.parent.glob("attempt-*/report.json"))) == 2
