from __future__ import annotations

from pathlib import Path

from mosaickv.evaluation.harness import EvaluationHarness
from mosaickv.evaluation.model import EvaluationRequest, ModelGeneration
from mosaickv.evaluation.storage import load_jsonl
from mosaickv.evaluation.synthetic import SyntheticColorModel
from mosaickv.evaluation.tasks import load_synthetic_samples


class CountingModel(SyntheticColorModel):
    def __init__(self, *, fail_sample: str | None = None) -> None:
        self.calls = 0
        self.fail_sample = fail_sample

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        self.calls += 1
        if request.sample_id == self.fail_sample:
            raise RuntimeError("intentional CI failure")
        return super().generate(request)


class EffectiveMethodModel(SyntheticColorModel):
    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        generated = super().generate(request)
        return ModelGeneration(
            generated.answer,
            generated.metrics,
            effective_method="mosaickv_full__mosaickv_exact_safety_fallback",
        )


def test_synthetic_task_completes_and_resumes_on_cpu(tmp_path: Path) -> None:
    model = CountingModel()
    raw = tmp_path / "raw.jsonl"
    harness = EvaluationHarness()
    first = harness.run(
        run_id="cpu-ci",
        task_name="synthetic_ci",
        samples=load_synthetic_samples(),
        model=model,
        raw_output=raw,
        manifest_path=str(tmp_path / "manifest.json"),
        seed=7,
        subset_size=3,
    )
    assert first.completed_samples == 3
    assert first.failed_samples == 0
    assert model.calls == 3
    second = harness.run(
        run_id="cpu-ci",
        task_name="synthetic_ci",
        samples=load_synthetic_samples(),
        model=model,
        raw_output=raw,
        manifest_path=str(tmp_path / "manifest.json"),
        seed=7,
        subset_size=3,
    )
    assert second.resumed_samples == 3
    assert model.calls == 3
    assert len(load_jsonl(raw)) == 3


def test_failed_sample_is_a_terminal_raw_row(tmp_path: Path) -> None:
    samples = load_synthetic_samples()
    failed_id = samples[0].sample_id
    raw = tmp_path / "raw.jsonl"
    summary = EvaluationHarness().run(
        run_id="failure-ci",
        task_name="synthetic_ci",
        samples=samples,
        model=CountingModel(fail_sample=failed_id),
        raw_output=raw,
        manifest_path=str(tmp_path / "manifest.json"),
        seed=0,
    )
    rows = {row.sample_id: row for row in load_jsonl(raw)}
    assert summary.failed_samples == 1
    assert len(rows) == len(samples)
    assert rows[failed_id].status.value == "failed"
    assert rows[failed_id].task_score is None
    assert "intentional CI failure" in (rows[failed_id].error or "")


def test_synthetic_scores_are_deterministic(tmp_path: Path) -> None:
    samples = load_synthetic_samples()
    scores: list[list[float | None]] = []
    for suffix in ("a", "b"):
        raw = tmp_path / f"{suffix}.jsonl"
        EvaluationHarness().run(
            run_id=f"run-{suffix}",
            task_name="synthetic_ci",
            samples=samples,
            model=SyntheticColorModel(),
            raw_output=raw,
            manifest_path=str(tmp_path / f"{suffix}.manifest.json"),
            seed=19,
            subset_size=2,
        )
        scores.append([row.task_score for row in load_jsonl(raw)])
    assert scores[0] == scores[1] == [1.0, 1.0]


def test_result_records_effective_method_instead_of_requested_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    EvaluationHarness().run(
        run_id="effective-method-ci",
        task_name="synthetic_ci",
        samples=load_synthetic_samples(),
        model=EffectiveMethodModel(),
        raw_output=raw,
        manifest_path=str(tmp_path / "manifest.json"),
        seed=0,
        subset_size=1,
    )

    assert load_jsonl(raw)[0].method == "mosaickv_full__mosaickv_exact_safety_fallback"
