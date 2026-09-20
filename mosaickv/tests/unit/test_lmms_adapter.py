from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import lmms_adapter
from mosaickv.evaluation.lmms_adapter import LmmsRequestBridge
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.evaluation.tasks import default_task_registry


class InspectingModel:
    model_id = "local/model"
    backend = "huggingface"
    method = "full_cache"
    retention_ratio = 1.0
    supports_video = False

    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        self.requests.append(request)
        return ModelGeneration("A", GenerationMetrics(active_kv_bytes=128))


class FailingModel(InspectingModel):
    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        raise RuntimeError("generation exploded")


@dataclass
class FakeInstance:
    args: tuple[object, ...]


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __getitem__(self, key: object) -> object:
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[int(key)]

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices: list[int]) -> FakeDataset:
        return FakeDataset([self.rows[index].copy() for index in indices])

    def add_column(self, name: str, values: list[str]) -> FakeDataset:
        rows = [row.copy() for row in self.rows]
        for row, value in zip(rows, values, strict=True):
            row[name] = value
        return FakeDataset(rows)


def test_lmms_request_contract_builds_standard_message_and_captures_metrics() -> None:
    model = InspectingModel()
    bridge = LmmsRequestBridge(model, "run-1")
    dataset: dict[str, object] = {"mmstar": {"test": [{"_mosaickv_sample_id": "mmstar:q1"}]}}
    request = FakeInstance(
        (
            "Choose the answer",
            {"max_new_tokens": 4},
            lambda _doc: [object()],
            0,
            "mmstar",
            "test",
        )
    )
    assert bridge.generate_until([request], dataset) == ["A"]
    assert model.requests[0].sample_id == "mmstar:q1"
    assert [part.type for part in model.requests[0].messages[0].content] == ["image", "text"]
    assert bridge.captures[0].generation is not None
    assert bridge.captures[0].generation.metrics.active_kv_bytes == 128


def test_lmms_bridge_returns_empty_response_but_retains_failure() -> None:
    model = FailingModel()
    bridge = LmmsRequestBridge(model, "run-1")
    dataset: dict[str, object] = {"mmstar": {"test": [{}]}}
    request = FakeInstance(("Question", {}, lambda _doc: [], 0, "mmstar", "test"))
    assert bridge.generate_until([request], dataset) == [""]
    assert "generation exploded" in (bridge.captures[0].error or "")


def test_factory_returns_an_lmms_subclass_without_global_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLmmsBase:
        def __init__(self) -> None:
            self.task_dict: dict[str, object] = {}

    monkeypatch.setattr(lmms_adapter, "_require_lmms_version", lambda: None)
    monkeypatch.setattr(
        lmms_adapter,
        "_require_module",
        lambda _name: SimpleNamespace(lmms=FakeLmmsBase),
    )
    adapter = lmms_adapter.create_lmms_model_adapter(InspectingModel(), run_id="run-1")
    assert isinstance(adapter, FakeLmmsBase)
    assert hasattr(adapter, "mosaickv_bridge")


def test_dataset_revision_scope_injects_exact_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def load_dataset(path: object, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return object()

    datasets_module = SimpleNamespace(load_dataset=load_dataset)
    monkeypatch.setattr(lmms_adapter, "_require_module", lambda _name: datasets_module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    revision = "a" * 40

    with lmms_adapter._pinned_dataset_revision("owner/dataset", revision):
        datasets_module.load_dataset("owner/dataset", split="test", token=True)
        datasets_module.load_dataset("owner/other", split="test")

    assert calls[0][1]["revision"] == revision
    assert calls[0][1]["token"] is False
    assert "revision" not in calls[1][1]


def test_dataset_revision_scope_rejects_wrong_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_module = SimpleNamespace(load_dataset=lambda *_args, **_kwargs: object())
    monkeypatch.setattr(lmms_adapter, "_require_module", lambda _name: datasets_module)

    with (
        lmms_adapter._pinned_dataset_revision(
            "owner/dataset", "a" * 40, dataset_config="InfographicVQA"
        ),
        pytest.raises(ValueError, match="configuration"),
    ):
        datasets_module.load_dataset("owner/dataset", "DocumentVQA", split="validation")


@pytest.mark.parametrize(
    ("family", "template"),
    (
        ("mmstar", "_default_template_yaml"),
        ("textvqa", "_default_template_textvqa_yaml"),
        ("docvqa", "_default_template_docvqa_yaml"),
        ("infovqa", "_default_template_infovqa_yaml"),
        ("videomme", "_default_template_yaml"),
    ),
)
def test_missing_lmms_templates_use_packaged_compatibility_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str,
    template: str,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_yaml_config(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    lmms_utils = SimpleNamespace(load_yaml_config=load_yaml_config)
    monkeypatch.setattr(lmms_adapter, "_require_module", lambda _name: lmms_utils)
    requested = (
        tmp_path / family / "reasoning" / ".." / template
        if family == "docvqa"
        else tmp_path / family / template
    )

    with lmms_adapter._lmms_template_compatibility():
        lmms_utils.load_yaml_config(yaml_path=str(requested), yaml_dir=str(requested.parent))

    replacement = Path(str(calls[0][1]["yaml_path"]))
    assert replacement.is_file()
    assert replacement.name == template
    assert replacement.parent.name == family
    assert calls[0][1]["yaml_dir"] == str(requested.parent)
    assert f"lmms_eval.tasks.{family}.utils" in replacement.read_text(encoding="utf-8")


def test_loaded_task_route_supports_explicit_train_override_without_mutating_config() -> None:
    spec = default_task_registry().resolve("textvqa_train_calibration")
    task = SimpleNamespace(
        config=SimpleNamespace(
            dataset_path="lmms-lab/textvqa",
            dataset_name=None,
            test_split="validation",
            validation_split=None,
        ),
        dataset={"train": object(), "validation": object()},
    )

    assert lmms_adapter._loaded_task_split(task, spec) == "train"
    assert task.config.test_split == "validation"


def test_prepare_task_activates_explicit_train_override_for_lmms_evaluator() -> None:
    spec = default_task_registry().resolve("textvqa_train_calibration")
    task = SimpleNamespace(
        config=SimpleNamespace(
            dataset_path="lmms-lab/textvqa",
            dataset_name=None,
            test_split="validation",
            validation_split=None,
        ),
        dataset={
            "train": FakeDataset(
                [
                    {"question_id": 1, "answers": ["one"]},
                    {"question_id": 2, "answers": ["two"]},
                ]
            ),
            "validation": FakeDataset([{"question_id": 9, "answers": ["nine"]}]),
        },
        doc_to_target=lambda doc: doc["answers"],
    )

    selected_task, prepared, selected_ids = lmms_adapter._prepare_task(
        task,
        spec=spec,
        seed=17,
        subset_size=None,
        sample_offset=0,
        explicit_sample_ids=("textvqa_train_calibration:2",),
        excluded=frozenset(),
        prepare_samples=True,
    )

    assert selected_task is task
    assert task.config.test_split == "train"
    assert len(task.dataset["train"]) == 1
    assert selected_ids == ("textvqa_train_calibration:2",)
    assert prepared[0].sample_id == "textvqa_train_calibration:2"
    assert prepared[0].reference == '["two"]'


def test_textvqa_reference_provenance_uses_annotations_not_template_field_name() -> None:
    spec = default_task_registry().resolve("textvqa")
    task = SimpleNamespace(doc_to_target=lambda _doc: "answer")

    assert (
        lmms_adapter._prepared_reference(
            task,
            spec,
            {"question_id": 7, "answers": ["two", "2"]},
        )
        == '["two","2"]'
    )

    with pytest.raises(ValueError, match="requires reference field 'answers'"):
        lmms_adapter._prepared_reference(task, spec, {"question_id": 7})


def test_realworldqa_route_and_reference_are_explicit_and_cpu_only() -> None:
    spec = default_task_registry().resolve("realworldqa")
    task = SimpleNamespace(
        config=SimpleNamespace(
            dataset_path="lmms-lab/RealWorldQA",
            dataset_name=None,
            test_split="test",
            validation_split=None,
        ),
        dataset={"test": object()},
        doc_to_target=lambda _doc: "template-placeholder",
    )

    assert lmms_adapter._loaded_task_split(task, spec) == "test"
    assert (
        lmms_adapter._prepared_reference(
            task,
            spec,
            {"question_id": 17, "answer": "Downhill"},
        )
        == "Downhill"
    )
    with pytest.raises(ValueError, match="requires reference field 'answer'"):
        lmms_adapter._prepared_reference(task, spec, {"question_id": 17})


def test_loaded_task_route_rejects_configuration_or_implicit_split_mismatch() -> None:
    registry = default_task_registry()
    info = registry.resolve("infovqa")
    wrong_config = SimpleNamespace(
        config=SimpleNamespace(
            dataset_path=info.dataset_id,
            dataset_name="DocumentVQA",
            test_split="validation",
            validation_split=None,
        ),
        dataset={"validation": object()},
    )
    with pytest.raises(ValueError, match="configuration"):
        lmms_adapter._loaded_task_split(wrong_config, info)

    text = registry.resolve("textvqa")
    wrong_split = SimpleNamespace(
        config=SimpleNamespace(
            dataset_path=text.dataset_id,
            dataset_name=None,
            test_split="test",
            validation_split=None,
        ),
        dataset={"test": object(), "validation": object()},
    )
    with pytest.raises(ValueError, match="explicit split override"):
        lmms_adapter._loaded_task_split(wrong_split, text)


def test_lmms_auxiliary_outputs_are_routed_beside_raw_artifact(tmp_path: Path) -> None:
    raw_output = tmp_path / "run" / "results.jsonl"

    args = lmms_adapter._lmms_cli_args(raw_output)

    assert args.output_path == str(raw_output.parent.resolve())
    assert args.process_with_media is True
    assert not Path(args.output_path).is_relative_to(Path.cwd())


def test_paper_native_nested_and_corpus_only_scores_are_distinguished() -> None:
    assert (
        lmms_adapter._extract_optional_score(
            {"llm_as_judge_eval": {"true_false": True}}, "llm_as_judge_eval"
        )
        == 1.0
    )
    assert (
        lmms_adapter._extract_optional_score(
            {"coco_CIDEr": {"answer": ["caption"], "pred": "caption"}}, "coco_CIDEr"
        )
        is None
    )


def test_mathvista_score_matches_pinned_lmms_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # lmms-eval 0.7.2 creates its judge provider at import time even when its
    # deterministic quick extractor is sufficient.  The dummy provider keeps
    # this source-parity fixture offline; production runs retain their explicit
    # API_TYPE/provider configuration.
    monkeypatch.setenv("API_TYPE", "dummy")
    from lmms_eval.tasks.mathvista.utils import mathvista_process_results

    doc = {
        "question_type": "free_form",
        "answer_type": "integer",
        "query": "What is one plus one?",
        "choices": [],
        "answer": "2",
        "precision": 0,
        "pid": "fixture",
        "metadata": {"split": "testmini"},
    }
    result = mathvista_process_results(doc, ["2"])

    assert result["llm_as_judge_eval"]["true_false"] is True
    assert lmms_adapter._extract_optional_score(result, "llm_as_judge_eval") == 1.0


def test_split_metric_parquet_outputs_must_be_configured_as_a_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        lmms_adapter.run_lmms_development_evaluation(
            run_id="run",
            task_names=("mmstar",),
            model=InspectingModel(),
            raw_output=tmp_path / "raw.jsonl",
            manifest_path="manifest.json",
            seed=0,
            subset_size=1,
            metric_observations_parquet_output=tmp_path / "observations.parquet",
        )
