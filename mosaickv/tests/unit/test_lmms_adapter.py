from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import lmms_adapter
from mosaickv.evaluation.lmms_adapter import LmmsRequestBridge
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration


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


@pytest.mark.parametrize(
    ("family", "template"),
    (
        ("mmstar", "_default_template_yaml"),
        ("textvqa", "_default_template_textvqa_yaml"),
        ("docvqa", "_default_template_docvqa_yaml"),
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


def test_lmms_auxiliary_outputs_are_routed_beside_raw_artifact(tmp_path: Path) -> None:
    raw_output = tmp_path / "run" / "results.jsonl"

    args = lmms_adapter._lmms_cli_args(raw_output)

    assert args.output_path == str(raw_output.parent.resolve())
    assert args.process_with_media is True
    assert not Path(args.output_path).is_relative_to(Path.cwd())
