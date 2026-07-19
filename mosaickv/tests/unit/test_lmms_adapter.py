from __future__ import annotations

from dataclasses import dataclass
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
