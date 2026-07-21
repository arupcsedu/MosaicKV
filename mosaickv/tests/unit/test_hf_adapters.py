from __future__ import annotations

import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CachedKeyState,
    CacheSnapshot,
    HuggingFaceMultimodalAdapter,
    InternVL25Adapter,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Modality,
    QueryVectorState,
    Qwen25VLAdapter,
    runtime_adapter_class,
)
from mosaickv.adapters.huggingface.registry import _resolve_model_source


class _FakeTensor:
    def __init__(self, value: Any, *, device: str = "cpu") -> None:
        self.array = np.asarray(value)
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.array.dtype

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(self.array[key], device=self.device)

    def detach(self) -> _FakeTensor:
        return self

    def clone(self) -> _FakeTensor:
        return _FakeTensor(self.array.copy(), device=self.device)

    def cpu(self) -> _FakeTensor:
        return _FakeTensor(self.array, device="cpu")

    def to(self, device: Any = None, *, dtype: Any = None) -> _FakeTensor:
        array = self.array.astype(dtype) if dtype is not None else self.array
        return _FakeTensor(array, device=str(device or self.device))

    def tolist(self) -> Any:
        return self.array.tolist()


class _FakeCache:
    def __init__(self, layers: Sequence[tuple[_FakeTensor, _FakeTensor]]) -> None:
        self._layers = tuple(layers)

    def to_legacy_cache(self) -> tuple[tuple[_FakeTensor, _FakeTensor], ...]:
        return self._layers

    @classmethod
    def from_legacy_cache(cls, layers: Sequence[tuple[_FakeTensor, _FakeTensor]]) -> _FakeCache:
        return cls(layers)


class _FakeConfig:
    image_token_id = 99

    def __init__(self, attention: str = "eager") -> None:
        self._attn_implementation = attention


class _FakeParameter:
    device = "cpu"


class _FakeModel:
    dtype = np.float32
    training = True

    def __init__(self, attention: str = "eager") -> None:
        self.config = _FakeConfig(attention)
        self.eval_called = False

    def parameters(self) -> Iterator[_FakeParameter]:
        yield _FakeParameter()

    def eval(self) -> None:
        self.eval_called = True
        self.training = False


class _FakeProcessor:
    chat_template = "fake"

    def __init__(self) -> None:
        self.call: dict[str, Any] = {}

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        return "<image> describe"

    def __call__(self, **kwargs: Any) -> dict[str, _FakeTensor]:
        self.call = kwargs
        return {
            "input_ids": _FakeTensor([[1, 99, 99, 2]]),
            "attention_mask": _FakeTensor([[1, 1, 1, 1]]),
        }


class _FakeAdapter(HuggingFaceMultimodalAdapter):
    capabilities = AdapterCapabilities(
        model_family="fake",
        architectures=("_FakeModel",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=False,
        cache_classes=("_FakeCache",),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
    )

    def _language_layers(self) -> Sequence[Any]:
        return ()

    def _image_token_id(self) -> int | None:
        return 99

    def _processor_kwargs(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> dict[str, Any]:
        assert not videos
        return {"text": prompt, "images": list(images)}


def test_runtime_registry_is_fail_closed() -> None:
    assert runtime_adapter_class("llava-hf/llava-1.5-7b-hf") is Llava15Adapter
    assert runtime_adapter_class("Qwen/Qwen2.5-VL-3B-Instruct") is Qwen25VLAdapter
    assert (
        runtime_adapter_class("llava-hf/llava-onevision-qwen2-0.5b-ov-hf") is LlavaOneVisionAdapter
    )
    assert runtime_adapter_class("OpenGVLab/InternVL2_5-4B") is InternVL25Adapter
    with pytest.raises(LookupError, match="no runtime HF adapter"):
        runtime_adapter_class("unreviewed/model")


def test_adapter_capability_metadata_records_rope_and_disabled_features() -> None:
    for adapter_type in (Llava15Adapter, Qwen25VLAdapter, LlavaOneVisionAdapter, InternVL25Adapter):
        capabilities = adapter_type.capabilities
        assert capabilities.attention_implementations == ("eager",)
        assert capabilities.cached_key_state is CachedKeyState.POST_ROPE
        assert capabilities.query_vector_state is QueryVectorState.Q_PROJ_PRE_ROPE
        assert capabilities.supports_prototype_merge is False
        assert capabilities.supports_residual_repair is False


def test_prepare_inputs_maps_expanded_image_tokens_without_torch() -> None:
    model = _FakeModel()
    processor = _FakeProcessor()
    payload = object()
    adapter = _FakeAdapter(model, processor)
    prepared = adapter.prepare_inputs(
        [{"role": "user", "content": "describe this"}], media=[payload]
    )
    assert model.eval_called
    assert processor.call["images"] == [payload]
    assert prepared.logical_sequence_length == 4
    assert [(span.start, span.end, span.modality) for span in prepared.modality_map] == [
        (0, 1, Modality.TEXT),
        (1, 3, Modality.IMAGE),
        (3, 4, Modality.TEXT),
    ]


def test_non_eager_model_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="reload with attn_implementation='eager'"):
        _FakeAdapter(_FakeModel("sdpa"), _FakeProcessor())


def test_runtime_loader_requires_immutable_revision_before_import_or_download() -> None:
    with pytest.raises(ValueError, match="40-character commit SHA"):
        Llava15Adapter.from_pretrained("llava-hf/llava-1.5-7b-hf", revision="main")


def test_offline_hf_loader_resolves_the_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    result = _resolve_model_source("org/model", "a" * 40, local_files_only=True)

    assert result == str(snapshot.resolve())
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "a" * 40,
            "cache_dir": str(tmp_path / "hub"),
            "local_files_only": True,
        }
    ]


def test_cache_snapshot_layout_and_type_preserving_reinjection() -> None:
    adapter = _FakeAdapter(_FakeModel(), _FakeProcessor())
    key = _FakeTensor(np.arange(24).reshape(1, 2, 3, 4))
    value = _FakeTensor(np.arange(24).reshape(1, 2, 3, 4))
    cache = _FakeCache(((key, value),))
    snapshot = adapter.extract_past_key_values(cache)
    assert isinstance(snapshot, CacheSnapshot)
    assert snapshot.source_class is _FakeCache
    assert snapshot.active_sequence_length == 3
    assert snapshot.layers[0].sequence_dimension == 2

    layout = adapter.get_cache_layout(snapshot)
    assert layout.cache_class.endswith("._FakeCache")
    assert layout.layers[0].key_shape == (1, 2, 3, 4)
    reinjected = adapter.inject_past_key_values(snapshot)
    assert isinstance(reinjected, _FakeCache)
    reinjected_key = reinjected.to_legacy_cache()[0][0]
    assert np.array_equal(reinjected_key.array, key.array)


def test_all_required_runtime_methods_are_present() -> None:
    required = {
        "prepare_inputs",
        "prefill",
        "decode_one_token",
        "extract_past_key_values",
        "inject_past_key_values",
        "capture_query_vectors",
        "get_modality_map",
        "get_logical_sequence_length",
        "get_cache_layout",
        "supports_prototype_merge",
        "supports_residual_repair",
    }
    for adapter_type in (Llava15Adapter, Qwen25VLAdapter, LlavaOneVisionAdapter, InternVL25Adapter):
        assert required.issubset(set(dir(adapter_type)))
