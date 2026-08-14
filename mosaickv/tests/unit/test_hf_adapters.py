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
    AttentionCapture,
    CachedKeyState,
    CacheSnapshot,
    DecodeState,
    HuggingFaceMultimodalAdapter,
    InternVL25Adapter,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Modality,
    ModalitySpan,
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


def test_attention_capture_reduces_query_groups_during_prefill() -> None:
    torch = pytest.importorskip("torch")

    class _Attention(torch.nn.Module):
        def forward(self, value: Any) -> tuple[Any, Any]:
            probabilities = torch.arange(1 * 4 * 3 * 5, dtype=torch.float32).reshape(
                1, 4, 3, 5
            )
            return value, probabilities

    attention = _Attention()
    with AttentionCapture((attention,), query_window=1, kv_heads=(2,)) as capture:
        attention(torch.zeros(1))
    observed = capture.result()[0]
    source = torch.arange(1 * 4 * 3 * 5, dtype=torch.float32).reshape(1, 4, 3, 5)
    expected = source[..., -1:, :].reshape(1, 2, 2, 1, 5).mean(dim=2)

    assert observed.shape == (1, 2, 1, 5)
    assert torch.equal(observed, expected)


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
    assert runtime_adapter_class("llava-hf/llava-onevision-qwen2-7b-ov-hf") is LlavaOneVisionAdapter
    assert runtime_adapter_class("OpenGVLab/InternVL2_5-4B") is InternVL25Adapter
    with pytest.raises(LookupError, match="no runtime HF adapter"):
        runtime_adapter_class("unreviewed/model")


def test_qwen_video_paths_use_pinned_qwen_vision_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, Any], bool]] = []

    def fetch_video(
        value: dict[str, Any], *, return_video_sample_fps: bool
    ) -> tuple[np.ndarray[Any, Any], float]:
        calls.append((value, return_video_sample_fps))
        return np.zeros((40, 3, 4, 4), dtype=np.float32), 2.0

    fake_qwen_utils = types.ModuleType("qwen_vl_utils")
    fake_qwen_utils.fetch_video = fetch_video  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)

    kwargs = Qwen25VLAdapter._processor_kwargs(
        types.SimpleNamespace(
            _MIN_PIXELS=Qwen25VLAdapter._MIN_PIXELS,
            _MAX_PIXELS=Qwen25VLAdapter._MAX_PIXELS,
            _MAX_VIDEO_FRAMES=Qwen25VLAdapter._MAX_VIDEO_FRAMES,
            _VIDEO_FRAME_PIXELS=Qwen25VLAdapter._VIDEO_FRAME_PIXELS,
        ),
        "prompt",
        (),
        (Path("/scratch/djy8hg/cache/mosaickv/video.mp4"),),
    )

    assert calls == [
        (
            {
                "type": "video",
                "video": "/scratch/djy8hg/cache/mosaickv/video.mp4",
                "min_pixels": Qwen25VLAdapter._VIDEO_FRAME_PIXELS,
                "max_pixels": Qwen25VLAdapter._VIDEO_FRAME_PIXELS,
            },
            True,
        )
    ]
    assert len(kwargs["videos"]) == 1
    assert kwargs["videos"][0].shape == (32, 3, 4, 4)
    assert {"fps", "min_pixels", "max_pixels"}.isdisjoint(kwargs)


def test_adapter_capability_metadata_records_rope_and_safely_gated_features() -> None:
    for adapter_type in (Llava15Adapter, Qwen25VLAdapter, LlavaOneVisionAdapter, InternVL25Adapter):
        capabilities = adapter_type.capabilities
        if adapter_type is InternVL25Adapter:
            assert capabilities.attention_implementations == ("eager",)
        else:
            assert capabilities.attention_implementations == (
                "eager",
                "sdpa",
                "flash_attention_2",
            )
        assert capabilities.cached_key_state is CachedKeyState.POST_ROPE
        assert capabilities.query_vector_state is QueryVectorState.Q_PROJ_PRE_ROPE
        expected_phase_transform_support = adapter_type is not InternVL25Adapter
        assert capabilities.supports_prototype_merge is expected_phase_transform_support
        assert capabilities.supports_residual_repair is expected_phase_transform_support


def test_qwen_visual_grids_produce_exact_spatial_and_frame_spans() -> None:
    spans = (
        ModalitySpan(0, 1, Modality.TEXT),
        ModalitySpan(1, 5, Modality.IMAGE),
        ModalitySpan(5, 6, Modality.TEXT),
        ModalitySpan(6, 14, Modality.VIDEO),
        ModalitySpan(14, 15, Modality.TEXT),
    )

    enriched = Qwen25VLAdapter.enrich_modality_spans(
        spans,
        image_grid_thw=((1, 4, 4),),
        video_grid_thw=((2, 4, 4),),
        spatial_merge_size=2,
    )

    assert enriched == (
        spans[0],
        ModalitySpan(1, 5, Modality.IMAGE, image_index=0, grid_shape=(2, 2)),
        spans[2],
        ModalitySpan(
            6,
            10,
            Modality.VIDEO,
            image_index=0,
            frame_index=0,
            grid_shape=(2, 2),
        ),
        ModalitySpan(
            10,
            14,
            Modality.VIDEO,
            image_index=0,
            frame_index=1,
            grid_shape=(2, 2),
        ),
        spans[4],
    )


def test_qwen_visual_grid_mismatch_fails_instead_of_guessing_metadata() -> None:
    with pytest.raises(RuntimeError, match="requires 4"):
        Qwen25VLAdapter.enrich_modality_spans(
            (ModalitySpan(0, 3, Modality.IMAGE),),
            image_grid_thw=((1, 4, 4),),
            video_grid_thw=None,
            spatial_merge_size=2,
        )


def test_llava_patch_runs_gain_authoritative_spatial_metadata() -> None:
    spans = (
        ModalitySpan(0, 1, Modality.TEXT),
        ModalitySpan(1, 9, Modality.IMAGE),
        ModalitySpan(9, 10, Modality.TEXT),
    )

    enriched = Llava15Adapter._enrich_image_grid_spans(spans, rows=2, columns=2)

    assert enriched == (
        spans[0],
        ModalitySpan(1, 5, Modality.IMAGE, image_index=0, grid_shape=(2, 2)),
        ModalitySpan(5, 9, Modality.IMAGE, image_index=1, grid_shape=(2, 2)),
        spans[2],
    )
    assert Llava15Adapter._enrich_image_grid_spans(
        (ModalitySpan(0, 3, Modality.IMAGE),), rows=2, columns=2
    ) == (ModalitySpan(0, 3, Modality.IMAGE),)


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
    with pytest.raises(ValueError, match="does not support attention implementation"):
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


def test_decode_metadata_uses_amortized_buffers() -> None:
    torch = pytest.importorskip("torch")
    state = DecodeState(
        past_key_values=object(),
        attention_mask=torch.tensor([[1, 0, 1]], dtype=torch.long),
        active_cache_length=3,
        logical_sequence_length=5,
        next_decode_position=5,
        modality_map=(),
    )

    first_mask = _FakeAdapter._append_decode_attention_mask(state)
    mask_buffer = state.model_state["mosaickv_decode_attention_mask_buffer"]
    assert first_mask.tolist() == [[1, 0, 1, 1]]
    state.attention_mask = first_mask
    second_mask = _FakeAdapter._append_decode_attention_mask(state)
    assert second_mask.tolist() == [[1, 0, 1, 1, 1]]
    assert state.model_state["mosaickv_decode_attention_mask_buffer"] is mask_buffer

    first_position = _FakeAdapter._decode_cache_position(state, torch.device("cpu"))
    position_buffer = state.model_state["mosaickv_decode_position_buffer"]
    assert first_position.tolist() == [5]
    state.next_decode_position = 6
    second_position = _FakeAdapter._decode_cache_position(state, torch.device("cpu"))
    assert second_position.tolist() == [6]
    assert state.model_state["mosaickv_decode_position_buffer"] is position_buffer


def test_qwen_decode_reuses_multimodal_position_buffer() -> None:
    torch = pytest.importorskip("torch")

    class _Output:
        logits = torch.zeros((1, 1, 4))

    def model_call(self: Any, **kwargs: Any) -> _Output:
        self.call = kwargs
        return _Output()

    model_type = type(
        "Qwen2_5_VLForConditionalGeneration",
        (_FakeModel,),
        {"__call__": model_call},
    )
    model = model_type()
    adapter = Qwen25VLAdapter(model, _FakeProcessor())
    state = DecodeState(
        past_key_values=object(),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        active_cache_length=5,
        logical_sequence_length=5,
        next_decode_position=5,
        modality_map=(),
        model_state={
            "rope_deltas": torch.tensor([[7]], dtype=torch.long),
            "mosaickv_cache_all_valid": True,
        },
    )
    adapter._decode_forward(
        torch.tensor([[1]]),
        state,
        None,
        torch.tensor([5]),
    )
    first_buffer = state.model_state["mosaickv_qwen_decode_position_ids_buffer"]
    assert model.call["position_ids"].reshape(-1).tolist() == [12, 12, 12]
    assert model.call["attention_mask"] == {
        "full_attention": None,
        "sliding_attention": None,
    }
    state.next_decode_position = 6
    adapter._decode_forward(
        torch.tensor([[2]]),
        state,
        None,
        torch.tensor([6]),
    )
    assert state.model_state["mosaickv_qwen_decode_position_ids_buffer"] is first_buffer
    assert model.call["position_ids"].reshape(-1).tolist() == [13, 13, 13]
