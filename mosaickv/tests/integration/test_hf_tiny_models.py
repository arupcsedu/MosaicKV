from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mosaickv.adapters.huggingface import (  # noqa: E402
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Qwen25VLAdapter,
    compare_cache_reinjection,
    compare_mosaickv_retention_one,
    compare_with_generate,
)
from mosaickv.backends import (  # noqa: E402
    HuggingFaceMosaicKVModel,
    compare_runtime_retention_one,
)
from mosaickv.config import (  # noqa: E402
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    LookMConfig,
    ModelConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    VLCacheConfig,
)
from mosaickv.evaluation.messages import build_multimodal_messages  # noqa: E402
from mosaickv.evaluation.model import EvaluationRequest  # noqa: E402
from mosaickv.evaluation.oracle_queries import (  # noqa: E402
    collect_evaluation_only_true_future_queries,
)
from mosaickv.forecasting import forecast_from_prefill  # noqa: E402
from mosaickv.types import (  # noqa: E402
    Backend,
    BudgetUnit,
    ForecastMode,
    LookMMergeStrategy,
    MosaicKVMethod,
    Precision,
    RepairPolicy,
)


class _TextOnlyProcessor:
    chat_template = "unit-test-template"

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        return "tiny prompt"

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    def batch_decode(self, token_ids: Any, **_kwargs: Any) -> list[str]:
        values = token_ids.detach().cpu().reshape(-1).tolist()
        return [" ".join(str(int(value)) for value in values)]


class _TokenizerTemplate:
    chat_template = "tokenizer-unit-test-template"

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        return "tokenizer-rendered prompt"


class _VLCacheProcessor(_TextOnlyProcessor):
    """Synthetic expanded image boundary without invoking a vision tower."""

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 63, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _ProcessorWithoutTemplate(_TextOnlyProcessor):
    chat_template = None
    tokenizer = _TokenizerTemplate()

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        raise ValueError("processor has no chat template")


def _llava(*, kv_heads: int = 1) -> Llava15Adapter:
    from transformers import LlavaConfig, LlavaForConditionalGeneration

    config = LlavaConfig(
        vision_config={
            "model_type": "clip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "projection_dim": 16,
        },
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": kv_heads,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_index=63,
        attn_implementation="eager",
    )
    return Llava15Adapter(LlavaForConditionalGeneration(config).eval(), _TextOnlyProcessor())


def _qwen() -> Qwen25VLAdapter:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 4,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0],
        },
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        attn_implementation="eager",
    )
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    return Qwen25VLAdapter(model, _TextOnlyProcessor())


def test_qwen_uses_tokenizer_template_when_processor_template_is_missing() -> None:
    adapter = _qwen()
    adapter.processor = _ProcessorWithoutTemplate()

    prepared = adapter.prepare_inputs([{"role": "user", "content": "tiny prompt"}])

    assert prepared.logical_sequence_length == 4


def _onevision() -> LlavaOneVisionAdapter:
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration

    config = LlavaOnevisionConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "vision_use_head": False,
        },
        text_config={
            "model_type": "qwen2",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_index=62,
        video_token_index=63,
        image_grid_pinpoints=[[8, 8]],
        attn_implementation="eager",
    )
    model = LlavaOnevisionForConditionalGeneration(config).eval()
    return LlavaOneVisionAdapter(model, _TextOnlyProcessor())


def _runtime_config(method: MosaicKVMethod, ratio: float) -> RunConfig:
    return RunConfig(
        model=ModelConfig("tiny-random/LLaVA-1.5", "a" * 40, Precision.FP32),
        dataset=DatasetConfig("mosaickv/tiny-runtime", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 0, True),
        generation=GenerationConfig(max_new_tokens=4),
        cache=CacheConfig(8, BudgetUnit.BLOCKS, ratio, 1),
        method=method,
        forecasting=ForecastingConfig(
            enabled=method is not MosaicKVMethod.FULLKV,
            mode=ForecastMode.HYBRID,
            prompt_window=2,
            draft_steps=2,
            centroid_count=2,
        ),
        residual=ResidualConfig(require_pinned_memory=False),
    )


def _baseline_runtime_config(method: MosaicKVMethod, ratio: float) -> RunConfig:
    return RunConfig(
        model=ModelConfig("tiny-random/LLaVA-1.5", "a" * 40, Precision.FP32),
        dataset=DatasetConfig("mosaickv/tiny-runtime", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 17, True),
        generation=GenerationConfig(max_new_tokens=2),
        cache=CacheConfig(8, BudgetUnit.BLOCKS, ratio, 1),
        method=method,
        forecasting=ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=2,
            draft_steps=0,
            centroid_count=2,
        ),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
        repair=RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
    )


def _lookm_runtime_config(*, retention_one: bool = False) -> RunConfig:
    recent_ratio = 0.5 if retention_one else 0.25
    important_ratio = 0.5 if retention_one else 0.25
    return RunConfig(
        model=ModelConfig("tiny-random/LLaVA-1.5", "a" * 40, Precision.FP32),
        dataset=DatasetConfig("mosaickv/tiny-runtime", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 17, True),
        generation=GenerationConfig(max_new_tokens=2),
        cache=CacheConfig(
            2_147_483_647,
            BudgetUnit.BLOCKS,
            recent_ratio + important_ratio,
            1,
        ),
        method=MosaicKVMethod.LOOKM_REIMPL,
        forecasting=ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=2,
            draft_steps=0,
            centroid_count=2,
        ),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
        repair=RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
        lookm=LookMConfig(
            enabled=True,
            recent_ratio=recent_ratio,
            important_ratio=important_ratio,
            merge_strategy=LookMMergeStrategy.PIVOTAL,
        ),
    )


def _vl_cache_runtime_config(*, retention_one: bool = False) -> RunConfig:
    ratio = 1.0 if retention_one else 0.5
    return RunConfig(
        model=ModelConfig("tiny-random/LLaVA-1.5", "a" * 40, Precision.FP32),
        dataset=DatasetConfig("mosaickv/tiny-runtime", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 17, True),
        generation=GenerationConfig(max_new_tokens=2),
        cache=CacheConfig(2_147_483_647, BudgetUnit.BLOCKS, ratio, 1),
        method=MosaicKVMethod.VL_CACHE_REIMPL,
        forecasting=ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=1,
            draft_steps=0,
            centroid_count=1,
        ),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
        repair=RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
        vl_cache=VLCacheConfig(enabled=True),
    )


def _vl_cache_llava() -> Llava15Adapter:
    adapter = _llava(kv_heads=2)
    adapter.processor = _VLCacheProcessor()
    return adapter


def _runtime_request(run_id: str) -> EvaluationRequest:
    return EvaluationRequest(
        run_id=run_id,
        sample_id="tiny-sample",
        task="synthetic_smoke",
        messages=build_multimodal_messages("tiny prompt"),
        generation_kwargs={},
    )


@pytest.mark.hf
@pytest.mark.integration
@pytest.mark.parametrize("builder", [_llava, _qwen, _onevision])
def test_tiny_architecture_full_cache_and_reinjection_parity(builder: Any) -> None:
    """Exercise real HF architecture code without downloading model weights."""

    torch.manual_seed(0)
    adapter = builder()
    prepared = adapter.prepare_inputs([{"role": "user", "content": "tiny prompt"}])
    generation = compare_with_generate(adapter, prepared, max_new_tokens=16)
    reinjection = compare_cache_reinjection(adapter, prepared, max_new_tokens=16)
    mosaic_reinjection = compare_mosaickv_retention_one(
        adapter, prepared, max_new_tokens=16, block_size=3
    )
    assert generation.token_agreement == 1.0
    assert generation.maximum_logit_difference <= 1e-6
    assert reinjection.token_agreement == 1.0
    assert reinjection.maximum_logit_difference <= 1e-6
    assert mosaic_reinjection.token_agreement == 1.0
    assert mosaic_reinjection.maximum_logit_difference <= 1e-6


@pytest.mark.hf
@pytest.mark.integration
@pytest.mark.parametrize("builder", [_llava, _qwen, _onevision])
def test_tiny_architecture_hybrid_forecast_is_isolated_and_reproducible(
    builder: Any,
) -> None:
    torch.manual_seed(0)
    adapter = builder()
    prepared = adapter.prepare_inputs([{"role": "user", "content": "tiny prompt"}])
    prefill = adapter.prefill(prepared)
    cache_before = adapter.extract_past_key_values(prefill.state.past_key_values)
    config = ForecastingConfig(
        mode=ForecastMode.HYBRID,
        prompt_window=3,
        draft_steps=3,
        centroid_count=2,
        low_memory_centroids=True,
    )

    first = forecast_from_prefill(adapter, prefill, config)
    second = forecast_from_prefill(adapter, prefill, config)
    oracle = collect_evaluation_only_true_future_queries(adapter, prefill, future_steps=3)
    cache_after = adapter.extract_past_key_values(prefill.state.past_key_values)

    for before_layer, after_layer in zip(cache_before.layers, cache_after.layers, strict=True):
        assert torch.equal(before_layer.key, after_layer.key)
        assert torch.equal(before_layer.value, after_layer.value)
    assert first.provenance.reused_original_prefill
    assert first.provenance.draft_cache_isolated
    assert first.timing.draft_decode >= 0
    assert first.timing.total >= first.timing.draft_decode
    for layer_index, (first_layer, second_layer) in enumerate(
        zip(first.layers, second.layers, strict=True)
    ):
        oracle_layer = oracle.query_layers[layer_index]
        for left, right in zip(first_layer, second_layer, strict=True):
            assert torch.equal(left.normalized_centroids, right.normalized_centroids)
            assert torch.equal(left.forecast_weights, right.forecast_weights)
            assert left.draft_query_samples is not None
            provenance = left.provenance
            expected = (
                oracle_layer[
                    :,
                    provenance.query_head_start : provenance.query_head_end,
                    :,
                    :,
                ]
                .detach()
                .float()
                .reshape(-1, oracle_layer.shape[-1])
            )
            assert torch.equal(left.draft_query_samples, expected)


@pytest.mark.hf
@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.MOSAICKV_EXACT,
        MosaicKVMethod.MOSAICKV_PROTO,
        MosaicKVMethod.MOSAICKV_FULL,
    ),
)
def test_unified_runtime_methods_decode_and_write_complete_traces(
    method: MosaicKVMethod, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    runtime = HuggingFaceMosaicKVModel(
        _llava(),
        _runtime_config(method, 0.5),
        trace_directory=tmp_path,
    )

    generated = runtime.generate(_runtime_request(f"tiny-{method.value}"))

    assert generated.answer.strip()
    assert generated.metrics.generated_tokens == 4
    assert generated.metrics.active_kv_bytes is not None
    trace_path = next((tmp_path / f"tiny-{method.value}").glob("*.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "completed"
    assert trace["selected_blocks"]
    assert "prototypes" in trace
    assert "graph_edges" in trace
    assert "forecast_statistics" in trace
    assert "repair_events" in trace
    assert "timing_breakdown" in trace
    if method is not MosaicKVMethod.MOSAICKV_EXACT:
        assert trace["effective_method"].endswith("mosaickv_exact_safety_fallback")


@pytest.mark.hf
@pytest.mark.integration
def test_unified_runtime_retention_one_matches_fullkv_and_bytes_are_monotonic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    adapter = _llava()
    parity = compare_runtime_retention_one(
        adapter,
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        _runtime_config(MosaicKVMethod.MOSAICKV_EXACT, 1.0),
    )
    assert parity.token_agreement == 1.0
    assert parity.maximum_logit_difference <= 1e-6
    full = HuggingFaceMosaicKVModel(
        adapter,
        _runtime_config(MosaicKVMethod.FULLKV, 1.0),
        trace_directory=tmp_path,
    )
    full_generation = full.generate(_runtime_request("tiny-fullkv"))
    full_trace_path = next((tmp_path / "tiny-fullkv").glob("*.json"))
    full_trace = json.loads(full_trace_path.read_text(encoding="utf-8"))

    active_bytes: list[int] = []
    for ratio in (0.5, 0.75, 1.0):
        runtime = HuggingFaceMosaicKVModel(
            adapter,
            _runtime_config(MosaicKVMethod.MOSAICKV_EXACT, ratio),
            trace_directory=tmp_path,
        )
        generated = runtime.generate(_runtime_request(f"tiny-exact-{ratio}"))
        assert generated.metrics.active_kv_bytes is not None
        active_bytes.append(generated.metrics.active_kv_bytes)
        if ratio == 1.0:
            trace_path = next((tmp_path / f"tiny-exact-{ratio}").glob("*.json"))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            assert trace["generated_token_ids"] == full_trace["generated_token_ids"]
            assert generated.answer == full_generation.answer
    assert active_bytes == sorted(active_bytes)


@pytest.mark.hf
@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.RANDOM_KV,
        MosaicKVMethod.UNIFORM_KV,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        MosaicKVMethod.VALUE_TOPK,
    ),
)
def test_exact_baselines_use_shared_decode_runtime_and_complete_traces(
    method: MosaicKVMethod,
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    runtime = HuggingFaceMosaicKVModel(
        _llava(),
        _baseline_runtime_config(method, 0.5),
        trace_directory=tmp_path,
    )

    generated = runtime.generate(_runtime_request(f"tiny-{method.value}"))
    trace = json.loads(
        next((tmp_path / f"tiny-{method.value}").glob("*.json")).read_text(encoding="utf-8")
    )

    assert generated.answer.strip()
    assert generated.metrics.generated_tokens == 2
    assert generated.metrics.active_kv_bytes is not None
    assert trace["status"] == "completed"
    assert trace["selected_blocks"]
    assert trace["baseline_seed"] == 17
    assert trace["selected_source_bytes"] > 0
    assert trace["prototypes"] == []
    assert trace["repair_events"] == []
    assert trace["graph_edges"] == []


@pytest.mark.hf
@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.RANDOM_KV,
        MosaicKVMethod.UNIFORM_KV,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        MosaicKVMethod.VALUE_TOPK,
    ),
)
def test_every_exact_baseline_retention_one_matches_fullkv(method: MosaicKVMethod) -> None:
    torch.manual_seed(0)
    adapter = _llava()

    parity = compare_runtime_retention_one(
        adapter,
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        _baseline_runtime_config(method, 1.0),
    )

    assert parity.token_agreement == 1.0
    assert parity.maximum_logit_difference <= 1e-6


@pytest.mark.hf
@pytest.mark.integration
def test_full_kv_baseline_is_unmodified_reference_path(tmp_path: Path) -> None:
    runtime = HuggingFaceMosaicKVModel(
        _llava(),
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        trace_directory=tmp_path,
    )

    generated = runtime.generate(_runtime_request("tiny-full-kv"))
    trace = json.loads(next((tmp_path / "tiny-full-kv").glob("*.json")).read_text(encoding="utf-8"))

    assert generated.answer.strip()
    assert generated.metrics.compression_time == 0.0
    assert trace["tier_mode"] == "full_cache"
    assert trace["prototypes"] == []
    assert trace["repair_events"] == []


@pytest.mark.hf
@pytest.mark.integration
def test_lookm_reimpl_uses_shared_runtime_and_is_never_labeled_official(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    runtime = HuggingFaceMosaicKVModel(
        _llava(kv_heads=2),
        _lookm_runtime_config(),
        trace_directory=tmp_path,
    )

    generated = runtime.generate(_runtime_request("tiny-lookm-reimpl"))
    trace = json.loads(
        next((tmp_path / "tiny-lookm-reimpl").glob("*.json")).read_text(encoding="utf-8")
    )

    assert generated.answer.strip()
    assert generated.metrics.generated_tokens == 2
    assert trace["implementation"] == "lookm_reimpl"
    assert trace["official_code"] is False
    assert trace["merge_strategy"] == "pivotal"
    assert trace["selected_blocks"]
    assert trace["prototypes"] == []
    assert trace["repair_events"] == []


@pytest.mark.hf
@pytest.mark.integration
def test_lookm_reimpl_retention_one_matches_fullkv() -> None:
    torch.manual_seed(0)
    adapter = _llava(kv_heads=2)
    parity = compare_runtime_retention_one(
        adapter,
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        _lookm_runtime_config(retention_one=True),
    )

    assert parity.token_agreement == 1.0
    assert parity.maximum_logit_difference <= 1e-6


@pytest.mark.hf
@pytest.mark.integration
def test_vl_cache_reimpl_uses_shared_runtime_and_is_never_labeled_official(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    runtime = HuggingFaceMosaicKVModel(
        _vl_cache_llava(),
        _vl_cache_runtime_config(),
        trace_directory=tmp_path,
    )

    generated = runtime.generate(_runtime_request("tiny-vl-cache-reimpl"))
    trace = json.loads(
        next((tmp_path / "tiny-vl-cache-reimpl").glob("*.json")).read_text(encoding="utf-8")
    )

    assert generated.answer.strip()
    assert trace["method"] == "vl_cache_reimpl"
    vl_cache_trace = cast("dict[str, Any]", trace["vl_cache"])
    assert vl_cache_trace["implementation"] == "vl_cache_reimpl"
    assert vl_cache_trace["official_code"] is False
    assert trace["prototypes"] == []
    assert trace["repair_events"] == []


@pytest.mark.hf
@pytest.mark.integration
def test_vl_cache_reimpl_retention_one_matches_fullkv() -> None:
    torch.manual_seed(0)
    adapter = _vl_cache_llava()
    parity = compare_runtime_retention_one(
        adapter,
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        _vl_cache_runtime_config(retention_one=True),
    )

    assert parity.token_agreement == 1.0
    assert parity.maximum_logit_difference <= 1e-6
