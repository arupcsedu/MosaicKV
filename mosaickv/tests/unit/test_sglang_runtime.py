from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mosaickv.backends.sglang_runtime import (
    KVCacheGeometry,
    NativeMosaicKVUnsupported,
    SGLangFullKVModel,
    SGLangRuntimeOptions,
    SGLangTrialMeasurement,
    build_server_command,
    kv_cache_geometry,
    native_integration_capability,
    prepare_sglang_prompt,
    require_native_mosaickv_support,
)
from mosaickv.config import load_config
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.types import JsonObject, Precision


class _Encoded:
    input_ids: ClassVar[list[int]] = [1, 2, 3]


class _Processor:
    chat_template = "template"
    tokenizer: _Processor

    def __init__(self) -> None:
        self.tokenizer = self

    def apply_chat_template(
        self, chat: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        rendered: list[str] = []
        for message in chat:
            content = "".join(
                f"<{part['type']}>" if part["type"] != "text" else str(part["text"])
                for part in message["content"]
            )
            rendered.append(f"{message['role']}:{content}")
        return "|".join(rendered) + "|assistant:"

    def __call__(self, text: str, *, add_special_tokens: bool) -> _Encoded:
        assert text.endswith("|assistant:")
        assert not add_special_tokens
        return _Encoded()

    def batch_decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        del token_ids, skip_special_tokens, clean_up_tokenization_spaces
        return ["blue"]


def _trial(request_id: str, *, cached: int) -> SGLangTrialMeasurement:
    geometry = KVCacheGeometry(layers=2, kv_heads=2, head_dim=4, dtype_bytes=2)
    active_positions = 9
    return SGLangTrialMeasurement(
        request_id=request_id,
        answer="blue",
        token_ids=(7, 9),
        prompt_tokens=8,
        generated_tokens=2,
        active_cache_positions=active_positions,
        active_kv_bytes=geometry.active_bytes(active_positions),
        ttft_seconds=0.1,
        request_latency_seconds=0.2,
        decode_seconds=0.1,
        inter_token_latencies_seconds=(0.1,),
        token_timestamps_seconds=(0.1, 0.2),
        throughput_tokens_per_second=10.0,
        decode_throughput_tokens_per_second=10.0,
        cached_tokens=cached,
        prefix_cache_hit_rate=cached / 8,
        server_e2e_latency_seconds=0.19,
        server_prefill_seconds=0.08,
        server_queue_seconds=0.0,
        prometheus_cache_hit_rate=cached / 8,
        prometheus_cached_tokens_delta=float(cached),
        prometheus_generation_throughput=10.0,
        prometheus_token_usage=0.01,
        gpu_memory_source="test counter",
        gpu_memory_baseline_bytes=100,
        gpu_memory_peak_bytes=120,
        gpu_memory_peak_delta_bytes=20,
        finish_reason="length",
    )


def _native_trial(request_id: str) -> SGLangTrialMeasurement:
    trial = _trial(request_id, cached=0)
    return SGLangTrialMeasurement(
        **{
            **trial.to_json_object(),
            "compression_seconds": 0.004,
            "native_audit": {
                "status": "scheduler_transaction_committed",
                "selection": {
                    "logical_positions": 8,
                    "selected_positions": 8,
                    "source_blocks": 1,
                    "selected_blocks": [0],
                    "mandatory_blocks": [0],
                    "retention_ratio": 1.0,
                },
                "compression_seconds": 0.004,
                "logical_next_position": 8,
                "radix_cache_disabled": True,
            },
        }
    )


class _Runner:
    def __init__(self) -> None:
        self.sglang_version = "0.5.1.post3"
        self.cache_geometry = KVCacheGeometry(2, 2, 4, 2)
        self.engine_metadata: JsonObject = {
            "sglang_version": "0.5.1.post3",
            "cuda_graph": False,
            "overlap_schedule": False,
        }
        self.calls = 0
        self.flushes = 0
        self.closed = False

    def run(
        self,
        prompt: JsonObject,
        generation: Mapping[str, object],
        request_id: str,
    ) -> SGLangTrialMeasurement:
        assert str(prompt["text"]).endswith("|assistant:")
        assert isinstance(prompt["image_data"], str)
        assert generation["temperature"] == 0.0
        self.calls += 1
        return _trial(request_id, cached=0 if self.calls == 1 else 8)

    def close(self) -> None:
        self.closed = True

    def flush_cache(self) -> None:
        self.flushes += 1


class _NativeRunner(_Runner):
    def run(
        self,
        prompt: JsonObject,
        generation: Mapping[str, object],
        request_id: str,
    ) -> SGLangTrialMeasurement:
        del prompt, generation
        self.calls += 1
        return _native_trial(request_id)


def test_prepare_prompt_reuses_chat_boundary_and_makes_media_json_safe() -> None:
    messages = build_multimodal_messages(
        "What color?", (MediaItem(MediaKind.IMAGE, b"image-bytes"),)
    )
    prepared = prepare_sglang_prompt(_Processor(), "Qwen/Qwen2.5-VL-3B-Instruct", messages)

    assert prepared.rendered_text == "user:<image>What color?|assistant:"
    assert prepared.request_payload["modalities"] == ["image"]
    assert prepared.request_payload["image_data"] == "aW1hZ2UtYnl0ZXM="
    assert prepared.prompt_token_ids == (1, 2, 3)
    assert len(prepared.prompt_sha256) == len(prepared.media_sha256) == 64


def test_kv_geometry_uses_text_config_and_exact_byte_formula() -> None:
    class TextConfig:
        num_hidden_layers = 4
        num_attention_heads = 8
        num_key_value_heads = 2
        hidden_size = 64

    class Config:
        text_config = TextConfig()

    geometry = kv_cache_geometry(Config(), Precision.BF16)
    assert geometry == KVCacheGeometry(4, 2, 8, 2)
    assert geometry.bytes_per_position == 256
    assert geometry.active_bytes(11) == 2816


def test_correctness_server_command_disables_optimizations_and_records_revision() -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "configs" / "sglang_fullkv_3b.yaml")
    options = SGLangRuntimeOptions()
    command = build_server_command(config, options, model_source=config.model.id, port=30123)
    assert command[:3] == (
        command[0],
        "-m",
        "mosaickv.backends.sglang_server",
    )
    for required in (
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--enable-cache-report",
        "--stream-output",
    ):
        assert required in command
    for unsupported in (
        "--enable-deterministic-inference",
        "--skip-server-warmup",
        "--disable-fast-image-processor",
        "--model-impl",
        "--page-size",
        "--enable-multimodal",
    ):
        assert unsupported not in command
    assert command[command.index("--attention-backend") + 1] == "torch_native"
    assert command[command.index("--revision") + 1] == config.model.revision


def test_fullkv_wrapper_writes_trials_and_exact_active_bytes(tmp_path: Path) -> None:
    runner = _Runner()
    model = SGLangFullKVModel(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        processor=_Processor(),
        runner=runner,
        trace_directory=tmp_path,
        generation={"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0},
        cache_probe_repeats=2,
    )
    request = EvaluationRequest(
        run_id="run",
        sample_id="sample/one",
        task="synthetic_smoke",
        messages=build_multimodal_messages("What color?", (MediaItem(MediaKind.IMAGE, b"image"),)),
        generation_kwargs={},
    )

    output = model.generate(request)

    assert output.answer == "blue"
    assert output.effective_method == "full_kv"
    assert output.metrics.active_kv_bytes == 9 * 2 * 2 * 2 * 4 * 2
    traces = list((tmp_path / "run").glob("*.json"))
    assert len(traces) == 1
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert payload["measurement_type"] == "sglang_fullkv"
    assert payload["native_mosaickv"] is False
    assert payload["quality_answer"] == "blue"
    assert payload["trials"][1]["cached_tokens"] == 8
    assert payload["request_isolation"]["session_params"] is None


def test_request_isolation_rechecks_anchor_after_distinct_input(tmp_path: Path) -> None:
    runner = _Runner()
    model = SGLangFullKVModel(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        processor=_Processor(),
        runner=runner,
        trace_directory=tmp_path,
        generation={"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0},
        cache_probe_repeats=2,
    )
    for sample_id, media in (("first", b"red"), ("second", b"white")):
        model.generate(
            EvaluationRequest(
                run_id="run",
                sample_id=sample_id,
                task="synthetic_smoke",
                messages=build_multimodal_messages(
                    "What color?", (MediaItem(MediaKind.IMAGE, media),)
                ),
                generation_kwargs={},
            )
        )

    assert model.verify_request_isolation()
    anchor_path = next((tmp_path / "run").glob("first-*.json"))
    payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    probe = payload["request_isolation"]["post_intervening_request_probe"]
    assert probe["performed"] is True
    assert probe["token_ids_match_anchor"] is True
    assert probe["warm_distinct_token_ids_match_cold"] is True
    assert probe["intervening_distinct_input_fingerprints"] == 1
    assert runner.calls == 6
    assert runner.flushes == 1


def test_native_feature_is_version_pinned_and_unaudited_versions_fail() -> None:
    capability = native_integration_capability("0.5.1.post3")
    assert capability.supported
    assert capability.reason_code == "repository_patch_transaction_v1"
    require_native_mosaickv_support(enabled=True, sglang_version="0.5.1.post3")
    with pytest.raises(NativeMosaicKVUnsupported, match="unaudited_sglang_version"):
        require_native_mosaickv_support(enabled=True, sglang_version="0.5.2")


def test_native_server_disables_radix_cache_and_wrapper_records_transaction(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "configs" / "sglang_fullkv_3b.yaml")
    command = build_server_command(
        config,
        SGLangRuntimeOptions(enable_mosaickv=True),
        model_source=config.model.id,
        port=30123,
    )
    assert "--disable-radix-cache" in command

    model = SGLangFullKVModel(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        processor=_Processor(),
        runner=_NativeRunner(),
        trace_directory=tmp_path,
        generation={"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0},
        cache_probe_repeats=2,
        method="mosaickv_exact",
        retention_ratio=1.0,
    )
    output = model.generate(
        EvaluationRequest(
            run_id="native",
            sample_id="identity",
            task="synthetic_smoke",
            messages=build_multimodal_messages(
                "What color?", (MediaItem(MediaKind.IMAGE, b"image"),)
            ),
            generation_kwargs={},
        )
    )

    assert output.effective_method == "mosaickv_exact"
    assert output.metrics.compression_time == pytest.approx(0.004)
    trace = json.loads(next((tmp_path / "native").glob("*.json")).read_text())
    assert trace["measurement_type"] == "sglang_native_mosaickv"
    assert trace["trials"][0]["native_audit"]["logical_next_position"] == 8


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"mem_fraction_static": 1.0}, "mem_fraction_static"),
        ({"context_length": 1}, "context_length"),
        ({"cache_probe_repeats": 0}, "cache_probe_repeats"),
        ({"port": 70000}, "port"),
        ({"startup_timeout_seconds": 0}, "startup_timeout_seconds"),
    ),
)
def test_runtime_options_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SGLangRuntimeOptions(**kwargs)  # type: ignore[arg-type]
