from __future__ import annotations

import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mosaickv.backends.vllm_runtime import (
    NativeMosaicKVUnsupported,
    VLLMFullKVModel,
    VLLMRuntimeOptions,
    VLLMTrialMeasurement,
    _resolve_model_source,
    native_integration_capability,
    prepare_vllm_prompt,
    require_native_mosaickv_support,
)
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.types import JsonObject


class _Processor:
    chat_template = "template"

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


class _NoTemplateProcessor:
    chat_template = None
    tokenizer = None


def _trial(request_id: str, *, cached: int, mm_hit: int) -> VLLMTrialMeasurement:
    return VLLMTrialMeasurement(
        request_id=request_id,
        answer="blue",
        token_ids=(7, 9),
        prompt_tokens=8,
        generated_tokens=2,
        ttft_seconds=0.1,
        request_latency_seconds=0.2,
        decode_seconds=0.1,
        inter_token_latencies_seconds=(0.1,),
        token_timestamps_seconds=(0.1, 0.2),
        throughput_tokens_per_second=10.0,
        decode_throughput_tokens_per_second=10.0,
        num_cached_tokens=cached,
        prefix_cache_hit_rate=cached / 8,
        mm_cache_queries=1,
        mm_cache_hits=mm_hit,
        mm_cache_hit_rate=float(mm_hit),
        engine_prefill_seconds=0.08,
        engine_decode_seconds=0.1,
        engine_ttft_seconds=0.09,
        engine_inter_token_latencies_seconds=(0.095,),
        gpu_memory_source="test counter",
        gpu_memory_baseline_bytes=100,
        gpu_memory_peak_bytes=120,
        gpu_memory_peak_delta_bytes=20,
        finish_reason="length",
    )


class _Runner:
    def __init__(self) -> None:
        self.vllm_version = "0.11.2"
        self.engine_metadata: JsonObject = {
            "vllm_version": "0.11.2",
            "cuda_graph": False,
        }
        self.calls = 0
        self.closed = False

    def run(
        self,
        prompt: dict[str, Any],
        generation: Mapping[str, object],
        request_id: str,
    ) -> VLLMTrialMeasurement:
        assert prompt["prompt"].endswith("|assistant:")
        assert generation["temperature"] == 0.0
        self.calls += 1
        return _trial(
            request_id,
            cached=0 if self.calls == 1 else 8,
            mm_hit=0 if self.calls == 1 else 1,
        )

    def close(self) -> None:
        self.closed = True


def test_prepare_prompt_preserves_ordered_media_and_chat_template() -> None:
    image = b"image-bytes"
    messages = build_multimodal_messages("What color?", (MediaItem(MediaKind.IMAGE, image),))
    prepared = prepare_vllm_prompt(_Processor(), "Qwen/Qwen2.5-VL-3B-Instruct", messages)

    assert prepared.rendered_text == "user:<image>What color?|assistant:"
    assert prepared.engine_prompt["multi_modal_data"] == {"image": image}
    assert len(prepared.prompt_sha256) == 64
    assert len(prepared.media_sha256) == 64


def test_offline_model_source_resolves_pinned_snapshot_once(
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
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = _resolve_model_source("org/model", "immutable-sha", local_files_only=True)

    assert resolved == str(snapshot.resolve())
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "immutable-sha",
            "cache_dir": str(tmp_path / "hub"),
            "local_files_only": True,
        }
    ]


def test_offline_model_source_accepts_runtime_complete_docs_partial_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "hub"
    revision = "immutable-sha"
    snapshot = cache / "models--org--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model-00001-of-00001.safetensors",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    def incomplete_snapshot(**kwargs: object) -> str:
        raise RuntimeError("README.md was not downloaded")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = incomplete_snapshot  # type: ignore[attr-defined]
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = _resolve_model_source("org/model", revision, local_files_only=True)

    assert resolved == str(snapshot.resolve())


def test_llava_fallback_matches_hf_adapter_prompt_shape() -> None:
    messages = build_multimodal_messages("Describe it.", (MediaItem(MediaKind.IMAGE, b"x"),))
    prepared = prepare_vllm_prompt(_NoTemplateProcessor(), "llava-hf/llava-1.5-7b-hf", messages)
    assert prepared.rendered_text == "USER: <image>\nDescribe it. ASSISTANT:"


def test_fullkv_wrapper_writes_raw_trials_and_uses_first_measurement(tmp_path: Path) -> None:
    runner = _Runner()
    model = VLLMFullKVModel(
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
    assert output.metrics.ttft == 0.1
    assert output.metrics.generated_tokens == 2
    assert output.metrics.active_kv_bytes is None
    traces = list((tmp_path / "run").glob("*.json"))
    assert len(traces) == 1
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert payload["measurement_type"] == "vllm_fullkv"
    assert payload["native_mosaickv"] is False
    assert payload["trials"][0]["prefix_cache_hit_rate"] == 0.0
    assert payload["trials"][1]["prefix_cache_hit_rate"] == 1.0
    assert payload["trials"][1]["mm_cache_hit_rate"] == 1.0


def test_native_feature_is_explicitly_unsupported_for_installed_version() -> None:
    capability = native_integration_capability("0.11.2")
    assert not capability.supported
    assert capability.reason_code == "audited_0_11_2_missing_sparse_logical_block_table_hook"
    with pytest.raises(NativeMosaicKVUnsupported, match="No simulated MosaicKV row"):
        require_native_mosaickv_support(
            enabled=True,
            vllm_version="0.11.2",
            enforce_eager=True,
            attention_backend="eager",
        )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization"),
        ({"max_model_len": 1}, "max_model_len"),
        ({"cache_probe_repeats": 0}, "cache_probe_repeats"),
    ),
)
def test_runtime_options_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VLLMRuntimeOptions(**kwargs)  # type: ignore[arg-type]
