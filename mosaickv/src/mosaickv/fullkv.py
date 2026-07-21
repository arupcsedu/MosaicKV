"""FullKV: immutable, uncompressed ground-truth KV-cache reference."""

from __future__ import annotations

import hashlib
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from typing import Any

from mosaickv.adapters.huggingface import HuggingFaceMultimodalAdapter, PreparedInputs
from mosaickv.adapters.huggingface.base import _torch
from mosaickv.evaluation.model import (
    EvaluationRequest,
    GenerationMetrics,
    ModelGeneration,
)
from mosaickv.measurements.memory import active_kv_bytes, cache_tensors, cpu_residual_bytes
from mosaickv.measurements.statistics import aggregate_trials
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.measurements.timing import CudaEventTimer, ModuleCudaTimer, SynchronizationAudit
from mosaickv.measurements.types import (
    FullKVAggregate,
    FullKVTrialMeasurement,
    FullKVTrialOutput,
    MemoryMeasurements,
    PhaseTimings,
)


@dataclass(frozen=True, slots=True)
class FullKVBenchmarkConfig:
    """Repeated-trial controls for a FullKV measurement run."""

    warmups: int = 1
    repeated_trials: int = 5
    max_new_tokens: int = 16
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if self.repeated_trials < 1:
            raise ValueError("repeated_trials must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class FullKVSample:
    """One preprocessed sample used identically across warmups and trials."""

    sample_id: str
    prepared: PreparedInputs

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")


@dataclass(frozen=True, slots=True)
class FullKVBenchmarkOutput:
    """Raw trials, derived aggregate, and tokenization identity."""

    trials: tuple[FullKVTrialMeasurement, ...]
    aggregate: FullKVAggregate
    tokenization_sha: str


class FullKV:
    """Ground-truth full cache with no cache transformation of any kind."""

    method = "fullkv"
    backend = "huggingface"
    retention_ratio = 1.0
    supports_prototype_merge = False
    supports_residual_repair = False

    def __init__(
        self,
        adapter: HuggingFaceMultimodalAdapter,
        *,
        model_id: str,
        model_revision: str,
    ) -> None:
        if not model_id.strip() or not model_revision.strip():
            raise ValueError("FullKV requires model ID and immutable revision")
        self.adapter = adapter
        self.model_id = model_id
        self.model_revision = model_revision

    @staticmethod
    def _validate_generation(*, temperature: float, do_sample: bool) -> None:
        if temperature != 0.0:
            raise ValueError("FullKV reference requires temperature=0")
        if do_sample:
            raise ValueError("FullKV reference requires do_sample=False")

    @staticmethod
    def _decode_answer(processor: Any, token_ids: Any) -> str:
        decoder = getattr(processor, "batch_decode", None)
        if decoder is None:
            tokenizer = getattr(processor, "tokenizer", processor)
            decoder = getattr(tokenizer, "batch_decode", None)
        if decoder is None:
            raise RuntimeError("processor/tokenizer does not provide batch_decode")
        answers = decoder(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not answers:
            raise RuntimeError("batch_decode returned no answer")
        return str(answers[0])

    @staticmethod
    def _assert_device_resident_cache(cache: Any) -> None:
        for index, tensor in enumerate(cache_tensors(cache)):
            device = getattr(tensor, "device", None)
            if getattr(device, "type", str(device).split(":", maxsplit=1)[0]) != "cuda":
                raise RuntimeError(
                    f"FullKV cache tensor {index} is not CUDA-resident; offloading is forbidden"
                )

    def run_trial(
        self,
        prepared: PreparedInputs,
        *,
        run_id: str,
        sample_id: str,
        trial_index: int,
        dataset_id: str,
        dataset_revision: str,
        manifest_path: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        do_sample: bool = False,
        measurement_type: str = "reference_measured",
    ) -> FullKVTrialOutput:
        """Run one synchronized trial without changing the returned KV cache."""

        self._validate_generation(temperature=temperature, do_sample=do_sample)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("FullKV measured trials require CUDA")
        device = self.adapter.device
        if getattr(device, "type", str(device).split(":", maxsplit=1)[0]) != "cuda":
            raise RuntimeError(f"FullKV measured trial requires a CUDA model device, got {device}")

        before = capture_gpu_environment(torch)
        audit = SynchronizationAudit()
        torch.cuda.synchronize(device)
        audit.calls += 1
        torch.cuda.reset_peak_memory_stats(device)
        profiling = self.adapter.get_profiling_modules()
        vision_timer = ModuleCudaTimer(profiling.vision_encoder, torch, device, audit)
        projector_timer = ModuleCudaTimer(profiling.projector, torch, device, audit)
        language_timer = ModuleCudaTimer(profiling.language_model, torch, device, audit)
        total_timer = CudaEventTimer(torch, device, audit)
        ttft_timer = CudaEventTimer(torch, device, audit)
        host_started = time.perf_counter()
        total_timer.start()
        ttft_timer.start()
        with ExitStack() as stack:
            stack.enter_context(vision_timer)
            stack.enter_context(projector_timer)
            stack.enter_context(language_timer)
            prefill = self.adapter.prefill(prepared, capture_queries=False)
        if language_timer.invocation_count != 1 or language_timer.total_seconds is None:
            raise RuntimeError(
                "language-model prefill module must execute exactly once; "
                f"observed {language_timer.invocation_count}"
            )

        state = prefill.state
        self._assert_device_resident_cache(state.past_key_values)
        cache_identity = id(state.past_key_values)
        bytes_before_identity = active_kv_bytes(state.past_key_values)
        # FullKV's compression phase is intentionally an empty identity operation.
        compression_seconds = 0.0
        if id(state.past_key_values) != cache_identity:
            raise RuntimeError("FullKV compression identity invariant failed")
        if active_kv_bytes(state.past_key_values) != bytes_before_identity:
            raise RuntimeError("FullKV changed cache bytes during its no-op compression phase")
        ttft_seconds = ttft_timer.stop()

        tokens = [prefill.next_token_id]
        token = prefill.next_token_id
        decode_seconds: list[float] = []
        repair_seconds = 0.0
        for _index in range(max_new_tokens - 1):
            decode_timer = CudaEventTimer(torch, device, audit)
            decode_timer.start()
            step = self.adapter.decode_one_token(token, state, capture_queries=False)
            decode_seconds.append(decode_timer.stop())
            state = step.state
            token = step.next_token_id
            tokens.append(token)
            # FullKV never invokes repair or creates residual state.

        token_ids = torch.cat(tokens, dim=-1)
        total_seconds = total_timer.stop()
        host_total_seconds = time.perf_counter() - host_started
        measured_kv_bytes = active_kv_bytes(state.past_key_values)
        self._assert_device_resident_cache(state.past_key_values)
        memory = MemoryMeasurements(
            max_memory_allocated=int(torch.cuda.max_memory_allocated(device)),
            max_memory_reserved=int(torch.cuda.max_memory_reserved(device)),
            active_kv_bytes=measured_kv_bytes,
            cpu_residual_bytes=cpu_residual_bytes(None),
        )
        vision_seconds = vision_timer.total_seconds
        projector_seconds = projector_timer.total_seconds
        if (
            profiling.vision_includes_projector
            and vision_seconds is not None
            and projector_seconds is not None
        ):
            vision_seconds = max(0.0, vision_seconds - projector_seconds)
        timings = PhaseTimings(
            image_video_encoder=vision_seconds,
            projector=projector_seconds,
            language_model_prefill=language_timer.total_seconds,
            compression=compression_seconds,
            ttft=ttft_seconds,
            per_token_decode=tuple(decode_seconds),
            repair=repair_seconds,
            total_latency=total_seconds,
            host_total_latency=host_total_seconds,
        )
        after = capture_gpu_environment(torch)
        answer = self._decode_answer(self.adapter.processor, token_ids)
        measurement = FullKVTrialMeasurement(
            run_id=run_id,
            sample_id=sample_id,
            trial_index=trial_index,
            model_id=self.model_id,
            model_revision=self.model_revision,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            manifest_path=manifest_path,
            status="completed",
            error=None,
            answer=answer,
            generated_token_ids=tuple(int(value) for value in token_ids.detach().cpu().reshape(-1)),
            timings=timings,
            memory=memory,
            active_cache_length=state.active_cache_length,
            logical_sequence_length=state.logical_sequence_length,
            synchronization_calls=audit.calls,
            phase_event_counts={
                "image_video_encoder": vision_timer.invocation_count,
                "projector": projector_timer.invocation_count,
                "language_model_prefill": language_timer.invocation_count,
                "compression": 0,
                "decode": len(decode_seconds),
                "repair": 0,
                "ttft": 1,
                "total_latency": 1,
            },
            gpu_before=before,
            gpu_after=after,
            measurement_type=measurement_type,
        )
        return FullKVTrialOutput(answer=answer, token_ids=token_ids, measurement=measurement)


class FullKVBenchmarkRunner:
    """Run warmups and preserve every repeated FullKV trial."""

    def __init__(self, reference: FullKV, config: FullKVBenchmarkConfig) -> None:
        self.reference = reference
        self.config = config

    @staticmethod
    def update_tokenization_digest(digest: Any, sample: FullKVSample) -> None:
        """Add an unambiguous sample/tokenization record to a SHA-256 digest."""

        input_ids = sample.prepared.model_inputs["input_ids"].detach().cpu().contiguous()
        sample_bytes = sample.sample_id.encode("utf-8")
        digest.update(len(sample_bytes).to_bytes(8, "big"))
        digest.update(sample_bytes)
        digest.update(str(input_ids.dtype).encode("ascii"))
        digest.update(str(tuple(int(value) for value in input_ids.shape)).encode("ascii"))
        digest.update(input_ids.numpy().tobytes())

    def run_sample(
        self,
        sample: FullKVSample,
        *,
        run_id: str,
        dataset_id: str,
        dataset_revision: str,
        manifest_path: str,
    ) -> tuple[FullKVTrialMeasurement, ...]:
        """Warm up and measure one prepared sample, preserving failed trial rows."""

        for warmup_index in range(self.config.warmups):
            self.reference.run_trial(
                sample.prepared,
                run_id=run_id,
                sample_id=sample.sample_id,
                trial_index=warmup_index,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                manifest_path=manifest_path,
                max_new_tokens=self.config.max_new_tokens,
                measurement_type="validation_warmup",
            )

        trials: list[FullKVTrialMeasurement] = []
        for trial_index in range(self.config.repeated_trials):
            try:
                output = self.reference.run_trial(
                    sample.prepared,
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    trial_index=trial_index,
                    dataset_id=dataset_id,
                    dataset_revision=dataset_revision,
                    manifest_path=manifest_path,
                    max_new_tokens=self.config.max_new_tokens,
                )
                trials.append(output.measurement)
            except Exception as error:
                torch = _torch()
                with suppress(Exception):
                    # The original failure remains the scientifically relevant error.
                    torch.cuda.synchronize(self.reference.adapter.device)
                snapshot = capture_gpu_environment(torch)
                trials.append(
                    FullKVTrialMeasurement(
                        run_id=run_id,
                        sample_id=sample.sample_id,
                        trial_index=trial_index,
                        model_id=self.reference.model_id,
                        model_revision=self.reference.model_revision,
                        dataset_id=dataset_id,
                        dataset_revision=dataset_revision,
                        manifest_path=manifest_path,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                        answer=None,
                        generated_token_ids=(),
                        timings=None,
                        memory=None,
                        active_cache_length=None,
                        logical_sequence_length=None,
                        synchronization_calls=0,
                        phase_event_counts={},
                        gpu_before=snapshot,
                        gpu_after=snapshot,
                    )
                )
        return tuple(trials)

    def run(
        self,
        samples: tuple[FullKVSample, ...],
        *,
        run_id: str,
        dataset_id: str,
        dataset_revision: str,
        manifest_path: str,
    ) -> FullKVBenchmarkOutput:
        if not samples:
            raise ValueError("FullKV benchmark requires at least one sample")
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("FullKV sample IDs must be unique")
        trials: list[FullKVTrialMeasurement] = []
        for sample in samples:
            trials.extend(
                self.run_sample(
                    sample,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    dataset_revision=dataset_revision,
                    manifest_path=manifest_path,
                )
            )

        aggregate = aggregate_trials(
            run_id,
            trials,
            warmups=self.config.warmups,
            repeated_trials=self.config.repeated_trials,
            bootstrap_samples=self.config.bootstrap_samples,
            confidence_level=self.config.confidence_level,
            seed=self.config.seed,
        )
        digest = hashlib.sha256()
        for sample in samples:
            self.update_tokenization_digest(digest, sample)
        return FullKVBenchmarkOutput(tuple(trials), aggregate, digest.hexdigest())


class FullKVEvaluationModel:
    """Expose FullKV through the unified local/lmms-eval model protocol."""

    backend = "huggingface"
    method = "fullkv"
    retention_ratio = 1.0

    def __init__(
        self,
        reference: FullKV,
        *,
        dataset_id: str,
        dataset_revision: str,
        manifest_path: str,
        max_new_tokens: int,
    ) -> None:
        if not dataset_id.strip() or not dataset_revision.strip() or not manifest_path.strip():
            raise ValueError("FullKV evaluation provenance fields must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.reference = reference
        self.model_id = reference.model_id
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.manifest_path = manifest_path
        self.max_new_tokens = max_new_tokens
        self.supports_video = reference.adapter.capabilities.video
        self._measurements: list[FullKVTrialMeasurement] = []

    @property
    def raw_measurements(self) -> tuple[FullKVTrialMeasurement, ...]:
        """Return all per-sample FullKV observations made by this wrapper."""

        return tuple(self._measurements)

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Prepare one request and run the synchronized FullKV reference path."""

        unknown = sorted(
            set(request.generation_kwargs) - {"max_new_tokens", "temperature", "do_sample"}
        )
        if unknown:
            raise ValueError("unsupported FullKV generation argument(s): " + ", ".join(unknown))
        max_new_tokens = request.generation_kwargs.get("max_new_tokens", self.max_new_tokens)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if max_new_tokens != self.max_new_tokens:
            raise ValueError(
                "request max_new_tokens differs from the controlled FullKV configuration"
            )
        temperature = request.generation_kwargs.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        do_sample = request.generation_kwargs.get("do_sample", False)
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be boolean")
        prepared = self.reference.adapter.prepare_inputs(request.messages)
        output = self.reference.run_trial(
            prepared,
            run_id=request.run_id,
            sample_id=request.sample_id,
            trial_index=0,
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            manifest_path=self.manifest_path,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            do_sample=do_sample,
        )
        measurement = output.measurement
        self._measurements.append(measurement)
        if measurement.timings is None or measurement.memory is None:
            raise RuntimeError("completed FullKV generation has no measurements")
        timings = measurement.timings
        memory = measurement.memory
        return ModelGeneration(
            answer=output.answer,
            metrics=GenerationMetrics(
                ttft=timings.ttft,
                prefill_time=timings.language_model_prefill,
                compression_time=timings.compression,
                decode_time=timings.decode_total,
                end_to_end_time=timings.total_latency,
                generated_tokens=len(measurement.generated_token_ids),
                active_kv_bytes=memory.active_kv_bytes,
                residual_kv_bytes=memory.cpu_residual_bytes,
                peak_gpu_memory=memory.max_memory_allocated,
                repair_count=0,
                repaired_bytes=0,
            ),
        )


__all__ = [
    "FullKV",
    "FullKVBenchmarkConfig",
    "FullKVBenchmarkOutput",
    "FullKVBenchmarkRunner",
    "FullKVEvaluationModel",
    "FullKVSample",
]
