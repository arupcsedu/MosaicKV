#!/usr/bin/env python3
"""Run a controlled official PrefixKV versus ``prefixkv_reimpl`` experiment.

The runner imports the pinned upstream cache implementation without changing
the submodule.  Both implementations share one legacy LLaVA checkpoint, one
model instance, one generated offline profile, and the upstream explicit
generation loop.  Outputs are non-canonical whenever the enclosing MosaicKV
worktree is dirty; that status is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np
import torch
from rouge import Rouge

from mosaickv.baselines import (
    PrefixKVParityArtifact,
    PrefixKVParityControls,
    PrefixKVSampleObservation,
    build_prefixkv_reimpl_plan,
    compare_prefixkv_artifacts,
    generate_prefixkv_profile,
    prefixkv_calibration_observation,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PrefixKVConfig
from mosaickv.types import BudgetUnit, PrefixKVProfileMode

OFFICIAL_SHA = "597f1ab032704951550f93bcc8a23f1454b80aa4"
MODEL_REPOSITORY_SHA = "833edbdc7512240f2a3aa49feeb9468e2297bdbc"
MODEL_ID = "Zuyan/ElasticCache/llava-v1.5-7b"
DATASET_ID = "PrefixKV/LLaVA-Description"
DATASET_REVISION = "google-drive-1_I2sokdpv8hLzLUe8UUmFvihbh6Kmytv"


class CacheCriterion(Protocol):
    initial_layer_sizes: tuple[int, ...]
    initial_active_bytes: int
    final_active_bytes: int
    selected_positions: tuple[tuple[int, ...], ...]

    def __call__(
        self,
        past_key_values: Any,
        num_of_token: int,
        attentions: Any,
    ) -> Any: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _command_output(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_bytes(cache: Any) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size()) for layer in cache for tensor in layer[:2]
    )


def _layer_sizes(cache: Any) -> tuple[int, ...]:
    return tuple(int(layer[0].shape[-2]) for layer in cache)


def _gather_layer(layer: Any, positions: tuple[int, ...]) -> tuple[Any, Any]:
    key, value = layer[:2]
    key_index = torch.tensor(positions, dtype=torch.long, device=key.device)
    value_index = key_index.to(device=value.device)
    return (
        torch.index_select(key, -2, key_index),
        torch.index_select(value, -2, value_index),
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class OfficialCriterion:
    """Thin observer around the pinned, unmodified official ``PrefixKV`` class."""

    def __init__(
        self,
        official_class: type[Any],
        *,
        work_directory: Path,
        forget_ratio: float,
        layer_count: int,
    ) -> None:
        self._work_directory = work_directory
        self._inner = official_class(
            model_name="llava-v1.5-7b",
            start_size=1,
            recent_size=2047,
            k_seq_dim=2,
            v_seq_dim=2,
            ratio=forget_ratio,
            distance=-25,
            layer_num=layer_count,
            profile=False,
        )
        self.initial_layer_sizes = ()
        self.initial_active_bytes = 0
        self.final_active_bytes = 0
        self.selected_positions = ()

    def _derive_selected_positions(self, source_lengths: tuple[int, ...]) -> None:
        ratios = np.asarray(self._inner.ratios, dtype=np.float64)
        forget = np.rint(ratios * np.asarray(source_lengths)).astype(np.int32)
        selected: list[tuple[int, ...]] = []
        for layer_index, (length, removed) in enumerate(zip(source_lengths, forget, strict=True)):
            middle = torch.argsort(self._inner.score_sum[layer_index, :, 1 : length - 1], dim=-1)[
                :, int(removed) :
            ]
            middle = (middle + 1).sort().values
            full = torch.cat(
                (
                    torch.zeros((1, 1), dtype=middle.dtype, device=middle.device),
                    middle,
                    torch.full((1, 1), length - 1, dtype=middle.dtype, device=middle.device),
                ),
                dim=-1,
            )
            selected.append(tuple(int(value) for value in full[0].detach().cpu().tolist()))
        self.selected_positions = tuple(selected)

    def __call__(self, past_key_values: Any, num_of_token: int, attentions: Any) -> Any:
        first = not self.initial_layer_sizes
        source_lengths = _layer_sizes(past_key_values)
        with _working_directory(self._work_directory):
            result = self._inner(past_key_values, num_of_token, attentions)
        if first:
            self._derive_selected_positions(source_lengths)
            self.initial_layer_sizes = _layer_sizes(result)
            self.initial_active_bytes = _tensor_bytes(result)
        self.final_active_bytes = _tensor_bytes(result)
        return result


class ReimplementationCriterion:
    """Adapt the common MosaicKV PrefixKV plan to the official callback protocol."""

    def __init__(
        self,
        *,
        profile_path: Path,
        profile: Any,
        retention_ratio: float,
        layer_count: int,
    ) -> None:
        self._profile_path = profile_path
        self._profile = profile
        self._retention_ratio = retention_ratio
        self._layer_count = layer_count
        self._plan: Any | None = None
        self.initial_layer_sizes = ()
        self.initial_active_bytes = 0
        self.final_active_bytes = 0
        self.selected_positions = ()

    def _prefill(self, cache: Any, attentions: Any) -> Any:
        source_length = int(cache[0][0].shape[-2])
        full_state = FullKVState.from_tensors(
            tuple((layer[0], layer[1]) for layer in cache),
            block_size=1,
            sequence_dimension=-2,
            head_dimension=-3,
            mandatory_logical_positions=(0, source_length - 1),
            source_class=tuple,
            source_kind="legacy_tuple",
            cached_key_state="post_rope",
        )
        source_slots = len(full_state.blocks)
        cache_config = CacheConfig(
            budget_value=math.floor(source_slots * self._retention_ratio),
            budget_unit=BudgetUnit.BLOCKS,
            retention_ratio=self._retention_ratio,
            block_size=1,
        )
        method_config = PrefixKVConfig(
            enabled=True,
            profile_mode=PrefixKVProfileMode.OFFLINE_PROFILE,
            profile_path=str(self._profile_path),
            start_size=1,
            protect_size=1,
            eviction_distance=-25,
            official_repository_sha=OFFICIAL_SHA,
        )
        self._plan = build_prefixkv_reimpl_plan(
            full_state,
            tuple(attentions),
            method_config,
            cache_config,
            model_id=MODEL_ID,
            model_revision=MODEL_REPOSITORY_SHA,
            profile=self._profile,
        )
        self.initial_layer_sizes = self._plan.layer_cache_sizes
        self.selected_positions = tuple(
            layer.selected_physical_positions for layer in self._plan.layers
        )
        result = tuple(
            _gather_layer(layer, positions)
            for layer, positions in zip(cache, self.selected_positions, strict=True)
        )
        self.initial_active_bytes = _tensor_bytes(result)
        self.final_active_bytes = self.initial_active_bytes
        return result

    def _decode(self, cache: Any, logical_length: int) -> Any:
        assert self._plan is not None
        result: list[tuple[Any, Any]] = []
        for layer, layer_plan in zip(cache, self._plan.layers, strict=True):
            length = int(layer[0].shape[-2])
            target = logical_length * layer_plan.retention_ratio
            should_evict = math.trunc(length - target) > 0
            if should_evict and length > 1:
                offset = min(layer_plan.eviction_offset, length - 1)
                keep = tuple(index for index in range(length) if index != offset)
                result.append(_gather_layer(layer, keep))
            else:
                result.append((layer[0], layer[1]))
        packed = tuple(result)
        self.final_active_bytes = _tensor_bytes(packed)
        return packed

    def __call__(self, past_key_values: Any, num_of_token: int, attentions: Any) -> Any:
        if self._plan is None:
            if len(past_key_values) != self._layer_count:
                raise RuntimeError("PrefixKV cache layer count changed")
            return self._prefill(past_key_values, attentions)
        return self._decode(past_key_values, num_of_token)


def _prepare_sample(
    item: dict[str, Any],
    *,
    tokenizer: Any,
    image_processor: Any,
    media_root: Path,
    conv_templates: Any,
    tokenizer_image_token: Any,
    image_token_index: int,
    process_images: Any,
    model: Any,
) -> tuple[str, Any, Any, str]:
    from PIL import Image

    question = str(item["conversations"][0]["value"])
    answer = str(item["conversations"][1]["value"])
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    image = Image.open(media_root / str(item["image"])).convert("RGB")
    image_args = SimpleNamespace(image_aspect_ratio="pad")
    image_tensor = process_images([image], image_processor, image_args).to(
        model.device, dtype=torch.bfloat16
    )
    return prompt, input_ids, image_tensor, answer


def _timed_generate(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    image_tensor: Any,
    *,
    criterion: CacheCriterion | None,
    max_new_tokens: int,
) -> tuple[tuple[int, ...], str, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
            use_cache=True,
            kv_cache_criteria=criterion,
        )
    end.record()
    torch.cuda.synchronize()
    latency = float(start.elapsed_time(end)) / 1000.0
    generated = tuple(
        int(value) for value in output_ids[0, input_ids.shape[-1] :].detach().cpu().tolist()
    )
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return generated, answer, latency


def _teacher_forced_ppl(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    image_tensor: Any,
    answer: str,
    *,
    criterion: CacheCriterion,
    token_limit: int,
) -> float:
    answer_ids = tokenizer.encode(answer, return_tensors="pt").to(model.device)[:, 1:]
    answer_ids = answer_ids[:, :token_limit]
    if answer_ids.shape[-1] == 0:
        raise RuntimeError("teacher-forced answer has no tokens")
    losses: list[Any] = []
    cache = None
    logical_length = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.inference_mode():
        for index in range(int(answer_ids.shape[-1])):
            if cache is None:
                output = model(
                    input_ids,
                    images=image_tensor,
                    past_key_values=None,
                    use_cache=True,
                    output_attentions=True,
                )
            else:
                output = model(
                    answer_ids[:, index - 1 : index],
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=True,
                )
            logits = output.logits.reshape(-1, model.config.vocab_size)
            logical_length += int(logits.shape[0])
            label = answer_ids[:, index : index + 1].reshape(-1)
            losses.append(loss_fn(logits[-1:].float(), label))
            cache = criterion(output.past_key_values, logical_length, output.attentions)
    return float(torch.exp(torch.stack(losses).mean()).detach().cpu().item())


def _new_criterion(
    kind: str,
    *,
    official_class: type[Any],
    work_directory: Path,
    forget_ratio: float,
    retention_ratio: float,
    layer_count: int,
    profile_path: Path,
    profile: Any,
) -> CacheCriterion:
    if kind == "official":
        return OfficialCriterion(
            official_class,
            work_directory=work_directory,
            forget_ratio=forget_ratio,
            layer_count=layer_count,
        )
    return ReimplementationCriterion(
        profile_path=profile_path,
        profile=profile,
        retention_ratio=retention_ratio,
        layer_count=layer_count,
    )


def _software_versions() -> dict[str, str]:
    import accelerate
    import tokenizers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--calibration-index", type=int, default=1)
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ppl-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official PrefixKV parity requires CUDA")
    if args.calibration_index == args.evaluation_index:
        raise ValueError("calibration and evaluation indices must differ")
    if not 0 < args.retention_ratio <= 1:
        raise ValueError("retention ratio must be in (0, 1]")

    root = args.repository_root.resolve()
    official_root = root / "third_party" / "PrefixKV"
    if _git(official_root, "rev-parse", "HEAD") != OFFICIAL_SHA:
        raise RuntimeError("third_party/PrefixKV is not at the audited commit")
    sys.path.insert(0, str(official_root))
    from cache_generate import generate, greedy_search, sample
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from patch_attention_forward import patch_llama_attention_forward
    from prefixkv import PrefixKV as OfficialPrefixKV

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = args.output_directory / "official_workspace"
    (work_directory / "confs").mkdir(parents=True, exist_ok=True)

    records = json.loads(args.data_path.read_text(encoding="utf-8"))
    calibration_item = cast("dict[str, Any]", records[args.calibration_index])
    evaluation_item = cast("dict[str, Any]", records[args.evaluation_index])
    calibration_id = str(calibration_item["id"])
    evaluation_id = str(evaluation_item["id"])
    if calibration_id == evaluation_id:
        raise RuntimeError("calibration/evaluation sample ID overlap")

    patch_llama_attention_forward()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path), None, "llava-v1.5-7b", False, False, device="cuda"
    )
    model.eval()
    model.generate = types.MethodType(generate, model)
    model.greedy_search = types.MethodType(greedy_search, model)
    model.sample = types.MethodType(sample, model)
    layer_count = int(model.config.num_hidden_layers)

    _, calibration_ids, calibration_image, _ = _prepare_sample(
        calibration_item,
        tokenizer=tokenizer,
        image_processor=image_processor,
        media_root=args.media_root,
        conv_templates=conv_templates,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=IMAGE_TOKEN_INDEX,
        process_images=process_images,
        model=model,
    )
    with torch.inference_mode():
        calibration_output = model(
            calibration_ids,
            images=calibration_image,
            use_cache=True,
            output_attentions=True,
        )
    observation = prefixkv_calibration_observation(
        calibration_id, tuple(calibration_output.attentions)
    )
    profile = generate_prefixkv_profile(
        (observation,),
        model_id=MODEL_ID,
        model_revision=MODEL_REPOSITORY_SHA,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        calibration_split="detail_1k_calibration",
        evaluation_sample_ids=(evaluation_id,),
        retention_ratio=args.retention_ratio,
        seed=args.seed,
        start_size=1,
        protect_size=1,
    )
    profile.assert_evaluation_disjoint((evaluation_id,))
    profile_path = profile.write(args.output_directory / "profile.json")
    forget_ratio = 1.0 - args.retention_ratio
    official_profile_path = work_directory / "confs" / f"prefixkv_llava-v1.5-7b_{forget_ratio}.json"
    official_profile_path.write_text(
        json.dumps(list(profile.layer_forget_ratios)) + "\n", encoding="utf-8"
    )
    del calibration_output, calibration_image, calibration_ids
    gc.collect()
    torch.cuda.empty_cache()

    evaluation_prompt, evaluation_ids, evaluation_image, reference_answer = _prepare_sample(
        evaluation_item,
        tokenizer=tokenizer,
        image_processor=image_processor,
        media_root=args.media_root,
        conv_templates=conv_templates,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=IMAGE_TOKEN_INDEX,
        process_images=process_images,
        model=model,
    )

    full_tokens, full_answer, full_latency = _timed_generate(
        model,
        tokenizer,
        evaluation_ids,
        evaluation_image,
        criterion=None,
        max_new_tokens=args.max_new_tokens,
    )
    observations: dict[str, dict[str, Any]] = {}
    selected_positions: dict[str, list[list[int]]] = {}
    for kind in ("official", "reimplementation"):
        generation_criterion = _new_criterion(
            kind,
            official_class=OfficialPrefixKV,
            work_directory=work_directory,
            forget_ratio=forget_ratio,
            retention_ratio=args.retention_ratio,
            layer_count=layer_count,
            profile_path=profile_path,
            profile=profile,
        )
        tokens, answer, latency = _timed_generate(
            model,
            tokenizer,
            evaluation_ids,
            evaluation_image,
            criterion=generation_criterion,
            max_new_tokens=args.max_new_tokens,
        )
        ppl_criterion = _new_criterion(
            kind,
            official_class=OfficialPrefixKV,
            work_directory=work_directory,
            forget_ratio=forget_ratio,
            retention_ratio=args.retention_ratio,
            layer_count=layer_count,
            profile_path=profile_path,
            profile=profile,
        )
        perplexity = _teacher_forced_ppl(
            model,
            tokenizer,
            evaluation_ids,
            evaluation_image,
            reference_answer,
            criterion=ppl_criterion,
            token_limit=args.ppl_tokens,
        )
        rouge_l = Rouge().get_scores(answer, full_answer)[0]["rouge-l"]["f"]
        observations[kind] = {
            "sample_id": evaluation_id,
            "per_layer_cache_sizes": generation_criterion.initial_layer_sizes,
            "total_retained_bytes": generation_criterion.initial_active_bytes,
            "actual_active_kv_bytes": generation_criterion.initial_active_bytes,
            "generated_answer": answer,
            "generated_token_ids": tokens,
            "latency_seconds": latency,
            "perplexity": perplexity,
            "rouge_l_f1": float(rouge_l),
            "final_active_kv_bytes": generation_criterion.final_active_bytes,
            "rouge_l_f1_human_reference": float(
                Rouge().get_scores(answer, reference_answer)[0]["rouge-l"]["f"]
            ),
        }
        selected_positions[kind] = [
            list(layer) for layer in generation_criterion.selected_positions
        ]
        gc.collect()
        torch.cuda.empty_cache()

    environment_freeze = _command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    (args.output_directory / "environment.lock.txt").write_text(
        environment_freeze + "\n", encoding="utf-8"
    )
    gpu_query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_processes = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    hardware_payload = {"gpu_query": gpu_query, "visible_devices": torch.cuda.device_count()}
    protocol_payload = {
        "runner": "run_prefixkv_llava_parity.py",
        "upstream_attention_patch_sha256": _sha256_file(
            official_root / "patch_attention_forward.py"
        ),
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": None,
        },
        "ppl_tokens": args.ppl_tokens,
        "timing": "torch.cuda.Event with synchronize before and after",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        "calibration_index": args.calibration_index,
        "evaluation_index": args.evaluation_index,
    }
    source_length = observations["reimplementation"]["per_layer_cache_sizes"]
    target_slots = sum(int(value) for value in source_length) * int(
        model.config.num_attention_heads
    )
    controls = PrefixKVParityControls(
        model_id=MODEL_ID,
        model_revision=MODEL_REPOSITORY_SHA,
        tokenizer_revision=MODEL_REPOSITORY_SHA,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        calibration_sample_set_sha256=_sha256_json([calibration_id]),
        evaluation_sample_set_sha256=_sha256_json([evaluation_id]),
        prompt_payload_sha256=_sha256_json(
            {
                "prompt": evaluation_prompt,
                "input_ids": evaluation_ids.detach().cpu().tolist(),
            }
        ),
        media_payload_sha256=_sha256_file(args.media_root / str(evaluation_item["image"])),
        profile_sha256=profile.profile_sha256,
        environment_sha256=_sha256_bytes(environment_freeze.encode("utf-8")),
        hardware_sha256=_sha256_json(hardware_payload),
        measurement_protocol_sha256=_sha256_json(protocol_payload),
        cache_budget_value=target_slots,
        cache_budget_unit="blocks",
        block_size=1,
        retention_ratio=args.retention_ratio,
        official_forget_ratio=forget_ratio,
        generation_parameters=cast("dict[str, Any]", protocol_payload["generation"]),
        output_length_policy="fixed_max_new_tokens_eos_disabled",
        model_precision="bfloat16",
        backend="hf_legacy_llava_transformers_4.31",
        backend_configuration={
            "attention_implementation": "official_eager_patch",
            "device_map": "auto_single_visible_gpu",
            "use_cache": True,
        },
        attention_implementation="eager",
        seed=args.seed,
    )

    root_sha = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    config_payload = {
        "controls": asdict(controls),
        "profile": profile.to_json_object(),
        "protocol": protocol_payload,
    }
    config_sha = _sha256_json(config_payload)
    manifest_path = args.output_directory / "manifest.json"
    manifest = {
        "schema_version": 1,
        "run_id": args.output_directory.name,
        "paper_eligible": not dirty,
        "paper_ineligibility_reason": "dirty_worktree" if dirty else None,
        "git_sha": root_sha,
        "git_dirty": dirty,
        "config_sha256": config_sha,
        "official_repository": {
            "path": "third_party/PrefixKV",
            "commit_sha": OFFICIAL_SHA,
            "license": "MIT",
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REPOSITORY_SHA,
            "weight_index_sha256": _sha256_file(args.model_path / "pytorch_model.bin.index.json"),
            "precision": "bfloat16",
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "file_sha256": _sha256_file(args.data_path),
            "calibration_sample_ids": [calibration_id],
            "evaluation_sample_ids": [evaluation_id],
            "intersection": [],
        },
        "media": {
            "calibration_sha256": _sha256_file(args.media_root / str(calibration_item["image"])),
            "evaluation_sha256": _sha256_file(args.media_root / str(evaluation_item["image"])),
        },
        "profile": {
            "native_path": str(profile_path),
            "native_sha256": profile.profile_sha256,
            "official_raw_path": str(official_profile_path),
            "official_raw_file_sha256": _sha256_file(official_profile_path),
        },
        "software": _software_versions(),
        "cuda": {
            "driver_and_clocks": gpu_query,
            "processes_at_manifest_capture": gpu_processes,
        },
        "gpu": {
            "type": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        },
        "backend": "hf_legacy_llava_transformers_4.31",
        "attention_implementation": "official_eager_patch",
        "seed": args.seed,
        "measurement_types": [
            "baseline_official_measured",
            "baseline_reimpl_measured",
        ],
        "protocol": protocol_payload,
        "reference": {
            "generated_token_ids": full_tokens,
            "generated_answer": full_answer,
            "latency_seconds": full_latency,
        },
        "selected_positions": selected_positions,
        "extended_observations": observations,
        "created_unix_seconds": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifacts: dict[str, PrefixKVParityArtifact] = {}
    for kind, implementation, executable_sha, measurement_type in (
        ("official", "official_prefixkv", OFFICIAL_SHA, "baseline_official_measured"),
        ("reimplementation", "prefixkv_reimpl", root_sha, "baseline_reimpl_measured"),
    ):
        row = dict(observations[kind])
        row.pop("final_active_kv_bytes")
        row.pop("rouge_l_f1_human_reference")
        artifact = PrefixKVParityArtifact(
            implementation=implementation,
            official_repository_sha=OFFICIAL_SHA,
            executable_git_sha=executable_sha,
            config_sha256=config_sha,
            manifest_path=str(manifest_path),
            measurement_type=measurement_type,
            controls=controls,
            samples=(PrefixKVSampleObservation(**row),),
        )
        artifacts[kind] = artifact
        (args.output_directory / f"{kind}.json").write_text(
            json.dumps(artifact.to_json_object(), indent=2, sort_keys=True) + "\n"
        )
    comparison = compare_prefixkv_artifacts(artifacts["official"], artifacts["reimplementation"])
    (args.output_directory / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
