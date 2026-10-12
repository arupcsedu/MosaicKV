#!/usr/bin/env python3
"""Run a controlled PrefixKV reproduction or parity experiment.

The runner imports the pinned upstream cache implementation without changing
the submodule.  Both implementations share one legacy LLaVA checkpoint, one
model instance, one generated offline profile, and the upstream explicit
generation loop.  ``--official-profile-reproduction`` is deliberately a
different lane: it executes an upstream-shipped profile byte-for-byte on the
paper's 1,000-example evaluation file.  Upstream did not publish the sample
identities used to make that profile, so that lane is an official-code
reproduction, not evidence that a locally calibrated reimplementation is
equivalent.  Outputs are non-canonical whenever the enclosing MosaicKV
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


def _load_official_profile(path: Path, *, expected_layers: int | None = None) -> tuple[float, ...]:
    """Load an unmodified upstream PrefixKV profile and reject ambiguous input.

    The profile stores per-layer *forget* ratios.  It is intentionally not
    converted to MosaicKV's native profile format: this preserves the exact
    operating point distributed by the PrefixKV authors.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read official PrefixKV profile {path}: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("official PrefixKV profile must be a non-empty JSON list")
    if any(not isinstance(value, (float, int)) or isinstance(value, bool) for value in raw):
        raise ValueError("official PrefixKV profile must contain numeric forget ratios")
    ratios = tuple(float(value) for value in raw)
    if any(not 0.0 <= value <= 1.0 for value in ratios):
        raise ValueError("official PrefixKV forget ratios must be in [0, 1]")
    if expected_layers is not None and len(ratios) != expected_layers:
        raise ValueError(
            "official PrefixKV profile layer count does not match loaded model: "
            f"{len(ratios)} != {expected_layers}"
        )
    return ratios


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
) -> tuple[float, float, int]:
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
    nll_sum = float(torch.stack(losses).sum().detach().cpu().item())
    token_count = len(losses)
    return math.exp(nll_sum / token_count), nll_sum, token_count


def _parse_indices(value: str, *, field: str) -> tuple[int, ...]:
    """Parse a deterministic, explicit comma-separated index set."""

    parsed: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as error:
            raise ValueError(f"{field} must contain comma-separated integers") from error
        if index < 0:
            raise ValueError(f"{field} cannot contain negative indices")
        parsed.append(index)
    if not parsed:
        raise ValueError(f"{field} cannot be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{field} contains duplicate indices")
    return tuple(sorted(parsed))


def _load_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Load one explicit LLaVA-Description JSON source without implicit splits."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} must contain a non-empty JSON array")
    records = [cast("dict[str, Any]", item) for item in payload]
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"{label} contains a non-object record")
    for item in records:
        if not isinstance(item.get("id"), (str, int)) or not item.get("image"):
            raise ValueError(f"{label} record lacks an id or image")
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"{label} record lacks the required two-turn conversation")
    return records


def _select_calibration_items(
    records: list[dict[str, Any]],
    indices: tuple[int, ...],
    *,
    enabled: bool,
) -> tuple[dict[str, Any], ...]:
    """Select calibration rows only in a mode that has a calibration source.

    Official-profile reproduction deliberately has no recoverable calibration
    identities.  Its diagnostic default index must not be interpreted as an
    index into an empty synthetic calibration collection.
    """

    if not enabled:
        return ()
    if any(index >= len(records) for index in indices):
        raise IndexError("a calibration index is outside calibration data")
    return tuple(records[index] for index in indices)


def _parity_gate(
    comparison: dict[str, Any],
    *,
    minimum_token_agreement: float,
    maximum_relative_ppl_delta: float,
    maximum_rouge_l_delta: float,
    maximum_retained_byte_delta: int,
) -> dict[str, Any]:
    """Evaluate preregistered official-versus-reimplementation parity limits.

    The checks use only generated tokens, cache bytes, and frozen-reference
    metrics.  They never inspect benchmark labels or select a subset after
    observing outcomes.
    """

    failures: list[str] = []
    if comparison.get("status") != "comparable":
        failures.append("controls_or_sample_ids_not_comparable")
        return {"passed": False, "failures": failures, "checked_samples": 0}
    rows = comparison.get("samples")
    if not isinstance(rows, list) or not rows:
        failures.append("no_comparable_samples")
        return {"passed": False, "failures": failures, "checked_samples": 0}
    for row in rows:
        if not isinstance(row, dict):
            failures.append("malformed_parity_row")
            continue
        sample_id = str(row.get("sample_id", "unknown"))
        if float(row.get("token_agreement", -1.0)) < minimum_token_agreement:
            failures.append(f"token_agreement:{sample_id}")
        if abs(int(row.get("retained_byte_delta", 0))) > maximum_retained_byte_delta:
            failures.append(f"retained_bytes:{sample_id}")
        rouge_delta = row.get("rouge_l_f1_delta")
        if rouge_delta is None or abs(float(rouge_delta)) > maximum_rouge_l_delta:
            failures.append(f"rouge_l:{sample_id}")
    corpus = comparison.get("corpus")
    if not isinstance(corpus, dict):
        failures.append("missing_corpus_metrics")
    else:
        official_ppl = corpus.get("official_teacher_forced_ppl")
        reimpl_ppl = corpus.get("reimpl_teacher_forced_ppl")
        if (
            official_ppl is None
            or reimpl_ppl is None
            or float(official_ppl) <= 0
            or abs(float(reimpl_ppl) - float(official_ppl)) / float(official_ppl)
            > maximum_relative_ppl_delta
        ):
            failures.append("corpus_teacher_forced_ppl")
    return {
        "passed": not failures,
        "failures": failures,
        "checked_samples": len(rows),
        "thresholds": {
            "minimum_token_agreement": minimum_token_agreement,
            "maximum_relative_ppl_delta": maximum_relative_ppl_delta,
            "maximum_rouge_l_delta": maximum_rouge_l_delta,
            "maximum_retained_byte_delta": maximum_retained_byte_delta,
        },
    }


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
    parser.add_argument(
        "--calibration-data-path",
        type=Path,
        help="Separate, immutable calibration JSON. Required in --paper-reproduction mode.",
    )
    parser.add_argument(
        "--calibration-media-root",
        type=Path,
        help="Media root for --calibration-data-path; defaults to --media-root outside paper mode.",
    )
    parser.add_argument(
        "--calibration-dataset-id",
        help="Recorded calibration dataset ID; required in --paper-reproduction mode.",
    )
    parser.add_argument(
        "--calibration-dataset-revision",
        help="SHA-256 of --calibration-data-path; required in --paper-reproduction mode.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--calibration-indices",
        help="Comma-separated calibration indices. Overrides --calibration-index.",
    )
    parser.add_argument(
        "--evaluation-indices",
        help="Comma-separated evaluation indices. Overrides --evaluation-index.",
    )
    parser.add_argument("--calibration-index", type=int, default=1)
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--ppl-tokens",
        type=int,
        default=136,
        help="Maximum teacher-forced continuation tokens (paper-native default: 136).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paper-reproduction",
        action="store_true",
        help=(
            "require the registered PrefixKV reproduction contract: a separate calibration "
            "source, >=10 calibration samples, exactly 1000 evaluation samples, and a "
            "passing official-versus-reimplementation parity gate"
        ),
    )
    parser.add_argument(
        "--official-profile-reproduction",
        action="store_true",
        help=(
            "execute only the unmodified official PrefixKV implementation with an "
            "upstream-shipped profile on the registered 1,000-example evaluation "
            "set. This is an official-code reproduction lane; it is not a local "
            "reimplementation-parity claim because upstream profile calibration "
            "identities were not released."
        ),
    )
    parser.add_argument(
        "--official-profile-path",
        type=Path,
        help="Pinned upstream raw per-layer forget-ratio JSON required by official-profile mode.",
    )
    parser.add_argument(
        "--official-profile-shard",
        action="store_true",
        help=(
            "allow a strict subset of the 1,000 official evaluation identities. A shard is "
            "never paper eligible by itself and must be validated by a hash-bound union "
            "aggregator before it can contribute to the official-profile reproduction."
        ),
    )
    parser.add_argument("--minimum-token-agreement", type=float, default=0.99)
    parser.add_argument("--maximum-relative-ppl-delta", type=float, default=0.05)
    parser.add_argument("--maximum-rouge-l-delta", type=float, default=0.05)
    parser.add_argument("--maximum-retained-byte-delta", type=int, default=16_384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official PrefixKV parity requires CUDA")
    if not 0 < args.retention_ratio <= 1:
        raise ValueError("retention ratio must be in (0, 1]")
    if args.ppl_tokens < 1:
        raise ValueError("--ppl-tokens must be positive")
    if not 0 <= args.minimum_token_agreement <= 1:
        raise ValueError("--minimum-token-agreement must be in [0, 1]")
    if args.maximum_relative_ppl_delta < 0 or args.maximum_rouge_l_delta < 0:
        raise ValueError("PrefixKV parity tolerances must be nonnegative")
    if args.maximum_retained_byte_delta < 0:
        raise ValueError("--maximum-retained-byte-delta must be nonnegative")
    if args.calibration_media_root is not None and args.calibration_data_path is None:
        raise ValueError("--calibration-media-root requires --calibration-data-path")
    if args.paper_reproduction and args.official_profile_reproduction:
        raise ValueError(
            "--paper-reproduction and --official-profile-reproduction are mutually exclusive"
        )
    if args.official_profile_reproduction and args.official_profile_path is None:
        raise ValueError("--official-profile-reproduction requires --official-profile-path")
    if args.official_profile_shard and not args.official_profile_reproduction:
        raise ValueError("--official-profile-shard requires --official-profile-reproduction")
    if args.paper_reproduction and (
        args.calibration_data_path is None
        or args.calibration_media_root is None
        or not args.calibration_dataset_id
        or not args.calibration_dataset_revision
    ):
        raise ValueError(
            "--paper-reproduction requires calibration data/media paths and calibration "
            "dataset ID/revision"
        )

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

    records = _load_records(args.data_path, label="evaluation data-path")
    official_profile_source: Path | None = None
    if args.official_profile_reproduction:
        assert args.official_profile_path is not None
        official_profile_source = args.official_profile_path.resolve()
        allowed_profiles = (official_root / "confs").resolve()
        try:
            official_profile_source.relative_to(allowed_profiles)
        except ValueError as error:
            raise ValueError(
                "--official-profile-path must be an unmodified file below "
                "third_party/PrefixKV/confs"
            ) from error
        if not official_profile_source.is_file():
            raise ValueError(f"official PrefixKV profile does not exist: {official_profile_source}")
        expected_profile_name = (
            f"prefixkv_llava-v1.5-7b_{1.0 - args.retention_ratio}.json"
        )
        if official_profile_source.name != expected_profile_name:
            raise ValueError(
                "official PrefixKV profile does not match requested retention ratio: "
                f"expected {expected_profile_name}, got {official_profile_source.name}"
            )
        _load_official_profile(official_profile_source)

    calibration_source = None if args.official_profile_reproduction else (
        args.calibration_data_path or args.data_path
    )
    calibration_records = (
        _load_records(calibration_source, label="calibration data-path")
        if calibration_source is not None
        else []
    )
    calibration_media_root = args.calibration_media_root or args.media_root
    calibration_indices = (
        ()
        if calibration_source is None
        else (
            _parse_indices(args.calibration_indices, field="--calibration-indices")
            if args.calibration_indices is not None
            else (args.calibration_index,)
        )
    )
    evaluation_indices = (
        _parse_indices(args.evaluation_indices, field="--evaluation-indices")
        if args.evaluation_indices is not None
        else (args.evaluation_index,)
    )
    same_calibration_source = (
        calibration_source is not None and calibration_source.resolve() == args.data_path.resolve()
    )
    if same_calibration_source and set(calibration_indices).intersection(evaluation_indices):
        raise ValueError("calibration and evaluation indices must be disjoint")
    if any(index >= len(records) for index in evaluation_indices):
        raise IndexError("an evaluation index is outside evaluation data")
    calibration_items = _select_calibration_items(
        calibration_records,
        calibration_indices,
        enabled=calibration_source is not None,
    )
    evaluation_items = tuple(records[index] for index in evaluation_indices)
    calibration_ids = tuple(str(item["id"]) for item in calibration_items)
    evaluation_ids = tuple(str(item["id"]) for item in evaluation_items)
    if len(set((*calibration_ids, *evaluation_ids))) != len(calibration_ids) + len(evaluation_ids):
        raise RuntimeError("calibration/evaluation sample IDs overlap or are not unique")
    observed_dataset_revision = _sha256_file(args.data_path)
    if observed_dataset_revision != DATASET_REVISION:
        raise RuntimeError(
            "LLaVA-Description file digest differs from the registered paper-native revision: "
            f"expected {DATASET_REVISION}, observed {observed_dataset_revision}"
        )
    observed_calibration_revision = (
        _sha256_file(calibration_source) if calibration_source is not None else None
    )
    if args.paper_reproduction:
        assert calibration_source is not None
        if calibration_source.resolve() == args.data_path.resolve():
            raise RuntimeError("paper reproduction requires a separate calibration source")
        if len(calibration_items) < 10:
            raise RuntimeError("paper reproduction requires at least 10 calibration samples")
        if len(evaluation_items) != 1000:
            raise RuntimeError("paper reproduction requires exactly 1000 evaluation samples")
        if args.calibration_dataset_revision != observed_calibration_revision:
            raise RuntimeError(
                "calibration dataset revision must equal the SHA-256 of calibration data"
            )
    if (
        args.official_profile_reproduction
        and not args.official_profile_shard
        and len(evaluation_items) != 1000
    ):
        raise RuntimeError(
            "official-profile reproduction requires exactly 1000 LLaVA-Description examples"
        )

    patch_llama_attention_forward()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path), None, "llava-v1.5-7b", False, False, device="cuda"
    )
    model.eval()
    model.generate = types.MethodType(generate, model)
    model.greedy_search = types.MethodType(greedy_search, model)
    model.sample = types.MethodType(sample, model)
    layer_count = int(model.config.num_hidden_layers)

    calibration_media_hashes: dict[str, str] = {}
    profile: Any | None = None
    profile_path: Path | None = None
    if args.official_profile_reproduction:
        assert official_profile_source is not None
        official_ratios = _load_official_profile(
            official_profile_source, expected_layers=layer_count
        )
        # The upstream class opens a fixed relative ``confs/`` file.  Preserve
        # its bytes, rather than reserializing its ratios through local code.
        forget_ratio = 1.0 - args.retention_ratio
        official_profile_path = (
            work_directory / "confs" / f"prefixkv_llava-v1.5-7b_{forget_ratio}.json"
        )
        official_profile_path.write_bytes(official_profile_source.read_bytes())
        profile_sha256 = _sha256_file(official_profile_source)
        profile_payload: dict[str, Any] = {
            "kind": "official_upstream_raw_profile",
            "source_path": str(official_profile_source),
            "source_sha256": profile_sha256,
            "layer_forget_ratios": list(official_ratios),
            "calibration_provenance": "unverifiable_upstream_profile_ids_not_released",
        }
        execution_kinds = ("official",)
    else:
        calibration_observations = []
        for calibration_item in calibration_items:
            calibration_id = str(calibration_item["id"])
            _, calibration_input_ids, calibration_image, _ = _prepare_sample(
                calibration_item,
                tokenizer=tokenizer,
                image_processor=image_processor,
                media_root=calibration_media_root,
                conv_templates=conv_templates,
                tokenizer_image_token=tokenizer_image_token,
                image_token_index=IMAGE_TOKEN_INDEX,
                process_images=process_images,
                model=model,
            )
            with torch.inference_mode():
                calibration_output = model(
                    calibration_input_ids,
                    images=calibration_image,
                    use_cache=True,
                    output_attentions=True,
                )
            calibration_observations.append(
                prefixkv_calibration_observation(
                    calibration_id, tuple(calibration_output.attentions)
                )
            )
            calibration_media_hashes[calibration_id] = _sha256_file(
                calibration_media_root / str(calibration_item["image"])
            )
            del calibration_output, calibration_image, calibration_input_ids
            gc.collect()
            torch.cuda.empty_cache()
        profile = generate_prefixkv_profile(
            tuple(calibration_observations),
            model_id=MODEL_ID,
            model_revision=MODEL_REPOSITORY_SHA,
            dataset_id=(args.calibration_dataset_id or DATASET_ID),
            dataset_revision=(args.calibration_dataset_revision or observed_calibration_revision),
            calibration_split="detail_1k_calibration",
            evaluation_sample_ids=evaluation_ids,
            retention_ratio=args.retention_ratio,
            seed=args.seed,
            start_size=1,
            protect_size=1,
        )
        profile.assert_evaluation_disjoint(evaluation_ids)
        profile_path = profile.write(args.output_directory / "profile.json")
        forget_ratio = 1.0 - args.retention_ratio
        official_profile_path = (
            work_directory / "confs" / f"prefixkv_llava-v1.5-7b_{forget_ratio}.json"
        )
        official_profile_path.write_text(
            json.dumps(list(profile.layer_forget_ratios)) + "\n", encoding="utf-8"
        )
        profile_sha256 = profile.profile_sha256
        profile_payload = profile.to_json_object()
        execution_kinds = ("official", "reimplementation")

    observations: dict[str, list[dict[str, Any]]] = {kind: [] for kind in execution_kinds}
    selected_positions: dict[str, dict[str, list[list[int]]]] = {
        kind: {} for kind in execution_kinds
    }
    reference_rows: list[dict[str, Any]] = []
    prompt_payloads: list[dict[str, Any]] = []
    evaluation_media_hashes: dict[str, str] = {}
    for evaluation_item in evaluation_items:
        evaluation_id = str(evaluation_item["id"])
        (
            evaluation_prompt,
            evaluation_input_ids,
            evaluation_image,
            reference_answer,
        ) = _prepare_sample(
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
            evaluation_input_ids,
            evaluation_image,
            criterion=None,
            max_new_tokens=args.max_new_tokens,
        )
        reference_rows.append(
            {
                "sample_id": evaluation_id,
                "generated_token_ids": full_tokens,
                "generated_answer": full_answer,
                "latency_seconds": full_latency,
            }
        )
        prompt_payloads.append(
            {
                "sample_id": evaluation_id,
                "prompt": evaluation_prompt,
                "input_ids": evaluation_input_ids.detach().cpu().tolist(),
            }
        )
        evaluation_media_hashes[evaluation_id] = _sha256_file(
            args.media_root / str(evaluation_item["image"])
        )
        for kind in execution_kinds:
            generation_criterion = _new_criterion(
                kind,
                official_class=OfficialPrefixKV,
                work_directory=work_directory,
                forget_ratio=forget_ratio,
                retention_ratio=args.retention_ratio,
                layer_count=layer_count,
                profile_path=profile_path or official_profile_path,
                profile=profile,
            )
            tokens, answer, latency = _timed_generate(
                model,
                tokenizer,
                evaluation_input_ids,
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
                profile_path=profile_path or official_profile_path,
                profile=profile,
            )
            perplexity, nll_sum, token_count = _teacher_forced_ppl(
                model,
                tokenizer,
                evaluation_input_ids,
                evaluation_image,
                reference_answer,
                criterion=ppl_criterion,
                token_limit=args.ppl_tokens,
            )
            rouge_l = Rouge().get_scores(answer, full_answer)[0]["rouge-l"]["f"]
            observations[kind].append(
                {
                    "sample_id": evaluation_id,
                    "per_layer_cache_sizes": generation_criterion.initial_layer_sizes,
                    "total_retained_bytes": generation_criterion.initial_active_bytes,
                    "actual_active_kv_bytes": generation_criterion.initial_active_bytes,
                    "generated_answer": answer,
                    "generated_token_ids": tokens,
                    "latency_seconds": latency,
                    "perplexity": perplexity,
                    "nll_sum": nll_sum,
                    "evaluated_token_count": token_count,
                    "rouge_l_f1": float(rouge_l),
                    "final_active_kv_bytes": generation_criterion.final_active_bytes,
                    "rouge_l_f1_human_reference": float(
                        Rouge().get_scores(answer, reference_answer)[0]["rouge-l"]["f"]
                    ),
                }
            )
            selected_positions[kind][evaluation_id] = [
                list(layer) for layer in generation_criterion.selected_positions
            ]
            gc.collect()
            torch.cuda.empty_cache()
        del evaluation_image, evaluation_input_ids
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
        "paper_reproduction": bool(args.paper_reproduction),
        "official_profile_reproduction": bool(args.official_profile_reproduction),
        "official_profile_shard": bool(args.official_profile_shard),
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
        "calibration_source": (
            {
                "path": str(calibration_source.resolve()),
                "sha256": observed_calibration_revision,
                "dataset_id": args.calibration_dataset_id or DATASET_ID,
                "dataset_revision": args.calibration_dataset_revision
                or observed_calibration_revision,
            }
            if calibration_source is not None
            else {
                "status": "unverifiable_upstream_profile_ids_not_released",
                "source_profile_path": str(official_profile_source),
                "source_profile_sha256": profile_sha256,
            }
        ),
        "calibration_indices": list(calibration_indices),
        "evaluation_indices": list(evaluation_indices),
        "parity_thresholds": {
            "minimum_token_agreement": args.minimum_token_agreement,
            "maximum_relative_ppl_delta": args.maximum_relative_ppl_delta,
            "maximum_rouge_l_delta": args.maximum_rouge_l_delta,
            "maximum_retained_byte_delta": args.maximum_retained_byte_delta,
        },
    }
    # The cache sizes vary with prompt/image length.  This common control is
    # only an identifier for a matched parity set; per-sample physical bytes
    # remain in the observations and must be used for any budget comparison.
    source_length = observations[execution_kinds[0]][0]["per_layer_cache_sizes"]
    target_slots = sum(int(value) for value in source_length) * int(
        model.config.num_attention_heads
    )
    controls = PrefixKVParityControls(
        model_id=MODEL_ID,
        model_revision=MODEL_REPOSITORY_SHA,
        tokenizer_revision=MODEL_REPOSITORY_SHA,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        calibration_sample_set_sha256=_sha256_json(list(calibration_ids)),
        evaluation_sample_set_sha256=_sha256_json(list(evaluation_ids)),
        prompt_payload_sha256=_sha256_json(prompt_payloads),
        media_payload_sha256=_sha256_json(evaluation_media_hashes),
        profile_sha256=profile_sha256,
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
        "profile": profile_payload,
        "protocol": protocol_payload,
    }
    config_sha = _sha256_json(config_payload)
    manifest_path = args.output_directory / "manifest.json"
    manifest = {
        "schema_version": 1,
        "run_id": args.output_directory.name,
        # Final eligibility is resolved only after the parity gate below.
        "paper_eligible": False,
        "paper_ineligibility_reason": "pending_parity_gate",
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
            "file_sha256": observed_dataset_revision,
            "calibration": {
                "id": (
                    args.calibration_dataset_id or DATASET_ID
                    if calibration_source is not None
                    else None
                ),
                "revision": (
                    args.calibration_dataset_revision or observed_calibration_revision
                    if calibration_source is not None
                    else None
                ),
                "file_sha256": observed_calibration_revision,
                "separate_source": (
                    calibration_source.resolve() != args.data_path.resolve()
                    if calibration_source is not None
                    else None
                ),
                "provenance_status": (
                    "recorded_disjoint_source"
                    if calibration_source is not None
                    else "unverifiable_upstream_profile_ids_not_released"
                ),
            },
            "calibration_sample_ids": list(calibration_ids),
            "evaluation_sample_ids": list(evaluation_ids),
            "intersection": [],
        },
        "media": {
            "calibration_sha256": calibration_media_hashes,
            "evaluation_sha256": evaluation_media_hashes,
        },
        "profile": {
            "native_path": str(profile_path) if profile_path is not None else None,
            "native_sha256": profile.profile_sha256 if profile is not None else None,
            "official_raw_path": str(official_profile_path),
            "official_raw_file_sha256": _sha256_file(official_profile_path),
            "source_path": str(official_profile_source) if official_profile_source else None,
            "source_sha256": profile_sha256 if official_profile_source else None,
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
        "measurement_types": (
            ["baseline_official_measured"]
            if args.official_profile_reproduction
            else ["baseline_official_measured", "baseline_reimpl_measured"]
        ),
        "protocol": protocol_payload,
        "reference": {
            "samples": reference_rows,
        },
        "selected_positions": selected_positions,
        "extended_observations": observations,
        "created_unix_seconds": time.time(),
    }

    artifacts: dict[str, PrefixKVParityArtifact] = {}
    artifact_specs = [("official", "official_prefixkv", OFFICIAL_SHA, "baseline_official_measured")]
    if not args.official_profile_reproduction:
        artifact_specs.append(
            ("reimplementation", "prefixkv_reimpl", root_sha, "baseline_reimpl_measured")
        )
    for kind, implementation, executable_sha, measurement_type in artifact_specs:
        samples = []
        for observation in sorted(observations[kind], key=lambda item: str(item["sample_id"])):
            row = dict(observation)
            row.pop("final_active_kv_bytes")
            row.pop("rouge_l_f1_human_reference")
            samples.append(PrefixKVSampleObservation(**row))
        artifact = PrefixKVParityArtifact(
            implementation=implementation,
            official_repository_sha=OFFICIAL_SHA,
            executable_git_sha=executable_sha,
            config_sha256=config_sha,
            manifest_path=str(manifest_path),
            measurement_type=measurement_type,
            controls=controls,
            samples=tuple(samples),
        )
        artifacts[kind] = artifact
        (args.output_directory / f"{kind}.json").write_text(
            json.dumps(artifact.to_json_object(), indent=2, sort_keys=True) + "\n"
        )
    if args.official_profile_reproduction:
        comparison: dict[str, Any] = {
            "status": "not_applicable",
            "reason": (
                "official-profile reproduction executes the pinned official implementation only; "
                "local reimplementation parity requires a separately recorded calibration source"
            ),
        }
        parity_gate: dict[str, Any] = {
            "passed": None,
            "status": "not_applicable",
            "reason": comparison["reason"],
        }
    else:
        comparison = compare_prefixkv_artifacts(
            artifacts["official"], artifacts["reimplementation"]
        )
        parity_gate = _parity_gate(
            cast("dict[str, Any]", comparison),
            minimum_token_agreement=args.minimum_token_agreement,
            maximum_relative_ppl_delta=args.maximum_relative_ppl_delta,
            maximum_rouge_l_delta=args.maximum_rouge_l_delta,
            maximum_retained_byte_delta=args.maximum_retained_byte_delta,
        )
    (args.output_directory / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (args.output_directory / "parity-gate.json").write_text(
        json.dumps(parity_gate, indent=2, sort_keys=True) + "\n"
    )
    ineligibility: list[str] = []
    if dirty:
        ineligibility.append("dirty_worktree")
    if not (args.paper_reproduction or args.official_profile_reproduction):
        ineligibility.append("diagnostic_mode")
    if args.official_profile_reproduction and len(evaluation_items) != 1000:
        ineligibility.append("incomplete_official_profile_shard")
    if not args.official_profile_reproduction and not bool(parity_gate["passed"]):
        ineligibility.append("official_reimplementation_parity_gate_failed")
    manifest["paper_eligible"] = not ineligibility
    manifest["paper_ineligibility_reason"] = ";".join(ineligibility) if ineligibility else None
    manifest["parity_gate"] = parity_gate
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
