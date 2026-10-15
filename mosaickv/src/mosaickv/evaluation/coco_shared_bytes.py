"""COCO shared-byte validation and efficient, *corpus-recomputed* CIDEr.

Only PTB tokenization is cached. CIDEr's document frequencies and TF-IDF scores
are rebuilt for every bootstrap draw, including repeated source identities.
No shard scores or cached per-image CIDEr contributions are averaged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from mosaickv.evaluation.metrics.types import MetricSample

COCO_REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
COCO_PROMPT = "Provide a one-sentence caption for the provided image."
METHODS = ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
STORAGE_FIELDS = tuple(f"{phase}_active_kv_storage_bytes" for phase in ("initial", "final", "peak"))


def processor_signature(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    return dict(
        processor_class=type(processor).__qualname__,
        image_preprocessor=processor.image_processor.to_dict(),
        chat_template=getattr(processor, "chat_template", None),
        tokenizer_chat_template=getattr(tokenizer, "chat_template", None),
        special_tokens=tokenizer.special_tokens_map,
        vocabulary=tokenizer.get_vocab(),
    )


def control_hashes(
    config: Any, sample: dict[str, Any], processor: dict[str, Any], gpu: dict[str, Any]
) -> dict[str, str]:
    from mosaickv.evaluation.shared_byte_contract import CONTROL_FIELDS, digest

    values = dict(
        model=asdict(config.model),
        processor=processor,
        chat_template=[processor["chat_template"], processor["tokenizer_chat_template"]],
        prompts=sample["prompt"],
        media=[m["sha256"] for m in sample["media"]],
        preprocessing=processor["image_preprocessor"],
        tokenization=[processor["vocabulary"], processor["special_tokens"]],
        generation=asdict(config.generation),
        precision=config.model.precision.value,
        backend=config.execution.backend.value,
        attention=config.execution.attention_implementation,
        block_policy=[
            config.cache.block_size,
            config.cache.physical_budget_block_size,
            config.cache.packing_mode,
        ],
        mandatory_policy=config.cache.physical_budget_mandatory_policy,
        decode_growth_policy="append_only_no_eviction",
        measurement=dict(warmups=5, quality_trials=1, output_tokens=64, cpu_threads=12),
        gpu_controls=gpu,
        dataset=asdict(config.dataset),
        references=sample["references"],
        evaluator="lmms_eval.coco_cap.PTBTokenizer+Cider_x100@0.7.2; pycocoevalcap==1.2",
    )
    if set(values) != set(CONTROL_FIELDS):
        raise ValueError("missing comparison controls")
    return {k: digest(v) for k, v in values.items()}


class TokenizedCider:
    """Freeze tokens, not scores, for a known union of paired predictions."""

    def __init__(self, samples: Sequence[MetricSample]) -> None:
        from lmms_eval.tasks.coco_cap import utils

        if not samples:
            raise ValueError("empty CIDEr cohort")
        self._scorer = utils.Cider
        keys = list(dict.fromkeys((s.prediction, s.references) for s in samples))
        tokenizer = utils.PTBTokenizer()
        refs = tokenizer.tokenize(
            {str(i): [{"caption": r} for r in k[1]] for i, k in enumerate(keys)}
        )
        preds = tokenizer.tokenize({str(i): [{"caption": k[0]}] for i, k in enumerate(keys)})
        self._tokens = {key: (refs[str(i)], preds[str(i)]) for i, key in enumerate(keys)}

    def __call__(self, samples: Sequence[MetricSample]) -> float:
        if not samples:
            raise ValueError("empty CIDEr draw")
        try:
            pairs = [self._tokens[(s.prediction, s.references)] for s in samples]
        except KeyError as error:
            raise ValueError("caption/reference changed after tokenization freeze") from error
        # Evaluator-local IDs deliberately count repeated bootstrap occurrences.
        score, _ = self._scorer().compute_score(
            {str(i): pair[0] for i, pair in enumerate(pairs)},
            {str(i): pair[1] for i, pair in enumerate(pairs)},
        )
        value = float(score) * 100.0
        if not math.isfinite(value):
            raise ValueError("non-finite CIDEr")
        return value


def dense_capacity_certificate(
    layout: Sequence[dict[str, Any]], cap_bytes: int, *, mandatory_positions: int = 1
) -> dict[str, Any]:
    """Independent block-size-one feasibility from original *FullKV* shapes.

    Called before any compressed selection; never derive capacity from selected
    bytes. This route uses ordinary batch-one B,H,S,D HF storage, no packing,
    aliasing, quantization or unequal per-head physical dimensions.
    """
    if not layout or type(cap_bytes) is not int or cap_bytes <= 0:
        raise ValueError("positive byte cap and source layout required")
    lengths, per_position = set(), 0
    for layer in layout:
        k, v = layer["key_shape"], layer["value_shape"]
        if len(k) != 4 or len(v) != 4 or k != v or k[0] != 1 or any(x <= 0 for x in k):
            raise ValueError("require batch-one dense B,H,S,D K/V layout")
        if layer["key_element_size"] != 2 or layer["value_element_size"] != 2:
            raise ValueError("this lane requires FP16 K and V")
        lengths.add(k[2])
        per_position += k[1] * k[3] * 4
    if len(lengths) != 1:
        raise ValueError("source layers must share sequence length")
    source_positions = next(iter(lengths))
    positions = min(source_positions, cap_bytes // per_position)
    if positions < mandatory_positions:
        raise ValueError("cap cannot hold mandatory positions")
    return dict(
        status="passed",
        boundary="unique_live_backing_storage",
        physical_byte_cap=cap_bytes,
        largest_feasible_initial_bytes=positions * per_position,
        source_positions=source_positions,
        bytes_per_position=per_position,
        retained_positions=positions,
        target_shortfall_bytes=cap_bytes - positions * per_position,
        proof="uniform dense block_size=1; floor(cap/bytes_per_position), bounded by source",
        source_layout=list(layout),
    )


def validate_caption_rows(
    rows: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]], tokens: int = 64
) -> None:
    import json

    expected = {s["sample_id"]: s for s in samples}
    if (
        len(expected) != len(samples)
        or len(rows) != len(expected)
        or {r["sample_id"] for r in rows} != set(expected)
    ):
        raise ValueError("missing/duplicate/unexpected samples; preserve failed denominator")
    for row in rows:
        if row["status"] != "completed" or row.get("error"):
            raise ValueError("failed sample blocks complete-cohort CIDEr")
        # An observed empty answer is a genuine zero-quality outcome, not a
        # missing result. Preserve it in CIDEr and the paired denominator.
        if not isinstance(row["answer"], str):
            raise ValueError("missing caption")
        if json.loads(row["reference"]) != expected[row["sample_id"]]["references"]:
            raise ValueError("reference annotation mismatch")
        if row["generated_tokens"] != tokens:
            raise ValueError("unexpected output length")
        for field in ("compression_time", "decode_time", "end_to_end_time", "prefill_time", "ttft"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"missing/non-finite timing: {field}")
        if row["repair_count"] != 0 or row["residual_kv_bytes"] != 0:
            raise ValueError("exact-only comparison cannot include repair/residuals")
        if any(type(row.get(field)) is not int or row[field] <= 0 for field in STORAGE_FIELDS):
            raise ValueError("missing physical initial/final/peak storage")


def sample_size_recommendation(
    *,
    pilot_n: int,
    ci_low: float,
    ci_high: float,
    target_half_width: float,
    seconds_per_example: float,
    available: int = 4096,
) -> dict[str, Any]:
    """Explicit pilot precision extrapolation, NOT a guaranteed power claim."""
    if (
        pilot_n < 2
        or ci_high < ci_low
        or target_half_width <= 0
        or seconds_per_example <= 0
        or available < pilot_n
    ):
        raise ValueError("invalid preregistration inputs")
    required = max(
        pilot_n, math.ceil(pilot_n * ((ci_high - ci_low) / (2 * target_half_width)) ** 2)
    )
    n = min(available, required)
    return dict(
        recommended_samples=n,
        estimated_required_samples=required,
        target_precision_feasible_estimate=required <= available,
        estimated_gpu_hours=n * seconds_per_example / 3600,
        estimated_four_gpu_hours=n * seconds_per_example / (4 * 3600),
        queue_time_included=False,
        precision_method="pilot CI width times sqrt(n/N); heuristic, not guaranteed power",
    )
