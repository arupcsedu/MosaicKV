"""Shared controlled vocabularies and JSON types."""

from enum import StrEnum
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class Backend(StrEnum):
    """Execution backends accepted by the configuration schema."""

    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"
    SYNTHETIC = "synthetic"


class Precision(StrEnum):
    """Model/cache precision labels."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    OTHER = "other"


class BudgetUnit(StrEnum):
    """Cache budget accounting units."""

    RETAINED_SLOTS = "retained_slots"
    BYTES = "bytes"


class MeasurementType(StrEnum):
    """Controlled measurement vocabulary from SCIENTIFIC_INTEGRITY.md."""

    REFERENCE = "reference_measured"
    METHOD = "method_measured"
    BASELINE_OFFICIAL = "baseline_official_measured"
    BASELINE_REIMPL = "baseline_reimpl_measured"
    ABLATION = "ablation_measured"
    DERIVED = "derived"
    VALIDATION_SMOKE = "validation_smoke"


class OutputLengthPolicy(StrEnum):
    """Generation output-length policies."""

    FIXED_MAX_NEW_TOKENS = "fixed_max_new_tokens"
    STOPPING_CRITERIA = "stopping_criteria"
