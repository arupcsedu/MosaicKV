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

    BLOCKS = "blocks"
    RETAINED_SLOTS = "retained_slots"
    BYTES = "bytes"


class MeasurementType(StrEnum):
    """Controlled measurement vocabulary from SCIENTIFIC_INTEGRITY.md."""

    REFERENCE = "reference_measured"
    METHOD = "method_measured"
    BASELINE_OFFICIAL = "baseline_official_measured"
    BASELINE_REIMPL = "baseline_reimpl_measured"
    BASELINE_SIMPLE = "baseline_simple_measured"
    ABLATION = "ablation_measured"
    DERIVED = "derived"
    VALIDATION_SMOKE = "validation_smoke"


class OutputLengthPolicy(StrEnum):
    """Generation output-length policies."""

    FIXED_MAX_NEW_TOKENS = "fixed_max_new_tokens"
    STOPPING_CRITERIA = "stopping_criteria"


class ForecastMode(StrEnum):
    """Online sources used to forecast future decoder queries."""

    PROMPT_WINDOW = "prompt_window"
    DRAFT_ROLLOUT = "draft_rollout"
    HYBRID = "hybrid"


class ForecastCovariance(StrEnum):
    """Prompt-query uncertainty representation."""

    DIAGONAL = "diagonal"
    FULL = "full"


class ResidualStorageDType(StrEnum):
    """Supported CPU residual payload encodings."""

    LOSSLESS = "lossless"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


class RepairPolicy(StrEnum):
    """Online and evaluation-only residual-repair trigger policies."""

    NONE = "none"
    ENTROPY = "entropy"
    PROTOTYPE_RISK = "prototype_risk"
    ENTROPY_OR_PROTOTYPE_RISK = "entropy_or_prototype_risk"
    ORACLE = "oracle"


class LookMMergeStrategy(StrEnum):
    """Paper-defined LOOK-M KV-pair merge variants."""

    AVERAGED = "averaged"
    PIVOTAL = "pivotal"
    WEIGHTED = "weighted"


class PrefixKVProfileMode(StrEnum):
    """Source of the layer-wise PrefixKV cache-size recipe."""

    OFFLINE_PROFILE = "offline_profile"
    FIXED_GLOBAL = "fixed_global"


class MosaicKVMethod(StrEnum):
    """Full-cache reference, simple baselines, and MosaicKV variants."""

    # ``fullkv`` is retained for backwards-compatible configs.  New baseline
    # sweeps should use the paper-facing ``full_kv`` spelling.
    FULLKV = "fullkv"
    FULL_KV = "full_kv"
    RANDOM_KV = "random_kv"
    UNIFORM_KV = "uniform_kv"
    PROMPT_ATTENTION_TOPK = "prompt_attention_topk"
    VALUE_TOPK = "value_topk"
    LOOKM_REIMPL = "lookm_reimpl"
    PREFIXKV_REIMPL = "prefixkv_reimpl"
    VL_CACHE_REIMPL = "vl_cache_reimpl"
    MOSAICKV_EXACT = "mosaickv_exact"
    MOSAICKV_PROTO = "mosaickv_proto"
    MOSAICKV_FULL = "mosaickv_full"

    @property
    def is_full_cache(self) -> bool:
        """Whether this method is the transformation-free reference path."""

        return self in {self.FULLKV, self.FULL_KV}

    @property
    def is_simple_baseline(self) -> bool:
        """Whether this is one of the locally defined transparent baselines."""

        return self in {
            self.FULL_KV,
            self.RANDOM_KV,
            self.UNIFORM_KV,
            self.PROMPT_ATTENTION_TOPK,
            self.VALUE_TOPK,
        }

    @property
    def is_compressed_baseline(self) -> bool:
        """Whether the policy selects an exact-only cache subset."""

        return self in {
            self.RANDOM_KV,
            self.UNIFORM_KV,
            self.PROMPT_ATTENTION_TOPK,
            self.VALUE_TOPK,
        }

    @property
    def is_mosaickv(self) -> bool:
        """Whether all configured MosaicKV planning stages are applicable."""

        return self in {
            self.MOSAICKV_EXACT,
            self.MOSAICKV_PROTO,
            self.MOSAICKV_FULL,
        }

    @property
    def is_published_reimplementation(self) -> bool:
        """Whether the method is local paper-faithful, never official code."""

        return self in {
            self.LOOKM_REIMPL,
            self.PREFIXKV_REIMPL,
            self.VL_CACHE_REIMPL,
        }

    @property
    def is_lookm_reimplementation(self) -> bool:
        """Whether this is the local LOOK-M implementation."""

        return self is self.LOOKM_REIMPL

    @property
    def is_prefixkv_reimplementation(self) -> bool:
        """Whether this is the local PrefixKV implementation."""

        return self is self.PREFIXKV_REIMPL

    @property
    def is_vl_cache_reimplementation(self) -> bool:
        """Whether this is the local ICLR VL-Cache implementation."""

        return self is self.VL_CACHE_REIMPL
