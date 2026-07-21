"""Cache-aware eager Hugging Face multimodal model adapters."""

from mosaickv.adapters.huggingface.base import (
    AttentionCapture,
    HuggingFaceMultimodalAdapter,
    QueryCapture,
)
from mosaickv.adapters.huggingface.internvl import InternVL25Adapter, InternVLVideo
from mosaickv.adapters.huggingface.llava import Llava15Adapter
from mosaickv.adapters.huggingface.llava_onevision import LlavaOneVisionAdapter
from mosaickv.adapters.huggingface.parity import (
    compare_cache_reinjection,
    compare_mosaickv_retention_one,
    compare_with_generate,
)
from mosaickv.adapters.huggingface.qwen2_5_vl import Qwen25VLAdapter
from mosaickv.adapters.huggingface.registry import (
    adapter_for_loaded_model,
    audited_model_revision,
    load_hf_adapter,
    runtime_adapter_class,
)
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    AdapterProfilingModules,
    CachedKeyState,
    CacheLayerLayout,
    CacheLayerSnapshot,
    CacheLayout,
    CacheSnapshot,
    DecodeOutput,
    DecodeState,
    GreedyDecodeOutput,
    Modality,
    ModalitySpan,
    ParityReport,
    PrefillOutput,
    PreparedInputs,
    QueryVectors,
    QueryVectorState,
)

__all__ = [
    "AdapterCapabilities",
    "AdapterProfilingModules",
    "AttentionCapture",
    "CacheLayerLayout",
    "CacheLayerSnapshot",
    "CacheLayout",
    "CacheSnapshot",
    "CachedKeyState",
    "DecodeOutput",
    "DecodeState",
    "GreedyDecodeOutput",
    "HuggingFaceMultimodalAdapter",
    "InternVL25Adapter",
    "InternVLVideo",
    "Llava15Adapter",
    "LlavaOneVisionAdapter",
    "Modality",
    "ModalitySpan",
    "ParityReport",
    "PrefillOutput",
    "PreparedInputs",
    "QueryCapture",
    "QueryVectorState",
    "QueryVectors",
    "Qwen25VLAdapter",
    "adapter_for_loaded_model",
    "audited_model_revision",
    "compare_cache_reinjection",
    "compare_mosaickv_retention_one",
    "compare_with_generate",
    "load_hf_adapter",
    "runtime_adapter_class",
]
