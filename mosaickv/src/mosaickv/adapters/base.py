"""Backend-independent model adapter contracts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from re import Pattern
from typing import cast

from mosaickv.types import JsonObject


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Audited static capability metadata; not runtime load evidence."""

    image: bool
    multi_image: bool
    video: bool
    hf_source: bool
    vllm_source: bool
    sglang_source: bool
    past_key_values_in_hf: bool
    query_projection_in_hf: bool


class ModelAdapter(ABC):
    """Minimal model metadata adapter interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable adapter name."""

    @abstractmethod
    def matches(self, model_id: str) -> bool:
        """Return whether the adapter handles a model ID."""

    @abstractmethod
    def inspect(self, model_id: str, revision: str | None = None) -> JsonObject:
        """Return metadata without loading model weights."""


class StaticModelAdapter(ModelAdapter):
    """Regex-matched adapter containing audited source metadata only."""

    def __init__(
        self,
        *,
        name: str,
        pattern: str,
        audited_revision: str,
        architecture: str,
        cache_layout: str,
        capabilities: ModelCapabilities,
    ) -> None:
        self._name = name
        self._pattern: Pattern[str] = re.compile(pattern)
        self._audited_revision = audited_revision
        self._architecture = architecture
        self._cache_layout = cache_layout
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return self._name

    def matches(self, model_id: str) -> bool:
        return self._pattern.fullmatch(model_id) is not None

    def inspect(self, model_id: str, revision: str | None = None) -> JsonObject:
        if not self.matches(model_id):
            raise ValueError(f"adapter {self.name!r} does not match model {model_id!r}")
        selected_revision = revision or self._audited_revision
        return {
            "adapter": self.name,
            "model_id": model_id,
            "revision": selected_revision,
            "audited_revision": self._audited_revision,
            "revision_matches_audit": selected_revision == self._audited_revision,
            "architecture": self._architecture,
            "cache_layout": self._cache_layout,
            "capabilities": cast("JsonObject", asdict(self._capabilities)),
            "runtime_load_verified": False,
            "loads_model_weights": False,
        }


__all__ = ["ModelAdapter", "ModelCapabilities", "StaticModelAdapter"]
