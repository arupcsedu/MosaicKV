"""Deterministic local model used only to validate the evaluation harness."""

from __future__ import annotations

from typing import ClassVar

from mosaickv.evaluation.messages import MediaKind
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration


class SyntheticColorModel:
    """Answer the packaged RGB fixture without model weights."""

    model_id = "mosaickv/synthetic-color-model"
    backend = "synthetic"
    method = "ci_fixture"
    retention_ratio = 1.0
    supports_video = False

    _COLORS: ClassVar[dict[tuple[int, int, int], str]] = {
        (0, 0, 255): "blue",
        (0, 255, 0): "green",
        (255, 0, 0): "red",
        (255, 255, 255): "white",
    }

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Return the color encoded by the single synthetic image part."""

        media = [
            part.value
            for message in request.messages
            for part in message.content
            if part.type == MediaKind.IMAGE.value
        ]
        if len(media) != 1 or not isinstance(media[0], tuple):
            raise ValueError("synthetic color model requires exactly one RGB tuple")
        try:
            answer = self._COLORS[media[0]]
        except KeyError as error:
            raise ValueError(f"unknown synthetic RGB value: {media[0]!r}") from error
        return ModelGeneration(answer=answer, metrics=GenerationMetrics(generated_tokens=1))


__all__ = ["SyntheticColorModel"]
