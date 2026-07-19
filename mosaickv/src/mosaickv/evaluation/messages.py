"""Backend-neutral multimodal chat message construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MediaKind(StrEnum):
    """Media types accepted by the evaluation boundary."""

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class MediaItem:
    """One image or video payload passed through without preprocessing."""

    kind: MediaKind
    payload: Any


@dataclass(frozen=True, slots=True)
class MessagePart:
    """One content part in the standardized chat representation."""

    type: str
    value: Any


@dataclass(frozen=True, slots=True)
class MultimodalMessage:
    """A role-tagged message with ordered media followed by prompt text."""

    role: str
    content: tuple[MessagePart, ...]


def build_multimodal_messages(
    prompt: str,
    media: tuple[MediaItem, ...] = (),
    *,
    system_prompt: str | None = None,
) -> tuple[MultimodalMessage, ...]:
    """Build one canonical message sequence without changing media payloads."""

    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    messages: list[MultimodalMessage] = []
    if system_prompt is not None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty when provided")
        messages.append(
            MultimodalMessage(
                role="system",
                content=(MessagePart(type="text", value=system_prompt),),
            )
        )
    parts = [MessagePart(type=item.kind.value, value=item.payload) for item in media]
    parts.append(MessagePart(type="text", value=prompt))
    messages.append(MultimodalMessage(role="user", content=tuple(parts)))
    return tuple(messages)


__all__ = [
    "MediaItem",
    "MediaKind",
    "MessagePart",
    "MultimodalMessage",
    "build_multimodal_messages",
]
