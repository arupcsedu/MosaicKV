"""Generation helpers shared by quality and fixed-length systems measurements."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from mosaickv.types import JsonObject


@dataclass(frozen=True, slots=True)
class QualityDecode:
    """EOS-aware answer text derived from a fixed-length timed trajectory."""

    text: str
    timed_token_count: int
    scored_token_count: int
    quality_completion_tokens: int
    terminal_token_id: int | None
    terminal_token_index: int | None

    def to_record(self) -> JsonObject:
        """Return JSON-safe provenance for the quality/timing boundary."""

        return cast(
            "JsonObject",
            {
                "policy": "decode_before_first_terminal_token",
                "timed_token_count": self.timed_token_count,
                "scored_token_count": self.scored_token_count,
                "quality_completion_tokens": self.quality_completion_tokens,
                "terminal_token_id": self.terminal_token_id,
                "terminal_token_index": self.terminal_token_index,
            },
        )


def _token_id_values(value: object) -> Iterable[int]:
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, int):
        return (value,) if value >= 0 else ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(token for item in value for token in _token_id_values(item))
    return ()


def terminal_token_ids(*owners: object) -> tuple[int, ...]:
    """Collect immutable EOS IDs from model, config, processor, and tokenizer objects."""

    queue = [owner for owner in owners if owner is not None]
    visited: set[int] = set()
    values: set[int] = set()
    while queue:
        owner = queue.pop(0)
        identity = id(owner)
        if identity in visited:
            continue
        visited.add(identity)
        values.update(_token_id_values(getattr(owner, "eos_token_id", None)))
        for name in ("generation_config", "config", "tokenizer"):
            nested = getattr(owner, name, None)
            if nested is not None:
                queue.append(nested)
    return tuple(sorted(values))


def _decoder(processor: Any) -> Any:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        tokenizer = getattr(processor, "tokenizer", processor)
        decoder = getattr(tokenizer, "batch_decode", None)
    if decoder is None:
        raise RuntimeError("processor/tokenizer does not provide batch_decode")
    return decoder


def _decoder_input(token_ids: Sequence[int]) -> Any:
    """Prefer the tensor form accepted by both HF and strict test processors."""

    try:
        import torch
    except ImportError:  # pragma: no cover - HF/server environments install torch
        return [list(token_ids)]
    return torch.tensor([list(token_ids)], dtype=torch.long)


def decode_quality_output(
    processor: Any,
    token_ids: Sequence[int],
    terminal_ids: Sequence[int],
) -> QualityDecode:
    """Decode only the answer prefix while preserving the full timed token count.

    Systems measurements intentionally execute a fixed number of greedy steps.
    Benchmark quality, however, must match ordinary generation semantics and must
    never score tokens produced after the model's first EOS/end-of-turn token.
    """

    timed = tuple(int(value) for value in token_ids)
    if not timed:
        raise ValueError("quality decoding requires at least one generated token")
    terminals = frozenset(int(value) for value in terminal_ids)
    terminal_index = next(
        (index for index, value in enumerate(timed) if value in terminals),
        None,
    )
    scored = timed if terminal_index is None else timed[:terminal_index]
    decoder = _decoder(processor)
    answers = decoder(
        _decoder_input(scored),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = str(answers[0]) if answers else ""
    if not text.strip() and scored and terminal_index is None:
        fallback = decoder(
            _decoder_input(scored),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text = str(fallback[0]) if fallback else ""
    terminal_id = None if terminal_index is None else timed[terminal_index]
    quality_completion_tokens = len(timed) if terminal_index is None else terminal_index + 1
    return QualityDecode(
        text=text,
        timed_token_count=len(timed),
        scored_token_count=len(scored),
        quality_completion_tokens=quality_completion_tokens,
        terminal_token_id=terminal_id,
        terminal_token_index=terminal_index,
    )


__all__ = ["QualityDecode", "decode_quality_output", "terminal_token_ids"]
