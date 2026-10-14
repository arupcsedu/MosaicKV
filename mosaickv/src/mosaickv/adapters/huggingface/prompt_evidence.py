"""Fail-closed user-instruction mapping for the audited LLaVA-1.5 processor.

Uses tokenizer offsets and verifies every ID against the actual processor output.
Image expansion is aligned explicitly; no decoded-string search or suffix guess.
Called only by opt-in evidence ablations, never by historical configurations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from mosaickv.cache_state import Modality


@dataclass(frozen=True, slots=True)
class PromptEvidenceMap:
    positions: tuple[int, ...]
    character_spans: tuple[tuple[int, int], ...]
    logical_length: int
    input_ids_sha256: str
    prompt_sha256: str
    provenance: str = "llava15_user_text_offset_and_expanded_id_alignment_v1"

    def trace(self) -> dict[str, Any]:
        result = asdict(self)
        result["mapping_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return result


def align_text_positions(
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    expanded_ids: list[int],
    image_token_id: int,
    spans: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Map overlapping text tokens while admitting only image-ID expansion."""
    if len(offsets) != len(token_ids) or not spans:
        raise ValueError("missing tokenizer offsets or user-text spans")
    positions: list[int] = []
    source = target = 0
    covered = [False] * len(spans)
    while source < len(token_ids):
        if target >= len(expanded_ids) or token_ids[source] != expanded_ids[target]:
            raise ValueError("tokenizer/processor IDs differ outside image expansion")
        if token_ids[source] == image_token_id:
            raw_start, expanded_start = source, target
            while source < len(token_ids) and token_ids[source] == image_token_id:
                source += 1
            while target < len(expanded_ids) and expanded_ids[target] == image_token_id:
                target += 1
            if target - expanded_start < source - raw_start:
                raise ValueError("processor contracted image-token placeholders")
            continue
        start, end = offsets[source]
        if start < 0 or end < start:
            raise ValueError("invalid tokenizer offset")
        for i, (left, right) in enumerate(spans):
            if end > start and start < right and end > left:
                covered[i] = True
                positions.append(target)
        source += 1
        target += 1
    if target != len(expanded_ids) or not all(covered):
        raise ValueError("incomplete processor alignment or unmapped user text")
    return tuple(sorted(set(positions)))


def map_prompt_evidence(adapter: Any, prepared: Any, messages: Any) -> PromptEvidenceMap:
    """Map all text in the sole user turn (question plus formatting instructions).

    System text is not treated as a question. Multi-turn prompts and unsupported
    tokenizers fail explicitly. Boundary-overlapping tokens are conservatively
    included. The token strings, media, and model inputs are never changed.
    """
    if adapter.capabilities.model_family != "llava_1_5":
        raise ValueError("prompt-evidence mapping is audited only for LLaVA-1.5")
    chat, _, videos = adapter._normalize_messages(messages, ())
    users = [i for i, message in enumerate(chat) if message["role"] == "user"]
    if videos or len(users) != 1 or any(m["role"] not in {"user", "system"} for m in chat):
        raise ValueError("prompt-evidence mapping requires one user turn and no video")
    tokenizer = getattr(adapter.processor, "tokenizer", None)
    if not callable(tokenizer) or not getattr(tokenizer, "is_fast", False):
        raise ValueError("prompt-evidence mapping requires a fast tokenizer with offsets")
    prompt = adapter._render_prompt(chat)
    spans: list[tuple[int, int]] = []
    user = users[0]
    for index, part in enumerate(chat[user]["content"]):
        if part["type"] != "text":
            continue
        marker = f"MOSAICKV_USER_OFFSET_MARKER_{index}_7F937B"
        if marker in prompt:
            raise ValueError("prompt evidence marker collision")
        marked = copy.deepcopy(chat)
        marked[user]["content"][index]["text"] = marker
        rendered = adapter._render_prompt(marked)
        if rendered.count(marker) != 1:
            raise ValueError("chat template does not preserve user text verbatim")
        before, after = rendered.split(marker)
        if before + part["text"] + after != prompt:
            raise ValueError("chat template transforms user text; mapping unsupported")
        spans.append((len(before), len(before) + len(part["text"])))
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    expanded = [int(v) for v in prepared.model_inputs["input_ids"][0].tolist()]
    image_id = adapter._image_token_id()
    if image_id is None or len(expanded) != prepared.logical_sequence_length:
        raise ValueError("missing image-token identity or inconsistent processor length")
    positions = align_text_positions(
        list(encoded["input_ids"]),
        [tuple(offset) for offset in encoded["offset_mapping"]],
        expanded,
        image_id,
        tuple(spans),
    )
    text = {
        p
        for span in prepared.modality_map
        if span.modality is Modality.TEXT
        for p in range(span.start, span.end)
    }
    if not positions or not set(positions) <= text:
        raise ValueError("mapped user evidence is empty or overlaps non-text cache positions")
    return PromptEvidenceMap(
        positions,
        tuple(spans),
        len(expanded),
        hashlib.sha256(json.dumps(expanded).encode()).hexdigest(),
        hashlib.sha256(prompt.encode()).hexdigest(),
    )
