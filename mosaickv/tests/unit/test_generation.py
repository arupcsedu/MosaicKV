from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from mosaickv.generation import decode_quality_output, terminal_token_ids


class FakeTokenizer:
    eos_token_id: ClassVar[list[int]] = [99, 100]

    @staticmethod
    def batch_decode(
        batches: Any,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        assert clean_up_tokenization_spaces is False
        special = {99, 100}
        values = batches.detach().cpu().tolist()
        return [
            " ".join(
                str(token) for token in batch if not skip_special_tokens or token not in special
            )
            for batch in values
        ]


@dataclass
class FakeProcessor:
    tokenizer: FakeTokenizer


class SpacedTokenizer(FakeTokenizer):
    @staticmethod
    def batch_decode(
        batches: Any,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        del batches, skip_special_tokens
        assert clean_up_tokenization_spaces is False
        return ["  answer text \n"]


def test_terminal_token_ids_are_collected_from_nested_model_objects() -> None:
    model = type(
        "Model",
        (),
        {
            "generation_config": type("Generation", (), {"eos_token_id": 99})(),
            "config": type("Config", (), {"eos_token_id": (100, 101)})(),
        },
    )()
    processor = FakeProcessor(FakeTokenizer())

    assert terminal_token_ids(model, processor) == (99, 100, 101)


def test_quality_decode_stops_at_eos_but_preserves_fixed_timed_length() -> None:
    result = decode_quality_output(
        FakeProcessor(FakeTokenizer()),
        (12, 13, 99, 77, 78),
        (99,),
    )

    assert result.text == "12 13"
    assert result.timed_token_count == 5
    assert result.scored_token_count == 2
    assert result.quality_completion_tokens == 3
    assert result.terminal_token_id == 99
    assert result.terminal_token_index == 2


def test_quality_decode_uses_all_tokens_when_no_terminal_is_generated() -> None:
    result = decode_quality_output(
        FakeProcessor(FakeTokenizer()),
        (12, 13),
        (99,),
    )

    assert result.text == "12 13"
    assert result.timed_token_count == 2
    assert result.scored_token_count == 2
    assert result.quality_completion_tokens == 2
    assert result.terminal_token_id is None
    assert result.terminal_token_index is None


def test_quality_decode_strips_only_outer_decoded_whitespace() -> None:
    result = decode_quality_output(
        FakeProcessor(SpacedTokenizer()),
        (12, 13),
        (99,),
    )

    assert result.text == "answer text"
    assert result.to_record()["outer_whitespace"] == "stripped"
