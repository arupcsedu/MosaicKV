"""Strict optional execution policy; original reader defaults stay unchanged."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptGraphPolicy:
    history_arena: bool = False
    captured_code_replay: bool = False
    row_major_codes: bool = False

    def __post_init__(self):
        if any(
            type(x) is not bool
            for x in (self.history_arena, self.captured_code_replay, self.row_major_codes)
        ):
            raise ValueError("explicit boolean graph execution policies required")

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) != {
            "history_arena",
            "captured_code_replay",
            "row_major_codes",
        }:
            raise ValueError("complete strict prompt-graph policy mapping required")
        return cls(**value)
