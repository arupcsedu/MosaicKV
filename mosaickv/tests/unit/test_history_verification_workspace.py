from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.history_verification_workspace import partition, verify
from mosaickv.evaluation.prompt_graph_policy import PromptGraphPolicy


def test_arena_views_are_contiguous_disjoint_and_include_full_owner():
    buffer = torch.zeros(1, 4, 40, 128, dtype=torch.bfloat16)
    parts = partition(buffer, (1, 4, 8, 128))
    views = [x for p in parts for x in p]
    assert all(x.is_contiguous() and x.shape == (1, 4, 8, 128) for x in views)
    assert [x.storage_offset() for x in views] == [0, 4096, 8192, 12288]
    assert all(x.untyped_storage().data_ptr() == buffer.data_ptr() for x in views)
    assert partition(buffer, (1, 4, 16, 128)) is None


class Pair:
    def __init__(self, values, *, incomplete=False):
        self.values = values
        self.shape = tuple(values[0].shape)
        self.archive = SimpleNamespace(cos=torch.zeros(1))
        self.calls = 0
        self.incomplete = incomplete

    def fill(self, attention, targets, *, direct):
        self.calls += 1
        for a, b in zip(targets, self.values, strict=True):
            if self.incomplete and self.calls == 2:
                a[..., 1:, :].copy_(b[..., 1:, :])
            else:
                a.copy_(b)

    def materialize(self, attention):
        out = tuple(torch.empty_like(x) for x in self.values)
        self.fill(attention, out, direct=True)
        return out

    def private(self):
        return self.values


@pytest.mark.parametrize("borrow", [False, True])
def test_three_independent_fills_and_six_bit_flags_are_preserved(borrow):
    source = tuple(torch.arange(4 * 8 * 128).reshape(1, 4, 8, 128).bfloat16() + i for i in (0, 7))
    candidate = Pair(source)
    old = Pair(tuple(x[..., :4, :] for x in source)), Pair(tuple(x[..., 4:, :] for x in source))
    scratch = (
        partition(torch.empty(1, 4, 40, 128, dtype=torch.bfloat16), candidate.shape)
        if borrow
        else None
    )
    flags = []
    verify(SimpleNamespace(attention=None), candidate, old, candidate.shape, flags, scratch)
    assert candidate.calls == 3 and len(flags) == 6 and all(bool(x) for x in flags)
    assert all(x.calls == 1 for x in old)


def test_reused_output_is_poisoned_each_time_and_cannot_hide_an_incomplete_writer():
    values = tuple(torch.ones(1, 4, 8, 128, dtype=torch.bfloat16) for _ in range(2))
    candidate = Pair(values, incomplete=True)
    scratch = partition(torch.empty(1, 4, 40, 128, dtype=torch.bfloat16), candidate.shape)
    flags = []
    verify(
        SimpleNamespace(attention=None), candidate, (Pair(values),), candidate.shape, flags, scratch
    )
    assert candidate.calls == 3 and len(flags) == 6
    assert [bool(x) for x in flags] == [True, True, False, False, True, True]


def test_persistent_history_may_not_alias_scratch():
    buffer = torch.ones(1, 4, 40, 128, dtype=torch.bfloat16)
    scratch = partition(buffer, (1, 4, 8, 128))
    candidate = Pair(tuple(x.clone() for x in scratch[0]))
    candidate.private = lambda: (scratch[0][0],)
    with pytest.raises(RuntimeError, match="aliases"):
        verify(
            SimpleNamespace(attention=None),
            candidate,
            (Pair(candidate.values),),
            candidate.shape,
            None,
            scratch,
        )


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"history_arena": True},
        {"history_arena": 1, "captured_code_replay": False, "row_major_codes": False},
        {
            "history_arena": False,
            "captured_code_replay": False,
            "row_major_codes": False,
            "extra": True,
        },
    ],
)
def test_graph_policy_rejects_partial_unknown_or_nonboolean_config(value):
    with pytest.raises(ValueError):
        PromptGraphPolicy.parse(value)


def test_graph_policy_default_does_not_enable_any_new_path():
    assert PromptGraphPolicy.parse(None) == PromptGraphPolicy(False, False, False)
