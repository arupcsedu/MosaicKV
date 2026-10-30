import random

import pytest

from mosaickv.evaluation.encoded_projection import (
    checkpoint_offsets_reference,
    checkpoint_words_reference,
)


@pytest.mark.parametrize("count", [0, 1, 31, 32, 63, 64])
def test_uniform_half_counts_have_exact_offsets_and_unchanged_bytes(count):
    halves = (count,) * 16
    words = checkpoint_words_reference(halves)
    assert len(words) * 2 == 16
    assert checkpoint_offsets_reference(words) == tuple(count * i for i in range(16))


def test_all_overflow_masks_and_random_mixed_counts_roundtrip():
    rng = random.Random(20261030)
    cases = [tuple(64 if (mask >> (i // 2)) & 1 else 0 for i in range(16)) for mask in range(256)]
    cases += [tuple(rng.randrange(65) for _ in range(16)) for _ in range(10000)]
    for halves in cases:
        words = checkpoint_words_reference(halves)
        assert all(0 <= word <= 65535 for word in words)
        assert checkpoint_offsets_reference(words) == tuple(sum(halves[:i]) for i in range(16))


@pytest.mark.parametrize("counts", [(), (0,) * 15, (True,) * 16, (-1,) * 16, (65,) * 16])
def test_invalid_counts_fail_closed(counts):
    with pytest.raises(ValueError):
        checkpoint_words_reference(counts)


@pytest.mark.parametrize("words", [(), (0,) * 7, (True,) * 8, (-1,) * 8, (65536,) * 8])
def test_invalid_word_shape_and_storage_fail_closed(words):
    with pytest.raises(ValueError):
        checkpoint_offsets_reference(words)


def test_legacy_encoding_is_default_and_packed_counts_are_explicit():
    from inspect import signature

    from mosaickv.evaluation.encoded_projection import EncodedFormat, build
    from mosaickv.evaluation.encoded_projection_kernel import _project, checkpoints, project

    assert EncodedFormat.__dataclass_fields__["packed_literal_counts"].default is False
    assert signature(build).parameters["packed_literal_counts"].default is False
    assert signature(checkpoints).parameters["packed_counts"].default is False
    assert signature(project).parameters["packed_counts"].default is False
    assert next(p for p in _project.params if p.name == "PACK_COUNTS").is_constexpr


@pytest.mark.parametrize("raises", [False, True])
def test_common_baseline_constructor_option_is_scoped_and_restored(monkeypatch, raises):
    import evicastkv_packed_checkpoint_baselines as driver

    import mosaickv.evaluation.encoded_projection as implementation

    calls = []

    def original(base_format, attention, *, packed_literal_counts=False):
        calls.append((base_format, attention, packed_literal_counts))
        if raises:
            raise ValueError("synthetic constructor failure")
        return "built"

    def fused_choices(*args, **kwargs):
        return implementation.build("source", "attention")

    monkeypatch.setattr(implementation, "build", original)
    monkeypatch.setattr(driver.encoded, "fused_choices", fused_choices)
    if raises:
        with pytest.raises(ValueError, match="synthetic constructor failure"):
            driver.choices()
    else:
        assert driver.choices() == "built"
    assert calls == [("source", "attention", True)]
    assert implementation.build is original
