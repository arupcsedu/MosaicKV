from __future__ import annotations

import pytest

from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.tasks import (
    TaskSample,
    default_task_registry,
    deterministic_sample_ids,
    load_synthetic_samples,
    select_samples,
)


def test_registry_routes_public_scoring_to_lmms_eval() -> None:
    registry = default_task_registry()
    assert set(registry.names()) == {
        "chartqa",
        "docvqa",
        "mmstar",
        "mmvet",
        "synthetic_ci",
        "textvqa",
        "video_mme",
    }
    assert registry.resolve("textvqa").development_lmms_task == "textvqa_val_lite"
    assert registry.resolve("video_mme").requires_video is True
    with pytest.raises(RuntimeError, match="lmms_eval"):
        registry.resolve("chartqa").score_local("1", ("1",))


def test_deterministic_selection_is_order_independent() -> None:
    ids = ("a", "b", "c", "d", "e")
    first = deterministic_sample_ids(ids, seed=91, subset_size=3)
    second = deterministic_sample_ids(tuple(reversed(ids)), seed=91, subset_size=3)
    assert first == second
    assert len(first) == len(set(first)) == 3
    assert first != deterministic_sample_ids(ids, seed=92, subset_size=3)


def test_sample_selection_rejects_duplicate_ids() -> None:
    sample = TaskSample("same", "question", ("answer",), ())
    with pytest.raises(ValueError, match="duplicate"):
        select_samples((sample, sample), seed=0, subset_size=1)


def test_standard_messages_preserve_media_order() -> None:
    image = MediaItem(MediaKind.IMAGE, object())
    video = MediaItem(MediaKind.VIDEO, "/cache/example.mp4")
    messages = build_multimodal_messages("Question?", (image, video), system_prompt="System")
    assert [message.role for message in messages] == ["system", "user"]
    assert [part.type for part in messages[1].content] == ["image", "video", "text"]
    assert messages[1].content[0].value is image.payload


def test_synthetic_fixture_has_unique_local_multimodal_samples() -> None:
    samples = load_synthetic_samples()
    assert len(samples) == 4
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert all(sample.media[0].kind == MediaKind.IMAGE for sample in samples)
