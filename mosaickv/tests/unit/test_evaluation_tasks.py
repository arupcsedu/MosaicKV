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
        "coco_caption",
        "docvqa",
        "infovqa",
        "llava_description",
        "mmstar",
        "milebench",
        "mmvet",
        "mmvet_prefixkv",
        "mathvista",
        "realworldqa",
        "synthetic_ci",
        "textvqa",
        "textvqa_train_calibration",
        "video_mme",
    }
    assert registry.resolve("textvqa").development_lmms_task == "textvqa_val"
    assert registry.resolve("mmstar").route_key == (
        "Lin-Chen/MMStar",
        None,
        "val",
        "mmstar",
    )
    assert registry.resolve("docvqa").development_lmms_task == "docvqa_val"
    assert registry.resolve("chartqa").development_lmms_task == "chartqa"
    coco = registry.resolve("coco_caption")
    assert coco.sample_metric == "coco_CIDEr"
    assert coco.paper_metrics == ("cider",)
    assert coco.lmms_task_directory == "coco_cap"
    mathvista = registry.resolve("mathvista")
    assert mathvista.sample_metric == "llm_as_judge_eval"
    assert mathvista.paper_metrics == ("mathvista_accuracy",)
    assert registry.resolve("llava_description").scoring_owner == "paper_native_prefixkv"
    assert registry.resolve("mmvet_prefixkv").paper_metrics == (
        "teacher_forced_ppl",
        "rouge_l_f1",
    )
    assert registry.resolve("milebench").scoring_owner == "paper_native_milebench"
    realworldqa = registry.resolve("realworldqa")
    assert realworldqa.route_key == (
        "lmms-lab/RealWorldQA",
        "default",
        "test",
        "realworldqa",
    )
    assert realworldqa.sample_metric == "exact_match"
    assert realworldqa.scoring_deterministic is True
    assert realworldqa.reference_field == "answer"
    text_train = registry.resolve("textvqa_train_calibration")
    assert text_train.route_key == (
        "lmms-lab/textvqa",
        "default",
        "train",
        "textvqa_val",
    )
    assert text_train.lmms_split_override == "train"
    assert text_train.reference_field == "answers"
    assert registry.resolve("textvqa").reference_field == "answers"
    assert registry.resolve("infovqa").route_key == (
        "lmms-lab-encoder/DocVQA",
        "InfographicVQA",
        "validation",
        "infovqa_val",
    )
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


def test_deterministic_selection_offsets_form_disjoint_shards() -> None:
    ids = tuple(f"sample-{index}" for index in range(11))
    full = deterministic_sample_ids(ids, seed=17, subset_size=None)
    shards = tuple(
        deterministic_sample_ids(ids, seed=17, subset_size=4, offset=offset) for offset in (0, 4, 8)
    )
    assert tuple(sample_id for shard in shards for sample_id in shard) == full
    assert [len(shard) for shard in shards] == [4, 4, 3]
    with pytest.raises(ValueError, match="offset"):
        deterministic_sample_ids(ids, seed=17, subset_size=4, offset=-1)


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
