import pytest

from mosaickv.evaluation.coco_all_methods import caption_text
from mosaickv.evaluation.metrics.types import MetricSample


def test_eos_metadata_is_not_passed_as_the_caption_to_cider():
    detail = {"answer": {"answer": "A skier on snow.", "token_ids": [1, 2], "stopped_by_eos": True}}
    sample = MetricSample("image-id", caption_text(detail), ("A person skiing.",))
    assert sample.prediction == "A skier on snow."
    assert hash((sample.prediction, sample.references))
    assert detail["answer"]["token_ids"] == [1, 2]


def test_invalid_caption_does_not_get_stringified_or_silently_scored():
    with pytest.raises(ValueError):
        caption_text({"answer": {"token_ids": [1]}})
