import pytest
from evicastkv_projection_source_cpu import metadata_features, previous


def test_original_worker_provenance_fields_are_preserved_without_allowing_labels_or_draft():
    payload = dict(task="synthetic", sample_id="fixture", prompt="Describe.", media=[])
    assert metadata_features(payload) == payload
    assert previous.load_messages(metadata_features(payload))
    for forbidden in ("answer", "references", "fullkv_prefix_tokens"):
        with pytest.raises(ValueError):
            metadata_features(dict(payload, **{forbidden: "unavailable"}))
    with pytest.raises(ValueError):
        metadata_features(dict(prompt="Describe.", media=[]))
