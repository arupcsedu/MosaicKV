import json

import pytest

from mosaickv.evaluation.fox_campaign import sha
from mosaickv.evaluation.source_teacher_protocol import load_targets


def test_identity_annotation_provenance_and_untruncated_token_count(tmp_path):
    path = tmp_path / "ledger.json"
    sample = dict(sample_id="id", prompt="prompt", media=["image"], teacher_forcing_target="target")
    path.write_text(
        json.dumps(
            dict(samples=[sample], dataset_id="dataset", dataset_revision="revision", split="split")
        )
    )
    plan = dict(
        teacher_ledgers={"task": str(path)},
        bound_files={str(path): sha(path)},
        teacher_maximum_tokens=1024,
    )
    payload = dict(task="task", sample_id="id", prompt="prompt", media=["image"])

    def tokenizer(value, *, add_special_tokens):
        assert value == "target" and add_special_tokens is False
        return dict(input_ids=[1] * 593)

    ids, record = load_targets(plan, payload, tokenizer)
    assert len(ids) == 593 and record["dataset"]["dataset_revision"] == "revision"
    assert record["annotation_not_used_for_fitting"] and len(record["target_artifact_sha256"]) == 64
    with pytest.raises(ValueError, match="prompt/media"):
        load_targets(plan, dict(payload, prompt="different"), tokenizer)
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_targets(plan, payload, tokenizer)
