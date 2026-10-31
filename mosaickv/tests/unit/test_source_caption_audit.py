import copy

import pytest
from audit_source_caption_study import FIELDS, audit_identity


def fixture():
    anchor = dict(
        sample_id="consumed",
        method="mosaickv_exact",
        precision="balanced_boundary8",
        logits_sha256="actual",
        scored_answer={"answer": "same", "timed_tokens": 128},
        storage=dict.fromkeys(FIELDS, 100),
    )
    source = copy.deepcopy(anchor)
    source.update(precision="source_boundary8", storage=dict.fromkeys(FIELDS, 90))
    return [anchor, source]


def test_lossless_source_audit_keeps_actual_bytes_and_full_trajectory_boundary():
    checks = audit_identity(fixture())
    assert len(checks) == 1 and checks[0]["logits_128_bitwise"]
    assert checks[0]["storage"][FIELDS[0]]["saved_bytes"] == 10


@pytest.mark.parametrize("corrupt", ["hash", "answer", "storage", "missing", "duplicate"])
def test_source_audit_fails_closed(corrupt):
    rows = fixture()
    if corrupt == "hash":
        rows[1]["logits_sha256"] = "drift"
    elif corrupt == "answer":
        rows[1]["scored_answer"]["answer"] = "changed"
    elif corrupt == "storage":
        rows[1]["storage"][FIELDS[0]] = 101
    elif corrupt == "missing":
        rows.pop(0)
    else:
        rows.append(copy.deepcopy(rows[1]))
    with pytest.raises(ValueError):
        audit_identity(rows)
