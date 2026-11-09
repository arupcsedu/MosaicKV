import json
from collections import Counter
from unittest.mock import patch

import prepared_address_timing as study
import pytest


def test_external_failure_or_missing_decision_blocks_GPU(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "BASELINES", tmp_path / "missing")
    monkeypatch.setattr(study, "original_cpu", lambda *a: pytest.fail("unsafe progression"))
    study.cpu(tmp_path, {})
    assert json.loads((tmp_path / "cpu.json").read_text())["status"] == "failed"


def test_common_dispatch_and_private_scores_are_identical_in_comparator():
    a, b = study.POLICIES[study.ARMS[0]], study.POLICIES[study.ARMS[2]]
    assert a["prepared_addresses"] is b["prepared_addresses"] is True
    assert a["private_prefill_scores"] is b["private_prefill_scores"] is True
    assert not a["history"] and b["history"]
    assert "verification_barrier" not in a and "verification_barrier" not in b
    with patch.object(study.timing, "ARMS", study.ARMS):
        rows = study.timing.schedule(3)
    assert Counter(r["arm"] for r in rows) == Counter(dict.fromkeys(study.ARMS, 25))
