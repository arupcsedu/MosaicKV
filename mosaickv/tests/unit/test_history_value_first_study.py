from collections import Counter
from unittest.mock import patch

import header_history_timing as timing
import history_value_first_timing as study


def test_common_scheduling_and_prefill_are_applied_to_storage_comparator_too():
    a, b = study.POLICIES[study.ARMS[0]], study.POLICIES[study.ARMS[2]]
    assert a["history_value_first"] is b["history_value_first"] is True
    assert a["private_prefill_scores"] is b["private_prefill_scores"] is True
    assert a["paired_source"] is b["paired_source"] is True
    assert a["history"] is False and b["history"] is True
    assert not any(p.get("verification_barrier") for p in study.POLICIES.values())
    with patch.object(timing, "ARMS", study.ARMS):
        rows = timing.schedule(3)
    assert Counter(r["arm"] for r in rows) == Counter(dict.fromkeys(study.ARMS, 25))
