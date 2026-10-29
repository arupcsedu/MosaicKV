"""Synthetic separate-metric point gate; never experimental measurements."""

from copy import deepcopy

import evicastkv_multiscale_prompt as module
import pytest


def fixture(initial, final, peak, ppl, rouge):
    return dict(
        metrics=dict(zip(module.common.STORAGE, (initial, final, peak), strict=True)),
        audited_decode_transaction_peak_bytes=peak,
        teacher_forced_ppl=ppl,
        rouge_l_f1=rouge,
    )


@pytest.mark.parametrize("field", [None, "teacher_forced_ppl", "rouge_l_f1", "bytes"])
def test_gate_does_not_hide_one_metric_or_storage_failure(field):
    baseline = fixture(100, 150, 160, 3.0, 0.5)
    native = fixture(200, 300, 400, 3.1, 0.4)
    row = deepcopy(baseline)
    if field == "bytes":
        row["metrics"][module.common.STORAGE[0]] = 99
    elif field == "teacher_forced_ppl":
        row[field] = 4.0
    elif field == "rouge_l_f1":
        row[field] = 0.1
    check = module.point_check(row, baseline, native)
    assert all(check.values()) == (field is None)
