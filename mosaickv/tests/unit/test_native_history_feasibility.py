import pytest

from mosaickv.evaluation.native_history_feasibility import ROWS, RepairBudgetError, validate_rows


def test_only_complete_bounded_epochs():
    assert tuple(range(16, 113, 16)) == ROWS
    for n in ROWS:
        validate_rows(n)
    for bad in (True, 16.0, 0, 15, 17, 128, -16):
        with pytest.raises(ValueError):
            validate_rows(bad)


def test_budget_failure_keeps_its_measured_diagnostic():
    diagnostic = dict(rows=16, repair_bytes=32769, registered_cap_bytes=32768)
    error = RepairBudgetError(diagnostic)
    assert error.diagnostic is diagnostic
    assert "unchanged" in str(error)
