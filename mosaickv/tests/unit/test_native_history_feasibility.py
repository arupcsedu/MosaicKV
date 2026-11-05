import pytest

from mosaickv.evaluation.native_history_feasibility import ROWS, validate_rows


def test_only_complete_bounded_epochs():
    assert tuple(range(16, 113, 16)) == ROWS
    for n in ROWS:
        validate_rows(n)
    for bad in (True, 16.0, 0, 15, 17, 128, -16):
        with pytest.raises(ValueError):
            validate_rows(bad)
