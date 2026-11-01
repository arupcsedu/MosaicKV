from pathlib import Path
from unittest.mock import patch

import evicastkv_draft_history_diagnostic as driver


def test_common_repair_controls_for_all_five_methods_and_own_finite_wrapper():
    assert len(driver.METHODS) == 5 and len(driver.VARIANTS) == 10
    assert all(
        driver.VARIANTS[m + "--shadow"] == (m, False)
        and driver.VARIANTS[m + "--repair"] == (m, True)
        for m in driver.METHODS
    )
    calls = []

    def scheduler(root, expected):
        driver.parent.subprocess.check_output(
            [
                "sbatch",
                str(driver.SOURCE / "mosaickv/slurm/evicastkv_funded_delimiter_diagnostic.sbatch"),
            ],
            text=True,
        )

    with (
        patch.object(driver.parent, "submit", scheduler),
        patch.object(
            driver.parent.subprocess,
            "check_output",
            lambda command, **kwargs: calls.append(command) or "123",
        ),
    ):
        driver.submit(Path("unused"), "unused")
    assert calls[0][1].endswith("/evicastkv_draft_history_diagnostic.sbatch")
