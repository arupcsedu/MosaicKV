from pathlib import Path
from unittest.mock import patch

import evicastkv_last_tail_diagnostic as driver


def test_frozen_variants_and_own_scheduler_binding():
    assert driver.COUNTS == {"shadow1": (1, False), "tail_selected1": (1, True)}
    captured = []

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
            lambda command, **kwargs: captured.append(command) or "123",
        ),
    ):
        driver.submit(Path("unused"), "unused")
    assert captured[0][1].endswith("/evicastkv_last_tail_diagnostic.sbatch")
