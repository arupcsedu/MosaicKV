from pathlib import Path
from unittest.mock import patch

import evicastkv_causal_head_diagnostic as driver


def test_only_own_finite_scheduler_and_frozen_single_head_variants():
    assert driver.COUNTS == {"shadow1": (1, False), "causal_selected1": (1, True)}
    commands = []

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
            lambda command, **kwargs: commands.append(command) or "123",
        ),
    ):
        driver.submit(Path("unused"), "unused")
    assert commands[0][1].endswith("/evicastkv_causal_head_diagnostic.sbatch")
