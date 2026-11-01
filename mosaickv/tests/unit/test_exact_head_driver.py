from pathlib import Path
from unittest.mock import patch

import evicastkv_exact_head_diagnostic as driver


def test_counts_and_shadow_are_frozen_before_results():
    assert driver.COUNTS == {
        "shadow1": (1, False),
        "heads1": (1, True),
        "heads4": (4, True),
        "heads8": (8, True),
    }
    script = driver.SOURCE / "mosaickv/slurm" / (Path(driver.__file__).stem + ".sbatch")
    assert script.exists() and "evicastkv_exact_head_diagnostic.py" in script.read_text()


def test_scheduler_dispatches_only_exact_head_driver():
    captured = []

    def scheduler(root, expected):
        driver.parent.subprocess.check_output(
            [
                "sbatch",
                str(driver.SOURCE / "mosaickv/slurm/evicastkv_funded_delimiter_diagnostic.sbatch"),
                "worker",
            ],
            text=True,
        )

    def invoked(command, **kwargs):
        captured.append(command)
        return "123"

    with (
        patch.object(driver.parent, "submit", scheduler),
        patch.object(driver.parent.subprocess, "check_output", invoked),
    ):
        driver.submit(Path("unused"), "unused")
    assert captured[0][1].endswith("/evicastkv_exact_head_diagnostic.sbatch")
