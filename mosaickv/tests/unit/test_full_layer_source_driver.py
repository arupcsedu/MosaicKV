from pathlib import Path
from unittest.mock import patch

import evicastkv_full_layer_source_feasibility as driver


def test_only_the_causal_wrapper_is_submitted():
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
    assert calls[0][1].endswith("/evicastkv_full_layer_source_feasibility.sbatch")
