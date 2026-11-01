from pathlib import Path
from unittest.mock import patch

import evicastkv_coherent_token_diagnostic as driver


def test_frozen_anchor_and_one_token_variants_and_bound_wrapper():
    assert driver.VARIANTS == ("anchor", "coherent")
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
    assert calls[0][1].endswith("/evicastkv_coherent_token_diagnostic.sbatch")
