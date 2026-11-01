from pathlib import Path
from unittest.mock import patch

import evicastkv_phase_capacity_diagnostic as driver


def test_frozen_ladder_all_methods_and_only_scoped_wrapper():
    assert driver.RATIOS == (0.5, 0.6, 0.65, 0.7)
    assert len(driver.VARIANTS) == 25
    assert len(driver.METHODS) == 5
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
    assert calls[0][1].endswith("/evicastkv_phase_capacity_diagnostic.sbatch")


def test_phase_kernel_shape_not_specialized_per_decode_token():
    from mosaickv.evaluation.phase_capacity_kernel import _rotate

    parameters = {p.name: p for p in _rotate.params}
    assert all(
        not parameters[n].is_constexpr and parameters[n].do_not_specialize
        for n in ("N", "SH", "SN")
    )
