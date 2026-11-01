from pathlib import Path
from unittest.mock import patch

import evicastkv_modality_normalizer as driver


def test_all_strengths_are_fixed_before_execution_and_wrapper_exists():
    assert driver.STRENGTHS == {"zero": 0.0, "half": 0.5, "full": 1.0}
    script = driver.SOURCE / "mosaickv/slurm" / (Path(driver.__file__).stem + ".sbatch")
    assert script.exists()
    assert "evicastkv_modality_normalizer.py" in script.read_text()


def test_scheduler_reuses_controls_but_dispatches_only_the_current_driver():
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
    assert len(captured) == 1
    assert captured[0][1].endswith("/evicastkv_modality_normalizer.sbatch")
