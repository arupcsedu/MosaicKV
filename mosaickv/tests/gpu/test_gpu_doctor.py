from __future__ import annotations

from typing import cast

import pytest

from mosaickv.doctor import doctor_report
from mosaickv.types import JsonObject


@pytest.mark.gpu
def test_gpu_doctor_reports_visible_device_without_weights() -> None:
    report = doctor_report()
    cuda = cast("JsonObject", report["cuda"])
    if not cuda["available"]:
        pytest.skip("no GPU visible in this test process")
    gpu_count = cuda["gpu_count"]
    assert isinstance(gpu_count, int) and gpu_count >= 1
    backends = cast("JsonObject", report["backends"])
    for backend in backends.values():
        backend_record = cast("JsonObject", backend)
        assert backend_record["model_weights_loaded"] is False
