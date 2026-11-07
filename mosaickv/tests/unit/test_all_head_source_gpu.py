import pytest
import torch

from mosaickv.evaluation.all_head_source_gpu import CodecChoice, DeviceSymbols, decode
from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes


def test_selector_backing_storage_is_not_free():
    payload = torch.arange(128, dtype=torch.int16).byte()
    choice = CodecChoice(DeviceCodes(128, payload), torch.tensor([0], dtype=torch.uint8))
    assert choice.raw
    assert choice.length == 128
    assert choice.serialized_bytes == 129
    assert choice.tensors()[-1].numel() == 1


def test_forged_cpu_or_incomplete_descriptor_rejected_before_kernel_launch():
    packet = DeviceSymbols(
        384, torch.zeros(10, dtype=torch.uint8), torch.zeros(3, dtype=torch.int32), False
    )
    with pytest.raises(ValueError, match="encoder-produced CUDA"):
        decode(None, packet)
    with pytest.raises(ValueError, match="source symbol packet"):
        decode(None, object())
