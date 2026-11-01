import os
import subprocess
import sys

import pytest
import torch

from mosaickv.evaluation.precision_tier import pack


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("pattern", ["random", "constant", "signed_zero"])
def test_private_workspace_math_preserves_headers_codes_and_caller(bits, axis, pattern):
    x = torch.randn(1, 2, 37, 128, generator=torch.Generator().manual_seed(119)).bfloat16()
    if pattern == "constant":
        x.fill_(2.75)
    if pattern == "signed_zero":
        x.zero_()
        x[:, :, ::2, :] = -0.0
    snapshot = x.clone()
    old = pack(x, bits=bits, axis=axis, compact_workspace=False)
    actual = pack(x, bits=bits, axis=axis)
    for a, b in zip(old.tensors(), actual.tensors(), strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert torch.equal(snapshot.view(torch.int16), x.view(torch.int16))


def test_metadata_gradient_path_falls_back_to_legacy():
    x = torch.randn(1, 1, 8, 128).bfloat16().requires_grad_()
    actual = pack(x, bits=3, axis=2)
    (actual.minimum.sum() + actual.scale.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="explicit compact"):
        pack(x, bits=3, axis=2, compact_workspace=1)


def test_integer_serializer_cpu_interpreter_handles_padding_and_cross_row_words():
    program = """
import torch
from mosaickv.evaluation.quantizer_workspace_kernel import words_kernel,bytes_kernel
for length in (17,32,600,604):
 padded=((length+127)//128)*128
 for bits in (2,3,4,8):
  q=torch.randint(0,1<<bits,(8,padded),generator=torch.Generator().manual_seed(bits+length)).float()
  exact=q[:,:length].contiguous().to(torch.int64).flatten()
  output=torch.empty(exact.numel() if bits==8 else exact.numel()//8*bits,dtype=torch.uint8)
  if bits==8:
   bytes_kernel[((exact.numel()+1023)//1024,)](q,output,exact.numel(),length,padded,1024)
   expected=exact.byte()
  else:
   count=exact.numel()//8
   words_kernel[((count+127)//128,)](q,output,count,length,padded,bits,128)
   words=(exact.reshape(-1,8) << (torch.arange(8)*bits)).sum(-1)
   expected=torch.stack(tuple(((words>>(8*i))&255).byte() for i in range(bits)),-1).flatten()
  assert torch.equal(output,expected),(length,bits)
print('integer serialization CPU interpreter exact; not CUDA execution')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=dict(os.environ, TRITON_INTERPRET="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
