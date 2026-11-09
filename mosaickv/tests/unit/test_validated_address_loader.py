import os
import weakref

import pytest
import torch

from mosaickv.evaluation.prepared_address_dispatch import addresses as reference
from mosaickv.evaluation.validated_address_loader import addresses, load, report


@pytest.mark.parametrize("count", [3, 24])
def test_all_actual_buffer_addresses_match_reference_without_retaining_owners(count):
    buffers = tuple(
        torch.empty(0 if i % 5 == 0 else 17 + i, dtype=torch.uint8) for i in range(count)
    )
    assert addresses(buffers, device_type="cpu") == reference(buffers, device_type="cpu")
    record = report()
    assert record["added_GPU_storage_bytes"] == 0 and not record["with_cuda_compiler"]
    assert len(record["source_sha"]) == len(record["binary_sha"]) == 64
    refs = tuple(weakref.ref(t) for t in buffers)
    del buffers
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("case", ["empty", "wrong_type", "grad", "wrong_device"])
def test_bulk_validation_preserves_every_failure_contract(case):
    args = (torch.ones(2), torch.ones(4), torch.ones(8))
    if case == "empty":
        args = ()
    elif case == "wrong_type":
        args = (None, *args[1:])
    elif case == "grad":
        args = (torch.ones(2, requires_grad=True), *args[1:])
    with pytest.raises(ValueError):
        addresses(args, device_type="cuda" if case == "wrong_device" else "cpu")



@pytest.mark.parametrize("kind", ["relative", "outside", "symlink"])
def test_native_build_rejects_non_scratch_and_escaped_paths(kind, tmp_path, monkeypatch):
    root = "relative" if kind == "relative" else "/usr/forbidden-native-build"
    if kind == "symlink":
        link = tmp_path / "outside"
        link.symlink_to("/usr", target_is_directory=True)
        root = str(link / "forbidden-native-build")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", root)
    load.cache_clear()
    with pytest.raises(ValueError, match="canonical task scratch"):
        load()
    load.cache_clear()


def test_build_does_not_change_caller_environment():
    before = {key: os.environ.get(key) for key in ("PATH", "MAX_JOBS")}
    load()
    assert before == {key: os.environ.get(key) for key in before}
