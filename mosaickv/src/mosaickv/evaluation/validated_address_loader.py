"""CPU-only bulk validation; compile once outside home, allocate no GPU data."""

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as compile_extension

    source = Path(__file__).with_name("csrc") / "validated_addresses.cpp"
    digest = hashlib.sha256(source.read_bytes() + str(torch.__version__).encode()).hexdigest()[:16]
    root = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", "/scratch/djy8hg/cache/mosaickv/torch_extensions")
    )
    allowed = Path("/scratch/djy8hg").resolve()
    if not root.is_absolute() or not root.resolve().is_relative_to(allowed):
        raise ValueError("canonical task scratch native-build directory required")
    directory = root / ("validated_addresses_" + digest)
    directory.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MAX_JOBS")
    previous_path = os.environ.get("PATH")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + (previous_path or "")
    if previous is None:
        os.environ["MAX_JOBS"] = "2"
    try:
        return compile_extension(
            name="evicast_validated_addresses_" + digest,
            sources=[str(source)],
            build_directory=str(directory),
            extra_cflags=["-O2"],
            with_cuda=False,
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("MAX_JOBS", None)
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path


def addresses(arguments, *, device_type="cuda"):
    if not arguments or not isinstance(arguments[-1], torch.Tensor):
        raise ValueError("complete live tensor arguments required")
    if arguments[-1].device.type != device_type:
        raise ValueError("private nongrad buffers on one device required")
    if device_type not in ("cuda", "cpu"):
        raise ValueError("only CUDA execution or CPU ABI fixtures supported")
    current = torch.cuda.current_device() if device_type == "cuda" else -1
    return tuple(load().addresses(tuple(arguments), current))


def report():
    source = Path(__file__).with_name("csrc") / "validated_addresses.cpp"
    module = load()
    return dict(
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        GLIBCXX_CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        with_cuda_compiler=False,
        CXX=os.environ.get("CXX", "c++"),
        compiler_version=subprocess.check_output(
            [os.environ.get("CXX", "c++"), "--version"], text=True
        ).splitlines()[0],
        added_GPU_storage_bytes=0,
    )
