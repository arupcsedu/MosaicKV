"""Read-only C++ allocator segment snapshot; no model math or GPU allocation."""

import hashlib
import os
from pathlib import Path


def load_reader():
    import torch
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "allocator_segments.cpp"
    identity = hashlib.sha256(source.read_bytes() + torch.__version__.encode()).hexdigest()[:16]
    root = Path(os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv"))
    if not str(root).startswith("/scratch/djy8hg/"):
        raise ValueError("scratch-only native audit cache required")
    build = root / "allocator-segments" / identity
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="evicastkv_allocator_" + identity,
        sources=[str(source)],
        build_directory=str(build),
        extra_include_paths=["/usr/local/cuda/include"],
        extra_ldflags=["-ltorch_cuda", "-lc10_cuda"],
        extra_cflags=["-O2"],
        with_cuda=False,
        verbose=False,
    )
