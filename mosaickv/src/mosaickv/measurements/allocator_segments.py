"""Read-only C++ allocator segment snapshot; no model math or GPU allocation."""

import fcntl
import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def load_reader():
    import torch

    source = Path(__file__).with_name("csrc") / "allocator_segments.cpp"
    identity = hashlib.sha256(
        source.read_bytes() + Path(__file__).read_bytes() + torch.__version__.encode()
    ).hexdigest()[:16]
    root = Path(os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv"))
    if not str(root).startswith("/scratch/djy8hg/"):
        raise ValueError("scratch-only native audit cache required")
    build = root / "allocator-segments" / identity
    build.mkdir(parents=True, exist_ok=True)
    name = "evicastkv_allocator_" + identity
    if name in sys.modules:
        return sys.modules[name]
    binary = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    torch_root = Path(torch.__file__).parent
    with (build / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            temporary = build / (name + f".{os.getpid()}.tmp.so")
            command = [
                os.environ.get("CXX", "/usr/bin/g++"),
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-O2",
                f"-DTORCH_EXTENSION_NAME={name}",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                "-I" + str(torch_root / "include"),
                "-I" + sysconfig.get_path("include"),
                "-I/usr/local/cuda/include",
                str(source),
                "-L" + str(torch_root / "lib"),
                "-Wl,-rpath," + str(torch_root / "lib"),
                "-lc10",
                "-lc10_cuda",
                "-ltorch_cuda",
                "-ltorch_cpu",
                "-ltorch_python",
                "-o",
                str(temporary),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    "native allocator compiler failed:\n" + result.stdout + result.stderr
                )
            temporary.replace(binary)
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise ImportError("compiled allocator extension has no loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module
