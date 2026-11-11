"""Pinned, one-shot native allocator callback; loading never initializes CUDA.

The extension intentionally remains mapped for process lifetime: PyTorch's
tracker API has no detach. Events are numeric observations, not KV ownership.
"""

import ctypes
import fcntl
import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

_TORCH_GIT = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
_HEADERS = {
    "c10/cuda/CUDACachingAllocator.h": (
        "c64b6427a8b25e15e60226e0ce3157fbf652abf767a0fd8e03d212f05d22e295"
    ),
    "c10/util/ApproximateClock.h": (
        "18dccd618afff0970f1f714069b253489bb21b99ad7044464a9258b96dd864c5"
    ),
}
_PINNED_HANDLES = []


def _load_recorder(*, testing=False):
    """The test build exposes injection; production never exposes that API."""
    import torch

    if torch.version.git_version != _TORCH_GIT or torch.__version__ != "2.8.0+cu128":
        raise RuntimeError("native trace ABI/source review is pinned to torch 2.8.0+cu128")
    if not hasattr(os, "RTLD_NODELETE"):
        raise RuntimeError("process-lifetime RTLD_NODELETE support required")
    source = Path(__file__).with_name("csrc") / "native_allocator_events.cpp"
    torch_root = Path(torch.__file__).parent
    header = torch_root / "include/c10/cuda/CUDACachingAllocator.h"
    header_hashes = {
        name: hashlib.sha256((torch_root / "include" / name).read_bytes()).hexdigest()
        for name in _HEADERS
    }
    if header_hashes != _HEADERS:
        raise RuntimeError("installed native allocator/clock headers differ from source review")
    header_sha = header_hashes["c10/cuda/CUDACachingAllocator.h"]
    identity = hashlib.sha256(
        source.read_bytes()
        + Path(__file__).read_bytes()
        + header_sha.encode()
        + str(torch._C._GLIBCXX_USE_CXX11_ABI).encode()
        + torch.__version__.encode()
        + _TORCH_GIT.encode()
        + bytes([bool(testing)])
    ).hexdigest()[:20]
    root = Path(os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv"))
    if not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("canonical scratch-only native audit cache required")
    build = root / "native-allocator-events" / identity
    build.mkdir(parents=True, exist_ok=True)
    name = "evicastkv_native_events_" + identity
    if name in sys.modules:
        return sys.modules[name]
    binary = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
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
                "-pthread",
                f"-DTORCH_EXTENSION_NAME={name}",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                "-I" + str(torch_root / "include"),
                "-I" + sysconfig.get_path("include"),
                "-I/usr/local/cuda/include",
                *(["-DMOSAICKV_NATIVE_ALLOCATOR_TESTING=1"] if testing else []),
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
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=120
            )
            if result.returncode:
                raise RuntimeError(
                    "native trace compiler failed:\n" + result.stdout + result.stderr
                )
            temporary.replace(binary)
    # The attached callback and its static storage must outlive Python modules.
    handle = ctypes.CDLL(str(binary), mode=os.RTLD_NOW | os.RTLD_NODELETE)
    _PINNED_HANDLES.append(handle)
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise ImportError("compiled native trace extension has no loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.metadata()["synthetic_test_build"] is not bool(testing):
        raise RuntimeError("native trace build-mode mismatch")
    if hasattr(module, "_test_only_capture") != bool(testing):
        raise RuntimeError("synthetic injection must not exist in production")
    module.build_provenance = {
        "torch_version": torch.__version__,
        "torch_git_version": _TORCH_GIT,
        "header_path": str(header),
        "header_sha256": header_sha,
        "reviewed_header_hashes": header_hashes,
        "cpp_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary": str(binary),
        "testing": bool(testing),
        "RTLD_NODELETE": True,
    }
    sys.modules[name] = module
    return module


def load_recorder():
    """Load only; call torch.cuda.init before start(capacity) on a GPU worker.

    Exactly one start/stop session is permitted per process. The native
    caching allocator must be enabled; memory-history recording is optional.
    Raw approximate timestamps are not CUDA time or wall-clock timestamps.
    """
    return _load_recorder(testing=False)
