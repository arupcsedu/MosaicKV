"""Version-pinned SGLang launcher for controlled Qwen media preprocessing.

SGLang 0.5.1.post3 hard-codes Qwen image resize bounds in
``sglang.srt.multimodal.processors.qwen_vl.resize_image``.  The ordinary
launcher consequently cannot reproduce the HF/lmms preprocessing profile.
This launcher changes those two module globals before ``TokenizerManager`` is
created; it does not modify the installed package or model weights.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys

_AUDITED_VERSION = "0.5.1.post3"
_MIN_ENV = "MOSAICKV_QWEN_MIN_PIXELS"
_MAX_ENV = "MOSAICKV_QWEN_MAX_PIXELS"


def _positive_int_environment(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"controlled SGLang launcher requires {name}")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer; received {raw!r}") from error
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1; received {value}")
    return value


def main() -> None:
    """Patch the audited preprocessing constants, then run public launch APIs."""

    version = importlib.metadata.version("sglang")
    if version != _AUDITED_VERSION:
        raise RuntimeError(
            f"controlled SGLang launcher audited {_AUDITED_VERSION}; installed {version}"
        )
    minimum = _positive_int_environment(_MIN_ENV)
    maximum = _positive_int_environment(_MAX_ENV)
    if minimum > maximum:
        raise RuntimeError(f"{_MIN_ENV} ({minimum}) must not exceed {_MAX_ENV} ({maximum})")

    # Preserve SGLang's ordinary launcher import order. Importing qwen_vl
    # directly before http_server initializes the quantization registry causes
    # an installed-source circular import through layers.linear/awq.
    from sglang.srt.entrypoints.http_server import launch_server  # type: ignore[import-untyped]
    from sglang.srt.server_args import prepare_server_args  # type: ignore[import-untyped]
    from sglang.srt.utils import kill_process_tree  # type: ignore[import-untyped]

    qwen_vl = importlib.import_module("sglang.srt.multimodal.processors.qwen_vl")

    qwen_vl.MIN_PIXELS = minimum  # type: ignore[attr-defined]
    qwen_vl.MAX_PIXELS = maximum  # type: ignore[attr-defined]
    if qwen_vl.resize_image.__globals__["MIN_PIXELS"] != minimum:
        raise RuntimeError("failed to bind SGLang Qwen minimum pixels")
    if qwen_vl.resize_image.__globals__["MAX_PIXELS"] != maximum:
        raise RuntimeError("failed to bind SGLang Qwen maximum pixels")

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()


__all__ = ["main"]
