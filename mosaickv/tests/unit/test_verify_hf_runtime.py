from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _verifier(package: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(package))
    path = package / "scripts/verify_hf_runtime.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_verify_hf_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_loader_uses_transformers_455_torch_dtype_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = Path(__file__).parents[2]
    verifier = _verifier(package, monkeypatch)

    kwargs = verifier._checkpoint_model_kwargs(
        Path("/scratch/djy8hg/cache/mosaickv/huggingface/hub"),
        allow_download=False,
    )

    assert kwargs["torch_dtype"] == "auto"
    assert "dtype" not in kwargs
    assert kwargs["local_files_only"] is True
    assert Path(kwargs["cache_dir"]).is_relative_to(Path("/scratch/djy8hg"))
