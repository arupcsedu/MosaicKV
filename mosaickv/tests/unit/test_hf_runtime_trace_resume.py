from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel


def test_trace_path_preserves_orphaned_attempts(tmp_path: Path) -> None:
    runtime = object.__new__(HuggingFaceMosaicKVModel)
    runtime.trace_directory = tmp_path
    request = cast(Any, SimpleNamespace(run_id="run", sample_id="sample"))

    initial = runtime._trace_path(request)
    initial.parent.mkdir(parents=True)
    initial.write_text("{}\n", encoding="utf-8")
    retry_one = runtime._trace_path(request)
    retry_one.write_text("{}\n", encoding="utf-8")
    retry_two = runtime._trace_path(request)

    assert retry_one.name.endswith("-attempt-001.json")
    assert retry_two.name.endswith("-attempt-002.json")
    assert initial.read_text(encoding="utf-8") == "{}\n"
