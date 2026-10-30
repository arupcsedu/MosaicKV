"""CPU instrumentation cleanup; no model performance measurements."""

import pytest

import mosaickv.evaluation.projection_cache as source
from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer
from mosaickv.evaluation.projection_profile import ranges


@pytest.mark.parametrize("fail", [True, False])
def test_profile_ranges_preserve_values_and_restore_methods_on_failure(monkeypatch, fail):
    value = object()

    def original(*args):
        return value

    monkeypatch.setattr(source, "reconstruct_archive", original)
    source_method = source.ProjectionLayer.materialize
    kv_method = PromptStorageLayer.materialize
    try:
        with ranges():
            assert source.reconstruct_archive(None, None) is value
            assert source.reconstruct_archive is not original
            if fail:
                raise RuntimeError("synthetic profile exit")
    except RuntimeError:
        assert fail
    assert source.reconstruct_archive is original
    assert source.ProjectionLayer.materialize is source_method
    assert PromptStorageLayer.materialize is kv_method
