import pytest
import torch

from mosaickv.evaluation import exact_rotary_dispatch as rotary


def test_common_dispatch_is_default_off_and_does_not_patch_the_module():
    from transformers.models.llama import modeling_llama as llama

    original = llama.apply_rotary_pos_emb
    with rotary.transport():
        assert llama.apply_rotary_pos_emb is original
    assert rotary._scope.get() is None


def test_explicit_scope_does_not_silently_accept_missing_rotary_calls():
    @rotary.managed
    def call(**kw):
        return [], [], {}

    assert not call()[2]["exact_rotary_dispatch"]["enabled"]
    with pytest.raises(ValueError, match="boolean"):
        call(exact_rotary_dispatch=1)
    with pytest.raises(RuntimeError, match="engagement"):
        call(exact_rotary_dispatch=True)
    assert rotary._scope.get() is None


def test_scoped_model_function_is_restored_even_after_failure():
    from transformers.models.llama import modeling_llama as llama

    original = llama.apply_rotary_pos_emb
    token = rotary._scope.set(dict(enabled=True, calls=0))
    try:
        with pytest.raises(RuntimeError, match="planned"), rotary.transport():
            assert llama.apply_rotary_pos_emb is rotary.apply
            raise RuntimeError("planned")
        assert llama.apply_rotary_pos_emb is original
    finally:
        rotary._scope.reset(token)


def test_unsupported_cpu_geometry_fails_before_any_triton_launch():
    query = torch.zeros(1, 32, 1, 128, dtype=torch.bfloat16)
    phases = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="geometry"):
        rotary.apply(query, query, phases, phases)
