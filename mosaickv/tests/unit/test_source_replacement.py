import gc
import weakref
from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.source_replacement import compact, maybe_compact, use


@pytest.mark.parametrize("n,width", ((17, 64), (128, 128), (129, 64), (603, 128)))
def test_physical_replacement_is_bitwise_with_original_headers_and_positions(n, width):
    torch.manual_seed(4)
    values = torch.randn(1, 1, n, width).bfloat16()
    omitted = torch.arange(n) % 3 == 0
    body = pack_unprotected(values, omitted[None], bits=3, axis=2)
    expected = unpack_cpu(body)
    result, proof = compact(body, omitted)
    assert torch.equal(expected.view(torch.int16), unpack_cpu(result).view(torch.int16))
    assert result.minimum is body.minimum and result.scale is body.scale
    assert result.shape == body.shape
    assert proof["saved_bytes"] == backing_bytes(body.tensors()) - backing_bytes(result.tensors())
    assert proof["saved_bytes"] > 0
    assert result.codes.untyped_storage().nbytes() == result.codes.numel()
    assert result.row_map.untyped_storage().nbytes() == n * 4


def test_returned_format_has_no_old_code_owner():
    body = pack(torch.zeros(1, 1, 32, 128).bfloat16(), bits=3, axis=2)
    original = weakref.ref(body.codes)
    result, _ = compact(body, torch.ones(32, dtype=torch.bool))
    del body
    gc.collect()
    assert original() is None
    assert torch.count_nonzero(unpack_cpu(result)) == 0
    assert result.stored_rows == 1  # charged dummy payload, never dereferenced


def test_nonzero_codes_cannot_be_silently_retired():
    body = pack(torch.arange(32 * 128).reshape(1, 1, 32, 128).bfloat16(), bits=3, axis=2)
    with pytest.raises(ValueError, match="nonzero"):
        compact(body, torch.ones(32, dtype=torch.bool))


def test_map_cost_can_make_a_small_replacement_unprofitable():
    body = pack(torch.zeros(1, 1, 603, 8).bfloat16(), bits=3, axis=2)
    same, proof = compact(body, torch.arange(603) == 0)
    assert same is body and not proof["replaced"]


def test_default_and_retention_identity_scopes_do_not_transform_sources():
    body = pack(torch.zeros(1, 1, 32, 128).bfloat16(), bits=3, axis=2)
    omitted = torch.ones(32, dtype=torch.bool)
    assert maybe_compact(body, omitted) is body
    with use(False) as proof:
        assert maybe_compact(body, omitted) is body
        assert not proof["layers"]
    with use(True) as proof:
        assert maybe_compact(body, omitted) is not body
        assert len(proof["layers"]) == 1
    assert maybe_compact(body, omitted) is body


def test_corrupt_map_is_rejected():
    body = pack(torch.zeros(1, 1, 32, 128).bfloat16(), bits=3, axis=2)
    result, _ = compact(body, torch.ones(32, dtype=torch.bool))
    with pytest.raises(ValueError, match="invalid"):
        unpack_cpu(replace(result, row_map=torch.full((32,), 9, dtype=torch.int32)))


def test_real_retirement_audit_checks_objects_and_counts_map_bytes():
    from mosaickv.evaluation.source_replacement import audit_retirement

    body = pack(torch.zeros(1, 1, 32, 128).bfloat16(), bits=3, axis=2)
    with use(True) as proof:
        result = maybe_compact(body, torch.ones(32, dtype=torch.bool))
        del body
        gc.collect()
        audit_retirement(proof)
        assert proof["replaced_code_owners"] == 1
        assert proof["old_code_tensor_objects_alive"] == 0
        assert proof["saved_gpu_bytes"] > 0
        assert not proof["whole_request_peak_certified"]
        assert result.row_map is not None


def test_replacement_variants_are_isolated_and_common_parent_is_measured():
    from storage_six_ablation import component_policies

    policies = component_policies(replacement_study=True)
    assert policies["source_replaced_parent"]["enabled"] is False
    assert policies["source_replaced"]["replace_source_codes"]
    assert not policies["full"]["replace_source_codes"]
    assert policies["signed_gain"]["routing_signal"] == "value_energy"
    assert not policies["signed_gain"]["demand_weighted_coverage"]
    assert policies["actual_attention_risk"]["prototype_risk_threshold"] == 0.01


@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("tail", (False, True))
def test_actual_parent_192_wrapper_binds_complete_kernel_abi(monkeypatch, mapped, tail):
    import inspect
    from types import SimpleNamespace

    from mosaickv.evaluation import source_192_feasibility as reader
    from mosaickv.evaluation.paper_geometry import registered_geometry

    calls = []
    name = "_decode_joint" if tail else "_decode"
    original = getattr(reader, name)

    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                options = {
                    k: v for k, v in kwargs.items() if k not in ("num_warps", "enable_fp_fusion")
                }
                bound = inspect.signature(original.fn).bind(*args, **options)
                calls.append(bound.arguments)
                return SimpleNamespace(metadata=SimpleNamespace(global_scratch_size=0), n_spills=0)

            return call

    monkeypatch.setattr(reader, name, Kernel())
    codes = torch.empty(1, dtype=torch.uint8)
    mapping = torch.empty(603, dtype=torch.int32) if mapped else None
    body = SimpleNamespace(
        shape=(1, 1, 603, 4096),
        codes=codes,
        minimum=codes,
        scale=codes,
        row_map=mapping,
        stored_rows=402 if mapped else 0,
    )
    packet = SimpleNamespace(source=body)
    out = torch.empty(192, 4096, dtype=torch.bfloat16)
    with registered_geometry(603):
        reader.decode(
            packet,
            out,
            576 if tail else 0,
            27 if tail else 192,
            192,
            torch.empty(4096, dtype=torch.bfloat16) if tail else None,
        )
    assert len(calls) == 1
    assert calls[0]["HAS_ROW_MAP"] is mapped
    assert calls[0]["ROW_MAP"] is (mapping if mapped else codes)
    assert calls[0]["SOURCE_ROWS"] == (402 if mapped else 0)
