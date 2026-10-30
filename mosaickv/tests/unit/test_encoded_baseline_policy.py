from types import SimpleNamespace

import evicastkv_encoded_baselines as module
import torch

import mosaickv.evaluation.direct_constructor as constructor
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import pack_cpu


def test_non_source_KV_values_are_not_reinterpreted_as_projection_inputs(monkeypatch):
    pair = tuple(pack_cpu(torch.randn(1, 2, 3, 16).bfloat16()) for _ in range(2))
    format = ExactFormat("kv", pair[0].shape, pair=pair)
    monkeypatch.setattr(constructor, "construct", lambda *a: ((format,), [{}]))
    result, proof = module.fused_choices(SimpleNamespace(), None, None, None, compact=True)
    assert result["common_encoded"] == (format,)
    assert not proof[0]["encoded_source"]["eligible"]


def test_promotion_requires_the_registered_engineering_decision(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(module, "VALIDATED", tmp_path)
    monkeypatch.setattr(module.base.common, "read", lambda _: dict(engineering_eligible=False))
    with pytest.raises(ValueError, match="numerical/memory/latency"):
        module.prepare(tmp_path / "new", "baselines")
