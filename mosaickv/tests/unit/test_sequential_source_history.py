from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
from mosaickv.evaluation.storage_runs import cache_from_formats


def raw(pair):
    return ExactFormat(
        "kv", tuple(pair[0].shape), pair=tuple(PackedBF16(tuple(x.shape), True, (x,)) for x in pair)
    )


def test_single_lease_restores_all_history_without_phase_or_position_changes():
    torch.manual_seed(821)
    prefix = tuple(torch.randn(1, 2, 8, 128).bfloat16() for _ in range(2))
    attention = torch.nn.Linear(4, 4, bias=False)
    layer = SequentialSourceHistoryLayer(
        raw(prefix), attention, observe=lambda *args: None, direct=True, updates=32, audit=False
    )
    reference = [[], []]
    for step in range(18):
        if step == 16:
            layer.source_sealed.append(raw(tuple(x.clone() for x in layer.history)))
            layer.tail_used = 0
        pair = tuple(torch.randn(1, 2, 1, 128).bfloat16() for _ in range(2))
        for store, x in zip(reference, pair, strict=True):
            store.append(x.clone())
        layer.source_pending = True
        leased = layer.update(*pair)
        assert leased[0].data_ptr() == leased[1].data_ptr()
        for component in (0, 1):
            layer.materialize_component(component)
            expected = torch.cat((prefix[component], *reference[component]), dim=2)
            assert torch.equal(layer.leased[0].view(torch.int16), expected.view(torch.int16))
        assert layer.get_seq_length() == 8 + step + 1
        layer.release()
    assert not layer.source_pending and layer.history_length == 18 and layer.tail_used == 2


@pytest.mark.parametrize("updates", [1, 32, 128, 592, 1024])
def test_explicit_factory_supports_full_teacher_targets(updates):
    prefix = tuple(torch.zeros(1, 2, 8, 128).bfloat16() for _ in range(2))
    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=torch.nn.Linear(4, 4))]
    )
    cache = cache_from_formats(
        adapter,
        (raw(prefix),),
        updates=updates,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        consolidated_history=True,
        source_history_constructor="direct_batched_checks",
    )
    assert isinstance(cache.layers[0], SequentialSourceHistoryLayer)
    assert cache.sequential_source_history and cache.layers[0].maximum_decode_updates == updates


@pytest.mark.parametrize(
    "change",
    [
        {"sequential_lease": False},
        {"source_history": False},
        {"sealed": False},
        {"consolidated_history": False},
        {"source_history_constructor": "reference"},
        {"updates": 1025},
        {"shared_workspace": True},
        {"segmented_history": True},
    ],
)
def test_unsupported_combinations_fail_explicitly(change):
    args = dict(
        updates=128,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        consolidated_history=True,
        source_history_constructor="direct_batched_checks",
    )
    args.update(change)
    with pytest.raises(ValueError):
        cache_from_formats(None, (), **args)


def test_existing_sealed_raw_combination_stays_unsupported_without_new_flag():
    with pytest.raises(ValueError):
        cache_from_formats(None, (), updates=128, audit=False, sealed=True, sequential_lease=True)


@pytest.mark.parametrize("different_logits", [False, True])
def test_paired_control_has_same_audit_and_rejects_any_changed_logit(monkeypatch, different_logits):
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.sequential_source_history import paired_run

    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([float(different_logits and len(calls) == 2)]), dict(
            storage={"initial_active_kv_storage_bytes": 32},
            decode_wall_seconds=1.0,
            decode_cuda_ms=1.0,
        )

    monkeypatch.setattr(storage_runs, "run", run)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    if different_logits:
        with pytest.raises(RuntimeError, match="original raw-history logit bit"):
            paired_run(None, None, (), sealed=False, audit_history=True)
    else:
        _, result = paired_run(None, None, (), sealed=False, audit_history=True)
        assert result["storage"]["raw_history_control"]["logits_bitwise"]
    assert all(c["audit"] and c["sequential_lease"] for c in calls)
    assert not calls[0]["sealed"] and calls[1]["consolidated_history"]
