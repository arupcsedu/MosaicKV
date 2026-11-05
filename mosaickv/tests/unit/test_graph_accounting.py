import pytest
import torch

from mosaickv.evaluation import graph_accounting as accounting
from mosaickv.evaluation.prompt_graph_reader import (
    PromptGraphCache,
    PromptGraphLayer,
    TrackedPromptGraphLayer,
)


def fixture(policy, *, audit=False):
    with accounting.use(policy):
        cache = PromptGraphCache(audit_workspace=audit)
        cls = accounting.layer_class()
        for size in (100, 200, 300):
            layer = cls.__new__(cls)
            layer.leased = ()
            layer.prompt_graph_pool = cache.prompt_graph_pool
            layer.resident_bytes = size
            layer.lease_bytes = 0
            cache.layers.append(layer)
        cache.shared_bytes = 128
        cache.prompt_graph_pool.private_live_bytes = 16
        cache.observe()
    return cache


@pytest.mark.parametrize("audit", [True, False])
def test_single_pass_and_event_totals_match_original_at_every_transition(audit):
    caches = [fixture(policy, audit=audit) for policy in accounting.POLICIES]
    extras = torch.ones(33, dtype=torch.uint8)
    for step in range(100):
        for cache in caches:
            layer = cache.layers[step % 3]
            layer.resident_bytes += 7 if step % 2 else -3
            layer.leased = (object(),)
            layer.lease_bytes = 8192  # Shared backing, never counted per alias.
            cache.observe((extras, extras))
            layer.leased = ()
            layer.lease_bytes = 0
            cache.shared_bytes += 2
            cache.observe()
        assert len({c.peak_bytes for c in caches}) == 1
        assert len({c.observations for c in caches}) == 1
        assert caches[-1].accounting_total == sum(
            x.resident_bytes + x.lease_bytes for x in caches[-1].layers
        )
    assert caches[-1].accounting_notifications > 100
    assert (caches[-1].accounting_audits > 0) is audit


def test_audit_detects_unannounced_mutation_and_final_check_rejects_layer_replacement():
    cache = fixture("event_sum", audit=True)
    cache.layers[0]._resident_bytes += 1
    with pytest.raises(RuntimeError, match="independent"):
        cache.observe()
    cache = fixture("event_sum")
    cache.layers[0], cache.layers[1] = cache.layers[1], cache.layers[0]
    with pytest.raises(RuntimeError, match="independent"):
        cache.storage_record()


def test_unknown_or_shared_layers_rejected_and_context_restored():
    a, b = fixture("event_sum"), fixture("event_sum")
    b.layers.append(a.layers[0])
    with pytest.raises(RuntimeError, match="unique private"):
        b.observe()
    with pytest.raises(RuntimeError), accounting.use("event_sum"):
        assert accounting.layer_class() is TrackedPromptGraphLayer
        with pytest.raises(RuntimeError), accounting.use("single_pass"):
            pass
        raise RuntimeError("abort")
    assert accounting.current() == "original"
    assert accounting.layer_class() is PromptGraphLayer
    assert "resident_bytes" not in PromptGraphLayer.__dict__  # No old-control getter slowdown.


@pytest.mark.parametrize("value", [True, "", "estimated", None])
def test_unregistered_policy_rejected(value):
    with pytest.raises(ValueError), accounting.use(value):
        pass


def test_new_layers_bind_once_and_negative_totals_fail_closed():
    cache = fixture("event_sum")
    cache.observe()
    total = cache.accounting_total
    cache.observe()
    assert cache.accounting_total == total
    with pytest.raises(RuntimeError):
        cache.accounting_delta(-total - 1)
    with pytest.raises(RuntimeError):
        cache.accounting_delta(1.5)
    cache.layers.pop()
    with pytest.raises(RuntimeError, match="shrink"):
        cache.observe()


@pytest.mark.parametrize("value", [True, -1, 1.5, float("nan")])
def test_invalid_owner_byte_updates_rejected_before_counter_mutation(value):
    cache = fixture("event_sum")
    old = cache.accounting_total
    for field in ("resident_bytes", "lease_bytes"):
        with pytest.raises(RuntimeError):
            setattr(cache.layers[0], field, value)
        assert cache.accounting_total == old


def test_real_cache_factory_selects_policy_class_and_keeps_tensor_storage():
    from types import SimpleNamespace

    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.storage_runs import cache_from_formats

    pair = tuple(torch.randn(1, 2, 9, 128).bfloat16() for _ in (0, 1))
    packet = ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=torch.nn.Linear(256, 256))]
    )
    records = []
    for policy in accounting.POLICIES:
        with accounting.use(policy):
            cache = cache_from_formats(
                adapter,
                (packet,),
                updates=8,
                audit=True,
                sealed=True,
                source_history=True,
                source_history_constructor="direct_batched_checks",
                consolidated_history=True,
                sequential_lease=True,
                sequential_source_history=True,
                prompt_graph_reader=True,
            )
        assert isinstance(cache.layers[0], TrackedPromptGraphLayer) is (policy == "event_sum")
        records.append(cache.storage_record())
    for name in (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    ):
        assert len({r[name] for r in records}) == 1
