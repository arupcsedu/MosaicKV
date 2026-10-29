"""Observe pinned HF cache inputs, owners and concatenation replacements.

Private, single-thread inference only. Restores hooks on exit. This is not a
process-memory profiler and does not classify pre-RoPE activations as KV.
"""

from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

_CURRENT_LEDGER: ContextVar[Any | None] = ContextVar("mosaickv_cache_ownership_audit", default=None)
_DECODE_LEDGER: ContextVar[Any | None] = ContextVar(
    "mosaickv_decode_transaction_audit", default=None
)


def observe_kv_owners(role, pairs):
    """Register real cache owners at explicit, audited allocation boundaries."""
    ledgers = tuple(x for x in (_CURRENT_LEDGER.get(), _DECODE_LEDGER.get()) if x is not None)
    if ledgers:
        materialized = tuple(pairs)
        for ledger in ledgers:
            ledger.track(role, materialized)
            ledger.observe()
            ledger.allocation_operations[role] = ledger.allocation_operations.get(role, 0) + 1


def release_decode_owners(*roles):
    """A finished transaction's old buffers are not active decode owners.

    Global weak ownership deliberately remains: unused snapshots that really
    stay alive are still request storage, not silently hidden as active KV.
    """
    ledger = _DECODE_LEDGER.get()
    if ledger is not None:
        for role in roles:
            ledger._owners.pop(role, None)


class DynamicKVTransactions:
    def __init__(self, ledger):
        self.ledger = ledger
        self.update_count = 0
        self.concat_count = 0
        self._active = None
        self._stack = ExitStack()
        self.decode_ledger = None

    def begin_decode(self, cache):
        from mosaickv.measurements.kv_ownership import KVOwnershipLedger

        if self.decode_ledger is not None:
            raise RuntimeError("decode boundary already entered")
        self.decode_ledger = KVOwnershipLedger()
        token = _DECODE_LEDGER.set(self.decode_ledger)
        self._stack.callback(_DECODE_LEDGER.reset, token)
        self.observe_decode_cache(cache)

    def observe_decode_cache(self, cache):
        if self.decode_ledger is not None:
            self.decode_ledger.track("model_cache", self.pairs(cache))
            self.decode_ledger.observe()

    def observe(self, cache, **temporary):
        for ledger in (self.ledger, self.decode_ledger):
            if ledger is not None:
                ledger.track("model_cache", self.pairs(cache))
                ledger.observe(**temporary)

    @staticmethod
    def pairs(cache):
        return tuple(
            (layer.keys, layer.values)
            for layer in cache.layers
            if getattr(layer, "keys", None) is not None
            and getattr(layer, "values", None) is not None
        )

    def __enter__(self):
        import torch
        from transformers.cache_utils import Cache, DynamicCache

        original_update, original_cat = Cache.update, torch.cat
        token = _CURRENT_LEDGER.set(self.ledger)
        self._stack.callback(_CURRENT_LEDGER.reset, token)

        def update(cache, key, value, layer_idx, cache_kwargs=None):
            if not isinstance(cache, DynamicCache):
                raise RuntimeError("KV transaction audit supports DynamicCache only")
            if self._active is not None:
                raise RuntimeError("nested cache updates are outside the audited scope")
            self.update_count += 1
            self._active = (cache, key, value)
            try:
                self.observe(cache, cache_update_inputs=((key, value),))
                result = original_update(cache, key, value, layer_idx, cache_kwargs)
                self.observe(cache, cache_update_inputs=((key, value),))
                return result
            finally:
                self._active = None

        def cat(tensors, *args, **kwargs):
            result = original_cat(tensors, *args, **kwargs)
            if self._active is not None:
                cache, key, value = self._active
                self.concat_count += 1
                self.observe(
                    cache,
                    cache_update_inputs=((key, value),),
                    concat_result=((result, result),),
                )
            return result

        self._stack.enter_context(patch.object(Cache, "update", update))
        self._stack.enter_context(patch.object(torch, "cat", cat))
        return self

    def __exit__(self, *exc):
        self._active = None
        return self._stack.__exit__(*exc)

    def record(self):
        return {
            "cache_update_calls": self.update_count,
            "cache_concat_results_observed": self.concat_count,
            "boundary": "DynamicCache_inputs_owners_and_concat_results",
            "not_all_internal_prefill_activations_are_KV": True,
            "not_process_memory": True,
            "explicit_cache_allocation_operations": dict(self.ledger.allocation_operations),
            "decode_transaction_storage": (
                None
                if self.decode_ledger is None
                else dict(
                    self.decode_ledger.record(),
                    boundary="post_compression_KV_including_preallocation_concat_eviction_growth_inputs",
                )
            ),
        }
