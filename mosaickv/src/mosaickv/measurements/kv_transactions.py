"""Observe pinned HF cache inputs, owners and concatenation replacements.

Private, single-thread inference only. Restores hooks on exit. This is not a
process-memory profiler and does not classify pre-RoPE activations as KV.
"""

from contextlib import ExitStack
from unittest.mock import patch


class DynamicKVTransactions:
    def __init__(self, ledger):
        self.ledger = ledger
        self.update_count = 0
        self.concat_count = 0
        self._active = None
        self._stack = ExitStack()

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

        def update(cache, key, value, layer_idx, cache_kwargs=None):
            if not isinstance(cache, DynamicCache):
                raise RuntimeError("KV transaction audit supports DynamicCache only")
            if self._active is not None:
                raise RuntimeError("nested cache updates are outside the audited scope")
            self.update_count += 1
            self._active = (cache, key, value)
            try:
                self.ledger.track("model_cache", self.pairs(cache))
                self.ledger.observe(cache_update_inputs=((key, value),))
                result = original_update(cache, key, value, layer_idx, cache_kwargs)
                self.ledger.track("model_cache", self.pairs(cache))
                self.ledger.observe(cache_update_inputs=((key, value),))
                return result
            finally:
                self._active = None

        def cat(tensors, *args, **kwargs):
            result = original_cat(tensors, *args, **kwargs)
            if self._active is not None:
                cache, key, value = self._active
                self.concat_count += 1
                self.ledger.track("model_cache", self.pairs(cache))
                self.ledger.observe(
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
        }
