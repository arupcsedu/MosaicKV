"""Request-wide, audit-only semantic ownership over CUDA allocator history.

Allocations in ALL threads share one device trace. Only reviewed, exact stack
entries are classified. Unrecognized model temporaries and allocator workspace
remain unknown and block exact certification; a missing rule is not non-KV.
No default runtime policy is changed and this instrumentation is not timed.
"""

from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

from mosaickv.measurements.kv_timeline import CudaOwnershipSession

# Allocation APIs reachable from these exact functions construct/read ONLY
# cache representation or KV reconstruction buffers. Do NOT use broad rules
# like "model.forward is non-KV" or "all baseline allocations are model".
KV_FUNCTIONS = {
    "precision_tier.py": ("pack",),
    "shared_head_prototypes.py": ("construct", "integers", "serialize", "promote"),
    "shared_head_scope.py": ("choose_group",),
    "conditional_precision.py": ("pack_unprotected",),
    "exact_storage.py": ("encode", "build"),
    "shared_source_format.py": (
        "construct",
        "fit_transport",
        "fit_transport_components",
        "transport_corrections",
        "validate_source_tensors",
        "assert_transport_close",
    ),
    "shared_source_kernel.py": ("decode_source_for_construction", "project", "decode_chunk"),
    "prefill_native_overlap.py": ("fit",),
    "lt_source_format.py": ("fit", "verify_strides"),
    "all_head_source_gpu.py": ("encode", "decode", "encoder"),
    "predictive_source_gpu.py": ("encoder",),
    "layer_code_delta.py": ("convert", "__init__", "body"),
    "lossless_code_delta_gpu.py": ("encode", "decode"),
    "sequential_kv_lease.py": ("_new_lease", "materialize_component"),
    "source_history.py": ("update", "_seal_tail", "_initialize_storage"),
    "native_history_reader.py": ("construct",),
}
STACK_RULES = tuple(
    ("/mosaickv/evaluation/" + filename, name, "kv_workspace")
    for filename, names in KV_FUNCTIONS.items()
    for name in names
)


def cache_tensors(cache):
    """Use the actual common-cache ownership API, not a fabricated method."""
    if not hasattr(cache, "shared_tensors") or not hasattr(cache, "layers"):
        raise TypeError("explicit shared-tensor/layer ownership inventory required")
    return (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))


class _RequestSession(CudaOwnershipSession):
    def __init__(self, result, **kwargs):
        super().__init__(**kwargs)
        self.result = result  # metadata only; never a GPU owner

    def __exit__(self, kind, value, trace):
        try:
            return super().__exit__(kind, value, trace)
        finally:
            self.timeline.classify_stacks(STACK_RULES)
            self.result.update(self.report(opaque_workspace_classified=False))
            self.result.update(
                classification_rule_version="exact_file_function_v1",
                unknown_allocations_not_assumed_non_KV=True,
                timing_instrumentation_excluded=True,
            )


@contextmanager
def audit_request(adapter, *, max_entries=200000):
    from mosaickv.evaluation import prefill_native_overlap, storage_runs, streaming_prefill

    report = {}
    session = _RequestSession(report, max_entries=max_entries)
    with session, ExitStack() as stack:
        session.mark(
            (*adapter.model.parameters(), *adapter.model.buffers()),
            "model",
            "pinned immutable resident model parameters/buffers",
        )
        original_update = streaming_prefill.RetiringLayer.update

        @wraps(original_update)
        def update(layer, key_states, value_states, *args, **kwargs):
            session.mark(
                (key_states, value_states), "kv_persistent", "original prefill K/V cache ownership"
            )
            return original_update(layer, key_states, value_states, *args, **kwargs)

        stack.enter_context(patch.object(streaming_prefill.RetiringLayer, "update", update))
        original_cache = storage_runs.cache_from_formats

        @wraps(original_cache)
        def cache(*args, **kwargs):
            result = original_cache(*args, **kwargs)
            session.mark(
                cache_tensors(result),
                "kv_persistent",
                "private decode cache payload, maps, history and graph ownership",
            )
            return result

        stack.enter_context(patch.object(storage_runs, "cache_from_formats", cache))
        original_fit = prefill_native_overlap.Pipeline.fit

        @wraps(original_fit)
        def fit(pipeline, *args, **kwargs):
            result = original_fit(pipeline, *args, **kwargs)
            if pipeline.engine is not None:
                session.mark(
                    (
                        pipeline.engine.source,
                        pipeline.engine.output,
                        *pipeline.engine.extra_tensors,
                    ),
                    "kv_workspace",
                    "async constructor engine workspace on its owning worker thread",
                )
            return result

        stack.enter_context(patch.object(prefill_native_overlap.Pipeline, "fit", fit))
        yield report
