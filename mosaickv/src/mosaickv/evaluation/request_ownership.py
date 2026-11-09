"""Request-wide, audit-only semantic ownership over CUDA allocator history.

Allocations in ALL threads share one device trace. Only reviewed, exact stack
entries are classified. Unrecognized model temporaries and allocator workspace
remain unknown and block exact certification; a missing rule is not non-KV.
No default runtime policy is changed and this instrumentation is not timed.
"""

import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

from mosaickv.measurements.kv_timeline import CudaOwnershipSession
from mosaickv.measurements.ownership_review import RULES as REVIEWED_STACK_RULES
from mosaickv.measurements.ownership_review import review

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
    "exact_scale_replay.py": ("convert", "encode", "__init__"),
    "exact_scale_codec.py": ("to", "unpack_into"),
    "early_scale_headers.py": ("archive", "restore"),
    "mixed_history_projection.py": ("construct", "consolidate", "fill_component", "materialize"),
    "mixed_verification_barrier.py": ("flush",),
    "lossless_code_delta_gpu.py": ("encode", "decode"),
    "sequential_kv_lease.py": ("_new_lease", "materialize_component"),
    "source_history.py": ("update", "_seal_tail", "_initialize_storage"),
    "native_history_reader.py": ("construct",),
    "history_verification_workspace.py": ("verify",),
    "dual_projection.py": ("output_slots", "materialize", "build_dual"),
    "consolidated_history.py": ("merge_sources", "join_patches", "build_consolidated"),
    "encoded_projection_kernel.py": ("checkpoints",),
}
KV_FUNCTIONS["native_history_reader.py"] += ("phase_guard", "build_chunk", "consolidate", "repair")
STACK_RULES = tuple(
    ("/mosaickv/evaluation/" + filename, name, "kv_workspace")
    for filename, names in KV_FUNCTIONS.items()
    for name in names
)
STACK_RULES += (
    (
        "/transformers/models/llama/modeling_llama.py",
        "eager_attention_forward",
        "kv_shared_workspace",
    ),
    ("/transformers/models/clip/modeling_clip.py", "eager_attention_forward", "non_kv"),
    ("/transformers/models/llama/modeling_llama.py", "apply_rotary_pos_emb", "kv_shared_workspace"),
    ("/transformers/models/llama/modeling_llama.py", "rotate_half", "kv_shared_workspace"),
    ("/mosaickv/evaluation/exact_rotary_dispatch.py", "apply", "kv_shared_workspace"),
    ("/mosaickv/evaluation/streaming_prefill.py", "update", "kv_workspace"),
    ("/mosaickv/evaluation/request_precision.py", "ranked_positions", "kv_workspace"),
    ("/mosaickv/evaluation/source_history.py", "source_input", "kv_persistent"),
    ("/mosaickv/adapters/huggingface/base.py", "_greedy_token", "non_kv"),
    ("/mosaickv/adapters/huggingface/base.py", "_move_model_inputs", "non_kv"),
    ("/mosaickv/adapters/huggingface/base.py", "_decode_cache_position", "kv_workspace"),
    ("/torch/nn/functional.py", "silu", "non_kv"),
)
# These exact installed lines are LlamaRMSNorm's FP32 cast/variance/rsqrt/cast,
# not a broad "forward is non-KV" rule. Pin the file hash in each study plan.
STACK_RULES += tuple(
    ("/transformers/models/llama/modeling_llama.py", "forward", "non_kv", line)
    for line in (63, 64, 65, 66)
)
STACK_RULES += (("/mosaickv/evaluation/storage_runs.py", "run", "non_kv", 369),)
STACK_RULES += (("/mosaickv/measurements/kv_timeline.py", "_flush_cuda_events", "non_kv"),)
STACK_RULES += REVIEWED_STACK_RULES
STACK_RULES += tuple(
    ("/mosaickv/evaluation/" + filename, name, role)
    for filename, names, role in (
        ("packed_prototypes.py", ("construct", "selected_codes", "copy_packed"), "kv_workspace"),
        ("tiled_softmax.py", ("attention", "probabilities"), "kv_shared_workspace"),
        (
            "history_source_precision.py",
            ("consolidate", "join", "unpack_into", "fill_component"),
            "kv_workspace",
        ),
    )
    for name in names
)


@contextmanager
def allocator_lifetimes_only():
    """Replace redundant weakref scans, NOT numerical/stride/cache checks."""
    from mosaickv.evaluation.projection_cache import ScratchAudit

    def dispatch(mode, func, types, args=(), kwargs=None):
        mode.operations += 1
        return func(*args, **(kwargs or {}))

    with patch.object(ScratchAudit, "__torch_dispatch__", dispatch):
        yield


def cache_tensors(cache):
    """Use the actual common-cache ownership API, not a fabricated method."""
    if not hasattr(cache, "shared_tensors") or not hasattr(cache, "layers"):
        raise TypeError("explicit shared-tensor/layer ownership inventory required")
    return (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))


class _RequestSession(CudaOwnershipSession):
    def __init__(self, result, **kwargs):
        super().__init__(**kwargs)
        self.result = result  # metadata only; never a GPU owner
        self.source_receipts = []
        self.decode_started_us = None

    def __exit__(self, kind, value, trace):
        try:
            return super().__exit__(kind, value, trace)
        finally:
            receipts = []
            offset = getattr(self, "clock_translation", None)
            for receipt in self.source_receipts:
                key = (
                    None
                    if offset is None
                    else self.timeline.classify_at(
                        receipt["address"],
                        "kv_persistent",
                        "all-head original source before replacement",
                        receipt["begin_us"] - offset["maximum_offset_us"],
                        receipt["end_us"] - offset["minimum_offset_us"],
                    )
                )
                owner = self.timeline.allocations.get(key, {})
                retired = owner.get("end_time_us")
                receipts.append(
                    dict(
                        receipt,
                        allocation_generation=key,
                        retired_before_decode=retired is not None
                        and self.decode_started_us is not None
                        and retired <= self.decode_started_us - offset["maximum_offset_us"]
                        if offset
                        else False,
                    )
                )
            self.timeline.classify_stacks(STACK_RULES)
            all_classified = bool(self.timeline.allocations) and all(
                row["role"] != "unknown" for row in self.timeline.allocations.values()
            )
            classified = self.report(opaque_workspace_classified=all_classified)
            classified.update(
                classification_rule_version="exact_file_function_v1",
                unknown_allocations_not_assumed_non_KV=True,
                timing_instrumentation_excluded=True,
                all_head_source_retirement_receipts=receipts,
            )
            # A rejected short-lived timestamp mark assigns no owner. Complete
            # independent source-stack attribution may supersede that mark,
            # never a missing lifetime/history or an unclassified generation.
            self.result.update(review(classified))


@contextmanager
def audit_request(adapter, *, max_entries=200000):
    import torch

    from mosaickv.evaluation import (
        prefill_native_overlap,
        prompt_graph_reader,
        shared_head_scope,
        storage_runs,
        streaming_prefill,
        streaming_source_constructor,
    )
    from mosaickv.measurements.allocator_segments import load_reader

    native = load_reader()
    # Validate the SAME active addresses, unrounded requested bytes, and
    # rounded allocator bytes. Numerical graph metadata guards stay intact.
    import gc

    gc.collect()
    if prompt_graph_reader.allocator_blocks() != native.active_blocks():
        raise RuntimeError("native segment snapshot differs from installed Python API")

    report = {}
    session = _RequestSession(
        report, max_entries=max_entries, deferred_marks=True, clear_library_workspace=True
    )
    with session, ExitStack() as stack:
        stack.enter_context(allocator_lifetimes_only())
        stack.enter_context(
            patch.object(prompt_graph_reader, "allocator_blocks", native.active_blocks)
        )
        report["native_segment_snapshot_identity_verified"] = True
        session.mark(
            (*adapter.model.parameters(), *adapter.model.buffers()),
            "model",
            "pinned immutable resident model parameters/buffers",
        )
        kv_projections = {
            id(projection)
            for layer in adapter._language_layers()
            for projection in (layer.self_attn.k_proj, layer.self_attn.v_proj)
        }
        for module in adapter.model.modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                continue
            role = "kv_workspace" if id(module) in kv_projections else "non_kv"

            def output_owner(_module, _args, result, *, role=role):
                if isinstance(result, torch.Tensor):
                    session.mark(
                        (result,),
                        role,
                        "audited module identity: K/V projection"
                        if role == "kv_workspace"
                        else "audited non-KV linear/convolution/embedding output",
                    )

            handle = module.register_forward_hook(output_owner)
            stack.callback(handle.remove)
        original_update = streaming_prefill.RetiringLayer.update

        @wraps(original_update)
        def update(layer, key_states, value_states, *args, **kwargs):
            session.mark(
                (key_states, value_states), "kv_persistent", "original prefill K/V cache ownership"
            )
            return original_update(layer, key_states, value_states, *args, **kwargs)

        stack.enter_context(patch.object(streaming_prefill.RetiringLayer, "update", update))
        original_cache = storage_runs.cache_from_formats
        original_encoder = streaming_source_constructor.StreamingDirectSource.__call__

        @wraps(original_encoder)
        def encode_source(encoder, index, attention, original, args, kwargs, output):
            session.mark(
                (output[1],),
                "kv_shared_workspace",
                "prefill attention probabilities also consumed by source planner",
            )
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                session.mark(
                    (hidden,),
                    "kv_shared_workspace",
                    "normalized model input also used by source construction",
                )
            result = original_encoder(encoder, index, attention, original, args, kwargs, output)
            session.mark(
                result.tensors(),
                "kv_persistent",
                "constructed immutable prompt packet and shared source owners",
            )
            return result

        stack.enter_context(
            patch.object(
                streaming_source_constructor.StreamingDirectSource, "__call__", encode_source
            )
        )
        original_prototype = shared_head_scope.construct

        @wraps(original_prototype)
        def prototype(body, *args, **kwargs):
            begin = time.time_ns() // 1000
            address = body.codes.untyped_storage().data_ptr()
            end = time.time_ns() // 1000
            session.source_receipts.append(
                dict(
                    address=address,
                    begin_us=begin,
                    end_us=end,
                    KV_heads=kwargs["heads"],
                    components=["key", "value"],
                )
            )
            return original_prototype(body, *args, **kwargs)

        stack.enter_context(patch.object(shared_head_scope, "construct", prototype))

        @wraps(original_cache)
        def cache(*args, **kwargs):
            session.decode_started_us = time.time_ns() // 1000
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
