"""Common inference-only eager transport: reuse one backing buffer for K then V.

Only private request caches and audited Llama/MHA eager attention are allowed.
Selections, stored formats, original positions and attention math do not change.
K is overwritten only after QK/softmax complete on the same CUDA stream.
"""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mosaickv.evaluation.exact_storage import ExactFormat, ExactLayer, decoded
from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.precision_tier import PrecisionFormat
from mosaickv.evaluation.prompt_source_subphases import phase as source_phase
from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.text_source_storage import TextSourceFormat


def fill_component(packet, attention, component, destination, *, source_workspace=None):
    if type(component) is not int or component not in (0, 1):
        raise ValueError("key or value component required")
    from mosaickv.evaluation.hybrid_reader import HybridFormat

    if isinstance(packet, HybridFormat):
        packet.fill_component(attention, component, destination, source_workspace=source_workspace)
        return
    from mosaickv.evaluation.layer_code_delta import BoundDeltaFormat

    if isinstance(packet, BoundDeltaFormat):
        packet.fill_component(attention, component, destination, source_workspace=source_workspace)
        return
    from mosaickv.evaluation.full_prompt_source import FullPromptSourceFormat

    if isinstance(packet, FullPromptSourceFormat):
        packet.fill_component(attention, component, destination, source_workspace)
        return
    from mosaickv.evaluation.encoded_projection import EncodedFormat
    from mosaickv.evaluation.source_hotpath import restore_enabled, skip_prefix

    with source_phase("indexing_views"):
        n = packet.shape[-2]
        unused = not isinstance(packet, (EncodedFormat, TextSourceFormat)) and not (
            isinstance(packet, SharedSourceFormat) and destination.device.type != "cuda"
        )
        prefix = None if unused and skip_prefix() else destination[..., :n, :]

    if isinstance(packet, EncodedFormat):
        if (
            packet.archive is None
            or packet.pair
            or (packet.slots is not None and packet.slots.numel())
        ):
            raise NotImplementedError(
                "only complete encoded history without direct overlays supported"
            )
        if destination.device.type != "cuda":
            raise NotImplementedError("encoded sequential history requires actual CUDA")
        from mosaickv.evaluation.encoded_projection_kernel import project
        from mosaickv.evaluation.shared_source_format import apply_transport_patch

        project(
            packet.archive.source,
            packet.literal_checkpoints,
            attention,
            packet.archive,
            packet.output_slots,
            (prefix, prefix),
            component=component,
            packed_counts=packet.packed_literal_counts,
        )
        if packet.patches:
            apply_transport_patch(packet.patches[component], destination, prompt_length=n)
        return
    from mosaickv.evaluation.sparse_original_exceptions import ExceptionFormat

    if isinstance(packet, ExceptionFormat):
        from mosaickv.evaluation.shared_source_format import apply_transport_patch

        fill_component(
            packet.base, attention, component, destination, source_workspace=source_workspace
        )
        apply_transport_patch(packet.patches[component], destination, prompt_length=n)
        return
    if isinstance(packet, TextSourceFormat):
        fill_component(
            packet.base, attention, component, destination, source_workspace=source_workspace
        )
        projection = packet.projection
        if destination.device.type == "cuda":
            from mosaickv.evaluation.encoded_projection_kernel import project

            archive = SimpleNamespace(
                selected=SimpleNamespace(shape=packet.shape[1:3]),
                union=projection.union,
                cos=projection.cos,
                sin=projection.sin,
            )
            project(
                projection.source,
                projection.checkpoints,
                attention,
                archive,
                projection.slots,
                (prefix, prefix),
                component=component,
            )
            if projection.patches:
                from mosaickv.evaluation.dual_projection_kernel import apply_patch

                apply_patch(projection.patches[component], prefix)
        else:
            # CPU correctness oracle only, not a GPU memory measurement.
            other = torch.empty_like(prefix)
            targets = (prefix, other) if component == 0 else (other, prefix)
            projection.fill(attention, targets)
    elif isinstance(packet, HeadLocalFormat):
        fill_component(packet.base, attention, component, destination)
        if packet.protected:
            body = packet.protected[component]
            if destination.device.type == "cuda":
                from mosaickv.evaluation.head_local_precision_kernel import scatter

                scatter(body, packet.flat_tokens, destination, prompt_length=n)
            else:
                ids = packet.flat_tokens.long()
                destination[0, ids // n, ids % n] = decoded(body)[0, 0]
    elif isinstance(packet, PrecisionFormat):
        selected = replace(
            packet,
            pair=(packet.pair[component],),
            exact_values=(packet.exact_values[component],),
            corrections=(packet.corrections[component],) if packet.corrections else (),
        )
        selected.fill(attention, (destination,), direct=True)
    elif isinstance(packet, SharedSourceFormat):
        if destination.device.type == "cuda":
            from mosaickv.evaluation.head_local_precision_kernel import scatter
            from mosaickv.evaluation.shared_source_kernel import project

            if restore_enabled() and (
                packet.projection_backend != "fused"
                or packet.protected_patches_disjoint is not True
            ):
                raise NotImplementedError("fused restore requires verified disjoint source patches")

            if packet.projection_backend == "native_bmm_256":
                from mosaickv.evaluation.native_source_chunks import project_component

                if source_workspace is None:
                    raise RuntimeError("owned native source workspace required")
                project_component(packet, attention, component, destination, source_workspace)
            else:
                project(packet, attention, (destination, destination), component=component)
            if restore_enabled():
                from mosaickv.evaluation.fused_source_restore import restore

                restore(packet, component, destination)
                return
            scatter(packet.exact[component], packet.row_positions, destination, prompt_length=n)
            if packet.transport_patches:
                from mosaickv.evaluation.shared_source_format import apply_transport_patch

                apply_transport_patch(
                    packet.transport_patches[component], destination, prompt_length=n
                )
        else:
            prefix.copy_(packet.materialize(attention)[component])
    elif isinstance(packet, ExactFormat) and packet.kind == "kv":
        replace(packet, pair=(packet.pair[component],)).fill(attention, (destination,), direct=True)
    else:
        raise NotImplementedError("unaudited format cannot use a sequential KV lease")


class SequentialLayer(ExactLayer):
    def owned_tensors(self):
        source = getattr(self, "source_workspace", None)
        return (*super().owned_tensors(), *((source,) if source is not None else ()))

    def release(self):
        self.source_workspace = None
        super().release()

    def _new_lease(self, shape, source):
        lease = torch.empty(shape, dtype=source.dtype, device=source.device)
        return lease, lease

    def materialize(self):
        self.materialize_component(0)

    def materialize_component(self, component):
        if not self.leased:
            raise RuntimeError("private active layer lease required")
        from mosaickv.evaluation.native_source_chunks import new_workspace, source_packet

        native = source_packet(self.representation)
        if native is not None and component == 0:
            self.source_workspace = new_workspace(native)
            from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

            self.lease_bytes = backing_bytes((*self.leased, self.source_workspace))
            self.observe(())
        fill_component(
            self.representation,
            self.attention,
            component,
            self.leased[0],
            source_workspace=getattr(self, "source_workspace", None),
        )
        self.leased[0][..., self.prompt_length :, :].copy_(
            self.history[component][..., : self.history_length, :]
        )
        self.current_component = component
        self.component_fills = getattr(self, "component_fills", 0) + 1
        if native is not None and component == 1:
            from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

            self.source_workspace = None
            self.lease_bytes = backing_bytes(self.leased)
            self.observe(())


class SequentialAttentionTransport:
    """Scoped eager operator; rejects foreign caches, gradients, GQA and graphs."""

    def __init__(self, adapter, cache):
        self.adapter, self.cache, self.stack = adapter, cache, ExitStack()

    def __enter__(self):
        import transformers
        from transformers.models.llama import modeling_llama as llama

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("audited Transformers 4.55.2 only")
        mapping = {}
        for block, layer in zip(self.adapter._language_layers(), self.cache.layers, strict=True):
            attention = block.self_attn
            if (
                type(attention).__name__ != "LlamaAttention"
                or type(attention).__module__ != llama.__name__
                or attention.training
                or attention.config._attn_implementation != "eager"
                or attention.num_key_value_groups != 1
                or not isinstance(layer, SequentialLayer)
            ):
                raise NotImplementedError("private Llama full-MHA eval/eager layer required")
            mapping[attention] = layer
        original = llama.eager_attention_forward
        if getattr(original, "_evicastkv_sequential_private", False):
            raise RuntimeError("nested sequential request contexts unsupported")

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if module not in mapping:
                return original(
                    module, query, key, value, attention_mask, scaling, dropout, **kwargs
                )
            layer = mapping[module]
            if (
                torch.is_grad_enabled()
                or query.requires_grad
                or module.training
                or dropout
                or query.shape[0] != 1
                or query.shape[-2] != 1
                or not layer.leased
                or key.data_ptr() != layer.leased[0].data_ptr()
                or value.data_ptr() != key.data_ptr()
                or getattr(layer, "current_component", None) != 0
            ):
                raise RuntimeError("only isolated one-token inference may consume this lease")
            if query.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("sequential diagnostic does not support CUDA graphs")
            # Exact pinned eager source order; native repeat_kv is identity for MHA.
            weights = torch.matmul(query, key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, :, : key.shape[-2]]
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
            layer.materialize_component(1)
            output = torch.matmul(weights, value).transpose(1, 2).contiguous()
            from mosaickv.evaluation.prompt_prefetch import after_attention

            after_attention(layer)
            return output, weights

        forward._evicastkv_sequential_private = True
        self.stack.enter_context(patch.object(llama, "eager_attention_forward", forward))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def paired_run(adapter, prefill, formats, **kwargs):
    """Verification only: run both transports and return the one-buffer result."""
    import gc

    from mosaickv.evaluation import storage_runs

    if kwargs.get("sequential_lease") or kwargs.get("sealed", True):
        raise ValueError("paired control requires explicit raw history and no nested policy")
    ordinary, baseline = storage_runs.run(adapter, prefill, formats, **kwargs)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    actual, details = storage_runs.run(adapter, prefill, formats, sequential_lease=True, **kwargs)
    if not torch.equal(ordinary, actual):
        raise RuntimeError("common sequential transport changed a logit bit")
    for field in ("initial_active_kv_storage_bytes", "final_active_kv_storage_bytes"):
        if baseline["storage"][field] != details["storage"][field]:
            raise RuntimeError("lease scheduling changed persistent storage")
    if (
        baseline["storage"]["peak_active_kv_storage_bytes"]
        < details["storage"]["peak_active_kv_storage_bytes"]
    ):
        raise RuntimeError("one-buffer lease increased decode-owned backing peak")
    details["storage"]["sequential_lease_control"] = dict(
        logits_bitwise_equal=True,
        original_two_lease_storage=baseline["storage"],
        original_two_lease_wall_seconds=baseline["decode_wall_seconds"],
        original_two_lease_cuda_ms=baseline["decode_cuda_ms"],
        persistent_storage_identical=True,
        applies_to_compared_baselines=True,
    )
    return actual, details
