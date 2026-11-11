"""Private, layer-streaming native prefill for the bounded-reader diagnostic.

Every prompt attention operation still receives the complete original K/V.
Only after that operation returns is its cache retired into a source or raw
packet.  This is not a generic HF cache, a model-quality claim, or a six-stage
MosaicKV path.  Complete-request certification belongs to the external owner
ledger; this helper supplies explicit ownership events and retirement checks.
"""

from __future__ import annotations

import hashlib
import weakref
from contextlib import ExitStack
from unittest.mock import patch

import torch
from transformers import DynamicCache
from transformers.cache_utils import CacheLayerMixin

from mosaickv.adapters.huggingface.types import DecodeState, PrefillOutput, QueryVectors
from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class RetiredPrefillLayer(CacheLayerMixin):
    """Metadata only; completed native prompt attention must never run again."""

    is_sliding = False
    is_compileable = False

    def __init__(self, length):
        super().__init__()
        self.length = length

    def update(self, key_states, value_states, cache_kwargs=None):
        del key_states, value_states, cache_kwargs
        raise RuntimeError("retired prefill layer cannot be decoded or updated")

    def get_seq_length(self, cache_position=None):
        del cache_position
        return self.length

    def get_max_cache_shape(self):
        return -1

    def get_mask_sizes(self, cache_position):
        return self.length + cache_position.shape[0], 0


def _weak_tensor_bases(tensors):
    """Track view bases too: losing a view is not proof of owner retirement."""
    found = {}
    for tensor in tensors:
        while tensor is not None:
            found[id(tensor)] = weakref.ref(tensor)
            tensor = tensor._base
    return tuple(found.values())


class StreamingComponentPrefill:
    """One isolated, one-pass prompt invocation with no retained native pairs."""

    def __init__(
        self,
        adapter,
        prepared,
        *,
        mode,
        workspace,
        positions,
        observer=None,
        target_bytes=None,
        mandatory=(0,),
        retirement_tracker=None,
    ):
        import transformers
        from transformers.models.llama import modeling_llama

        self.adapter, self.prepared, self.mode = adapter, prepared, mode
        self.workspace, self.positions, self.observer = workspace, positions, observer
        self.target_bytes, self.mandatory = target_bytes, mandatory
        self.retirement_tracker = retirement_tracker
        self.modules = tuple(layer.self_attn for layer in adapter._language_layers())
        length = prepared.logical_sequence_length
        if (
            transformers.__version__ != "4.55.2"
            or mode not in {"source", "raw"}
            or not self.modules
            or not isinstance(workspace, ComponentWorkspace)
            or not 1 <= length <= workspace.slots
            or any(
                type(module) is not modeling_llama.LlamaAttention
                or module.training
                or module.config._attn_implementation != "eager"
                or module.num_key_value_groups != 1
                or module.head_dim != workspace.head_dim
                or module.k_proj.weight.dtype != torch.bfloat16
                or module.k_proj.weight.device != workspace.component.device
                or module.config.num_attention_heads != workspace.heads
                for module in self.modules
            )
            or positions.ndim != 1
            or positions.dtype != torch.long
            or positions.device != workspace.component.device
            or positions.numel() < length + 128
            or not positions.is_contiguous()
        ):
            raise ValueError(
                "pinned eager BF16 full-MHA Llama and complete shared workspace required"
            )
        if positions[:length].tolist() != list(range(length)):
            raise ValueError("original contiguous prompt logical positions required")
        mask = prepared.model_inputs.get("attention_mask")
        if mask is not None and not bool((mask != 0).all()):
            raise ValueError("single unpadded original prompt required")
        if "past_key_values" in prepared.model_inputs:
            raise ValueError("streaming prefill must start from an empty private cache")
        if mode == "source" and target_bytes is not None:
            raise ValueError("source construction has no raw target-byte argument")
        if mode == "raw" and (type(target_bytes) is not int or target_bytes < 1):
            raise ValueError("raw construction requires an explicit physical payload target")
        if workspace.component.device.type == "cuda" and retirement_tracker is None:
            raise ValueError(
                "CUDA streaming requires allocation-generation free-completed receipts"
            )
        self.cache = DynamicCache()
        self.cache._mosaickv_retired_prefill_only = True
        self.stack = ExitStack()
        self.packets, self.packet_reports, self.retired = [], [], []
        self.normalized, self.phases, self.phase_identity = None, None, None
        self.seen, self.updated, self.finished = set(), set(), set()
        self.retirement_confirmed = set()
        self.retirement_tokens, self.retirement_receipts = {}, {}
        self.events = []
        self.max_live_native_layers = 0
        self.active_layer = None
        self.construction_extras = ()
        self.invoked = False
        self.constructor_audits = []

    def current_owned_tensors(self):
        """Complete strong owners at this observation; returned only transiently."""
        return (
            *self.workspace.owned_tensors(),
            self.positions,
            *(t for packet in self.packets for t in packet.tensors()),
            *(() if self.normalized is None else (self.normalized,)),
            *(() if self.phases is None else self.phases),
            *(
                tensor
                for layer in self.cache.layers
                for tensor in (layer.keys, layer.values)
                if tensor is not None
            ),
            *(
                tensor
                for _index, references in self.retired
                for reference in references
                if (tensor := reference()) is not None
            ),
            *self.construction_extras,
        )

    def _observe(self, phase):
        live_native = sum(
            layer.keys is not None or layer.values is not None for layer in self.cache.layers
        )
        self.max_live_native_layers = max(self.max_live_native_layers, live_native)
        if live_native > 1:
            raise RuntimeError("streaming prefill retained more than one native K/V layer")
        tensors = self.current_owned_tensors()
        self.events.append(
            dict(
                phase=phase, tensor_backing_bytes=backing_bytes(tensors), native_layers=live_native
            )
        )
        if self.observer is not None:
            self.observer(tensors, phase)

    def _verify_retirements(self):
        for index, references in self.retired:
            if index in self.retirement_confirmed:
                continue
            if any(reference() is not None for reference in references):
                raise RuntimeError(f"native prefill owner survived retirement of layer {index}")
            if self.retirement_tracker is not None:
                self.retirement_receipts[index] = self.retirement_tracker.verify(
                    self.retirement_tokens[index], synchronize=True
                )
            self.retirement_confirmed.add(index)
            self._observe(f"prefill_layer_{index}_native_retired")

    def _before(self, module, args, kwargs, *, index):
        self._verify_retirements()
        self._observe(f"prefill_layer_{index}_previous_retired")
        if index in self.seen or index != len(self.seen) or self.active_layer is not None:
            raise RuntimeError(
                "one ordered, non-reentrant native prompt attention per layer required"
            )
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        phase = kwargs.get("position_embeddings")
        length = self.prepared.logical_sequence_length
        if (
            hidden is None
            or hidden.shape != (1, length, self.workspace.heads * self.workspace.head_dim)
            or hidden.dtype != torch.bfloat16
            or phase is None
            or len(phase) != 2
        ):
            raise ValueError("complete original prompt source and explicit RoPE phases required")
        identity = tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in phase)
        if self.phase_identity is None:
            self.phase_identity = identity
            if self.mode == "source":
                self.phases = tuple(t.detach().clone().contiguous() for t in phase)
                self._observe(f"prefill_layer_{index}_shared_phase_capture")
        elif identity != self.phase_identity:
            raise ValueError("one shared original Llama prompt phase required")
        if self.mode == "source":
            self.normalized = hidden.detach().clone().contiguous()
            self._observe(f"prefill_layer_{index}_source_captured")
        self.seen.add(index)
        self.active_layer = index

    def _constructor_observer(self, tensors):
        self.construction_extras = tuple(tensors)
        try:
            self._observe(f"prefill_layer_{self.active_layer}_constructor_workspace")
        finally:
            self.construction_extras = ()

    def _make_packet(self, module, pair, output, *, index):
        if self.mode == "source":
            from mosaickv.evaluation.bounded_component_source import construct

            packet = construct(
                self.normalized,
                pair,
                module,
                self.phases,
                take_source_ownership=True,
                workspace=self.workspace,
                exact_only=True,
                workspace_observer=self._constructor_observer,
            )
            report = packet.diagnostics
            self.normalized = None
        else:
            from mosaickv.evaluation.bounded_ragged_raw import build_raw_packet_layer

            probability = output[1]
            if probability is None or probability.shape[-2:] != (pair[0].shape[-2],) * 2:
                raise ValueError("original eager full-prompt attention probabilities required")
            # Same GPU clone -> FP32 query mean as the frozen all-layer capture.
            # The native probability map is an ordinary model activation. Its
            # method-specific cloned window and scoring buffers ARE charged.
            try:
                window = probability[..., -16:, :].detach().clone()
                self.construction_extras = (window,)
                self._observe(f"prefill_layer_{index}_selection_window")
                float_window = window.float()
                self.construction_extras = (window, float_window)
                self._observe(f"prefill_layer_{index}_selection_float")
                del window
                self.construction_extras = (float_window,)
                self._observe(f"prefill_layer_{index}_selection_window_retired")
                mean = float_window.mean(-2)[0]
                self.construction_extras = (float_window, mean)
                self._observe(f"prefill_layer_{index}_selection_mean")
                scores = mean.cpu()
                del float_window, mean
            finally:
                self.construction_extras = ()
            self._observe(f"prefill_layer_{index}_selection_GPU_buffers_retired")
            packet, report = build_raw_packet_layer(
                pair,
                scores,
                self.mandatory,
                layer_index=index,
                total_layers=len(self.modules),
                target_bytes=self.target_bytes,
                workspace=self.workspace,
                workspace_observer=self._constructor_observer,
            )
            del scores
        return packet, report

    def _after(self, module, args, output, *, index):
        from mosaickv.evaluation.projection_cache import ScratchAudit, identity

        del args
        if index != self.active_layer or index not in self.updated or index in self.finished:
            raise RuntimeError("native prompt update must precede one post-attention conversion")
        self._observe(f"prefill_layer_{index}_native_attention_complete")
        native = self.cache.layers[index]
        pair = native.keys, native.values
        if any(t is None for t in pair):
            raise RuntimeError("complete current native K/V pair required before conversion")
        excluded = {identity(t) for t in self.adapter.model.parameters()}
        excluded.update(identity(t) for t in self.current_owned_tensors())
        # Native eager output/probability owners already belong to the ordinary
        # model invocation. Slicing/detaching them does not create cache storage.
        # Actual cloned scoring windows and their FP32 reductions are new owners
        # and remain visible to ScratchAudit and the explicit selection events.
        excluded.update(identity(t) for t in output if isinstance(t, torch.Tensor))
        peak_extra_bytes, observed_dtypes = 0, set()

        def observe_constructor_results(tensors):
            nonlocal peak_extra_bytes
            tensors = tuple(t for t in tensors if t.device == self.workspace.component.device)
            peak_extra_bytes = max(peak_extra_bytes, backing_bytes(tensors))
            observed_dtypes.update(str(t.dtype) for t in tensors)
            self._constructor_observer(tensors)

        audit = ScratchAudit(excluded=excluded, observe=observe_constructor_results)
        before_events = len(self.events)
        with audit:
            packet, report = self._make_packet(module, pair, output, index=index)
        constructor_observations = sum(
            row["phase"] == f"prefill_layer_{index}_constructor_workspace"
            for row in self.events[before_events:]
        )
        if constructor_observations < audit.operations:
            raise RuntimeError("constructor ATen results lack complete ownership observations")
        self.constructor_audits.append(
            dict(
                layer=index,
                operations=audit.operations,
                observations=constructor_observations,
                peak_extra_tensor_backing_bytes=peak_extra_bytes,
                observed_same_device_dtypes=sorted(observed_dtypes),
                weak_results_only=True,
                includes_validation_booleans_indices_views_and_constructor_outputs=True,
                excludes_existing_owner_aliases_and_model_weights=True,
                excludes_ordinary_native_attention_output_aliases=True,
                opaque_vendor_internal_workspace_not_visible_to_ATen=True,
            )
        )
        report["constructor_audit_operations"] = audit.operations
        report["constructor_audit_observations"] = constructor_observations
        self.packets.append(packet)
        self.packet_reports.append(report)
        self._observe(f"prefill_layer_{index}_packet_constructed")
        self.retired.append((index, _weak_tensor_bases(pair)))
        if self.retirement_tracker is not None:
            self.retirement_tokens[index] = self.retirement_tracker.capture(
                pair, label=f"layer_{index}"
            )
        native.keys = native.values = None
        self.cache.layers[index] = RetiredPrefillLayer(self.prepared.logical_sequence_length)
        self.finished.add(index)
        self.active_layer = None
        self._observe(f"prefill_layer_{index}_retirement_pending_frame_return")

    def __enter__(self):
        original_update = self.cache.update

        def update(keys, values, layer_idx, cache_kwargs=None):
            if layer_idx != self.active_layer or layer_idx in self.updated:
                raise RuntimeError("one current-layer native K/V update per prompt required")
            result = original_update(keys, values, layer_idx, cache_kwargs)
            self.updated.add(layer_idx)
            self._observe(f"prefill_layer_{layer_idx}_KV_published")
            return result

        try:
            self.stack.enter_context(patch.object(self.cache, "update", update))
            for index, module in enumerate(self.modules):

                def before(module, args, kwargs, *, index=index):
                    self._before(module, args, kwargs, index=index)

                def after(module, args, output, *, index=index):
                    self._after(module, args, output, index=index)

                hook = module.register_forward_pre_hook(before, with_kwargs=True)
                self.stack.callback(hook.remove)
                hook = module.register_forward_hook(after)
                self.stack.callback(hook.remove)
            self._observe("request_workspace_ready_before_prefill")
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    @torch.inference_mode()
    def prefill(self):
        if self.invoked:
            raise RuntimeError("streaming prefill may execute only once")
        self.invoked = True
        inputs = dict(self.prepared.model_inputs)
        inputs.update(
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            cache_position=self.positions[: self.prepared.logical_sequence_length],
            logits_to_keep=1,
            past_key_values=self.cache,
        )
        output = self.adapter._prefill_forward(inputs)
        self._verify_retirements()
        layers = len(self.modules)
        if (
            len(self.packets) != layers
            or self.seen != set(range(layers))
            or self.updated != self.seen
            or self.finished != self.seen
            or self.retirement_confirmed != self.seen
            or self.normalized is not None
            or self.active_layer is not None
            or any(
                layer.keys is not None or layer.values is not None for layer in self.cache.layers
            )
        ):
            raise RuntimeError("incomplete streaming prompt conversion or native owner retirement")
        self._observe("prefill_complete")
        logits = output.logits[:, -1, :]
        mask = self.prepared.model_inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(self.prepared.model_inputs["input_ids"])
        metadata = self.adapter._capture_model_state(output)
        metadata.update(
            mosaickv_cache_all_valid=True,
            mosaickv_streaming_prefill_only=True,
            mosaickv_prefill_attention_collection={
                "private_module_hooks": self.mode == "raw",
                "model_output_collector": False,
                "audited_hook_only_llama_4_55_2": True,
            },
        )
        logical = self.prepared.logical_sequence_length
        state = DecodeState(
            self.cache, mask, logical, logical, logical, self.prepared.modality_map, metadata
        )
        prefill = PrefillOutput(logits, self.adapter._greedy_token(logits), state, QueryVectors(()))
        report = dict(
            mode=self.mode,
            layers=layers,
            native_update_layers=sorted(self.updated),
            constructed_layers=sorted(self.finished),
            retired_native_layers=sorted(self.retirement_confirmed),
            maximum_live_native_layers=self.max_live_native_layers,
            all_native_KV_views_and_bases_released=True,
            ordinary_full_prompt_attention_unchanged=True,
            copied_native_cache_snapshot=False,
            source_workspace_reused=True,
            prompt_packet_bytes=backing_bytes(tuple(t for p in self.packets for t in p.tensors())),
            packets=self.packet_reports,
            constructor_temporaries_audited=(
                len(self.constructor_audits) == layers
                and all(row["operations"] > 0 for row in self.constructor_audits)
            ),
            constructor_audits=self.constructor_audits,
            constructor_audit_layers=[row["layer"] for row in self.constructor_audits],
            ownership_events=self.events,
            excluded_noncache_activations="ordinary_native_projection_RoPE_attention_score_probability_output_activations_only;selection_copies_and_score_buffers_included",
            raw_score_gpu_window_retained=False,
            CPU_position_metadata_bytes=(
                backing_bytes(
                    tuple(p for packet in self.packets for p in packet.original_positions)
                )
                if self.mode == "raw"
                else 0
            ),
            complete_request_peak_certified=False,
            six_mechanism_performance_claim=False,
            native_layers_retired=layers,
            maximum_live_native_cache_layers=self.max_live_native_layers,
            outstanding_native_owner_references=0,
            all_native_owner_weakrefs_released=True,
            native_CUDA_free_completed_verified=(
                self.workspace.component.device.type == "cuda"
                and len(self.retirement_receipts) == layers
            ),
            native_retirement_receipts=[
                self.retirement_receipts[index] for index in sorted(self.retirement_receipts)
            ],
            uses_original_fullkv_snapshot=False,
            original_logical_positions_preserved=True,
            constructor_workspace_shared_with_decode=True,
            benchmark_labels_read=False,
            fallback_used=False,
        )
        if self.mode == "raw":
            maps = tuple(p for packet in self.packets for p in packet.original_positions)
            report["raw_position_map_sha256"] = hashlib.sha256(
                b"".join(p.numpy().tobytes() for p in maps)
            ).hexdigest()
            report["CPU_position_metadata_bytes"] = backing_bytes(maps)
        return prefill, tuple(self.packets), report


def run(adapter, prepared, **kwargs):
    """One inference-only private prefill; all hooks removed before returning."""
    with torch.inference_mode(), StreamingComponentPrefill(adapter, prepared, **kwargs) as stream:
        return stream.prefill()


__all__ = ["RetiredPrefillLayer", "StreamingComponentPrefill", "run"]
