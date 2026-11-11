"""Four-prefix input-gradient evidence; no weight training or benchmark labels.

One FullKV draft reuses the caller's existing prefill. A separate native-wide
three-bit cache supplies ordinary BF16 K/V leaves to the SAME adapter forward
outside inference mode. One backward pass ranks every original source row by
its first-order effect on short-horizon KL and decision margins. No candidate
cache is probed per ranked row. This is a prospective approximation, not proof
of later-prefix/answer fidelity or the complete six-mechanism design.
"""

from __future__ import annotations

import hashlib
import time
import weakref
from threading import RLock

import torch

from mosaickv.adapters.huggingface.llava import Llava15Adapter
from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import owner
from mosaickv.evaluation.native_quantized_source_control import (
    _tensor_signature,
    _validate,
    native_quantized_fill,
)

HORIZON = 4
KL_SCALE = 0.02
MARGIN_SCALE = 1.0
KIND = "four_token_native_cache_input_gradient_source_priority_diagnostic"


def _structure(value):
    if isinstance(value, torch.Tensor):
        return id(value), _tensor_signature(value), value.requires_grad
    if isinstance(value, dict):
        return tuple((str(k), _structure(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_structure(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__, id(value)


def _prefill_signature(adapter, prefill):
    state = prefill.state
    pairs, _ = adapter._legacy_layers(state.past_key_values)
    return (
        id(state),
        id(state.past_key_values),
        tuple(tuple(_structure(t) for t in pair) for pair in pairs),
        _structure(state.attention_mask),
        state.active_cache_length,
        state.logical_sequence_length,
        state.next_decode_position,
        _structure(state.modality_map),
        _structure(state.model_state),
        _structure(prefill.logits),
        _structure(prefill.next_token_id),
    )


def _model_signature(model):
    return (
        tuple(
            (
                name,
                id(p),
                _tensor_signature(p),
                p.requires_grad,
                None if p.grad is None else _structure(p.grad),
            )
            for name, p in model.named_parameters()
        ),
        tuple((name, _structure(b)) for name, b in model.named_buffers()),
        tuple((name, module.training) for name, module in model.named_modules()),
    )


def _state(cache, n, device, modalities):
    return DecodeState(
        cache,
        torch.ones((1, n), dtype=torch.long, device=device),
        n,
        n,
        n,
        modalities,
        dict(mosaickv_cache_all_valid=True, mosaickv_packed_all_valid=True),
    )


def _cache(pairs):
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer, pair in enumerate(pairs):
        cache.update(*(t.detach().clone().contiguous() for t in pair), layer)
    return cache


def _digest(logits):
    value = logits.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _loss(reference, candidate):
    ref_logp, logp = reference.log_softmax(-1), candidate.log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)
    token = reference.argmax(-1)
    mask = torch.nn.functional.one_hot(token, num_classes=reference.shape[-1]).bool()
    ref_margin = reference.gather(1, token[:, None]).squeeze(1) - reference.masked_fill(
        mask, float("-inf")
    ).amax(-1)
    candidate_margin = candidate.gather(1, token[:, None]).squeeze(1) - candidate.masked_fill(
        mask, float("-inf")
    ).amax(-1)
    shortfall = torch.relu(ref_margin - candidate_margin)
    objective = kl.sum() / KL_SCALE + shortfall.sum() / MARGIN_SCALE
    return objective, kl, ref_margin, candidate_margin, shortfall


class _SavedTensors:
    """Incremental weak census, with exactly the existing backing-byte boundary.

    This is not an allocator or whole-request KV peak certificate. Detached
    saved views have identical values and owners; the observer never retains
    graph-bearing activations or extends their lifetime itself. Storage must
    not be resized while saved, as required by this frozen-model diagnostic.
    A changed owner/size is rejected at unpack instead of reported as a pass.

    Each pack inspects its owner once. Weak callbacks decrement per-allocation
    reference counts; neither pack nor unpack scans other saved tensors. The
    metric key matches backing_bytes: (device, storage pointer, storage bytes).
    Callback closures hold only a weak observer reference and scalar metadata.
    """

    def __init__(self, model):
        self.excluded = {owner(p) for p in model.parameters()}
        self.refs, self.pack_calls, self.peak = {}, 0, 0
        self.owner_counts, self.live_bytes = {}, 0
        self.closed = False
        self._weak_self, self._lock = weakref.ref(self), RLock()

    def _released(self, token, reference, key):
        with self._lock:
            entry = self.refs.get(token)
            # A late callback must never retire a new object with a reused ID.
            if entry is None or entry[0] is not reference:
                return
            del self.refs[token]
            count = self.owner_counts[key] - 1
            if count:
                self.owner_counts[key] = count
            else:
                del self.owner_counts[key]
                self.live_bytes -= key[-1]

    def pack(self, tensor):
        value = tensor.detach()
        with self._lock:
            if self.closed:
                raise RuntimeError("Saved-tensor census is already closed")
            self.pack_calls += 1
            if owner(value) in self.excluded:
                return value
            storage = value.untyped_storage()
            key = str(value.device), storage.data_ptr(), storage.nbytes()
            token = id(value)

            def released(reference, tracker=self._weak_self, token=token, key=key):
                current = tracker()
                if current is not None:
                    current._released(token, reference, key)

            self.refs[token] = weakref.ref(value, released), key
            count = self.owner_counts.get(key, 0)
            self.owner_counts[key] = count + 1
            if not count:
                self.live_bytes += key[-1]
            self.peak = max(self.peak, self.live_bytes)
        return value

    def unpack(self, tensor):
        with self._lock:
            if self.closed:
                raise RuntimeError("Saved-tensor census is already closed")
            entry = self.refs.get(id(tensor))
            if entry is not None:
                storage = tensor.untyped_storage()
                key = str(tensor.device), storage.data_ptr(), storage.nbytes()
                if key != entry[1]:
                    raise RuntimeError("Saved tensor storage changed while being observed")
        return tensor

    def close(self):
        with self._lock:
            self.closed = True
            self.refs.clear()
            self.owner_counts.clear()
            self.live_bytes = 0


def _validate_inputs(adapter, prefill, packets, originals, horizon):
    if type(horizon) is not int or horizon != HORIZON:
        raise ValueError("The frozen draft policy requires exactly four cache-dependent prefixes")
    if type(adapter) is not Llava15Adapter or adapter._attention_implementation() != "eager":
        raise NotImplementedError("Only the audited eager Llava15Adapter is supported")
    for module in adapter.model.modules():
        if (
            module.training
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
        ):
            raise ValueError("Eval-only model without active forward/backward hooks required")
        if hasattr(module, "_hf_hook"):
            raise NotImplementedError(
                "Offload/dispatch hooks are not an audited autograd transport"
            )
    if any(torch.is_inference(p) for p in adapter.model.parameters()):
        raise NotImplementedError(
            "Inference model weights cannot be saved for cache-input backward"
        )
    blocks = tuple(adapter._language_layers())
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if not packets or not (len(packets) == len(originals) == len(blocks) == len(source)):
        raise ValueError("Every original model layer and native source packet is required")
    shape = tuple(originals[0][0].shape)
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128 or min(shape) < 1:
        raise ValueError("Batch-one full-MHA BF16 original prompt geometry required")
    n = shape[-2]
    state = prefill.state
    if (
        state.active_cache_length != n
        or state.logical_sequence_length != n
        or state.next_decode_position != n
        or state.attention_mask.shape != (1, n)
        or not (state.attention_mask == 1).all().item()
        or prefill.next_token_id.shape != (1, 1)
    ):
        raise ValueError(
            "An untouched all-valid original prefill with identical physical/logical n required"
        )
    for packet, pair, actual, block in zip(packets, originals, source, blocks, strict=True):
        _validate(packet, block.self_attn)
        if (
            tuple(packet.shape) != shape
            or len(pair) != 2
            or any(
                t.shape != shape or t.dtype != torch.bfloat16 or t.device != adapter.device
                for t in pair
            )
        ):
            raise ValueError("Common original BF16 native source geometry required")
        if any(
            owner(a) != owner(b) or tuple(a.stride()) != tuple(b.stride())
            for a, b in zip(pair, actual, strict=True)
        ):
            raise ValueError(
                "Original K/V must be the caller's unchanged prefill, not another cache"
            )
    return shape


def draft_row_scores(adapter, prefill, parent_packets, original_pairs, *, horizon=HORIZON):
    """Return signed CPU row gains and JSON-only provenance/cost observations.

    Gain = -grad(cache) dot (original prompt K/V - native-quantized prompt K/V).
    The four FullKV draft tokens come solely from the current original prefill.
    No saved long-horizon oracle, new prefill, benchmark label or candidate
    forward pass enters this ranking. A positive first-order gain is NOT a
    guarantee of a finite restoration's real model-level benefit.
    """
    packets, originals = tuple(parent_packets), tuple(tuple(pair) for pair in original_pairs)
    # Public adapters and constructors deliberately use inference mode. All
    # leaves, token indices and reference constants here must be NORMAL tensors.
    with torch.inference_mode(False), torch.enable_grad():
        shape = _validate_inputs(adapter, prefill, packets, originals, horizon)
        n, device = shape[-2], originals[0][0].device
        before_prefill = _prefill_signature(adapter, prefill)
        before_model = _model_signature(adapter.model)
        reference_state = source_state = None
        leaves, values, inputs = [], [], []
        gradients = ()
        reference = logits = objective = None
        audit = _SavedTensors(adapter.model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocator_before = int(torch.cuda.memory_allocated(device))
            cumulative_peak_before = int(torch.cuda.max_memory_allocated(device))
        else:
            allocator_before = cumulative_peak_before = None
        started = time.perf_counter()
        phases = {}
        try:
            begin = time.perf_counter()
            with torch.no_grad():
                reference_state = _state(_cache(originals), n, device, prefill.state.modality_map)
                token = prefill.next_token_id.detach().clone()
                ref_values = []
                for _ in range(horizon):
                    inputs.append(token.detach().clone())
                    out = adapter.decode_one_token(token, reference_state, capture_queries=False)
                    ref_values.append(out.logits.detach().float().clone())
                    token = out.next_token_id.detach().clone()
                reference = torch.cat(ref_values, 0).detach().clone()
                del ref_values, token, out
                reference_state.past_key_values = None
                reference_state = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phases["four_token_FullKV_draft_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            from transformers import DynamicCache

            cache = DynamicCache()
            for layer, (packet, block) in enumerate(
                zip(packets, adapter._language_layers(), strict=True)
            ):
                targets = tuple(
                    torch.empty(shape, device=device, dtype=torch.bfloat16) for _ in range(2)
                )
                native_quantized_fill(packet, block.self_attn, targets, direct=True)
                pair = tuple(t.detach().clone().requires_grad_(True) for t in targets)
                leaves.extend(pair)
                cache.update(*pair, layer)
                del targets, pair
            source_state = _state(cache, n, device, prefill.state.modality_map)
            del cache
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phases["native_parent_materialization_and_leaf_clone_seconds"] = (
                time.perf_counter() - begin
            )
            begin = time.perf_counter()
            with torch.autograd.graph.saved_tensors_hooks(audit.pack, audit.unpack):
                for step, token in enumerate(inputs):
                    position = torch.tensor([n + step], device=device, dtype=torch.long)
                    out = adapter._decode_forward(
                        token, source_state, None, position, output_attentions=False
                    )
                    values.append(out.logits[:, -1, :].float())
                    source_state.past_key_values = out.past_key_values
                    source_state.active_cache_length += 1
                    source_state.logical_sequence_length += 1
                    source_state.next_decode_position += 1
                logits = torch.cat(values, 0)
                objective, kl, ref_margin, source_margin, margin_shortfall = _loss(
                    reference, logits
                )
                if not torch.isfinite(objective).item():
                    raise RuntimeError("Nonfinite short-horizon native cache objective")
                gradients = torch.autograd.grad(objective, leaves, allow_unused=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phases["four_parent_forwards_and_input_backward_seconds"] = time.perf_counter() - begin
            if not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Nonfinite cache-input gradients")
            begin = time.perf_counter()
            scores = {}
            with torch.no_grad():
                for layer, original in enumerate(originals):
                    gain = torch.zeros(n, device=device, dtype=torch.float64)
                    for component in (0, 1):
                        leaf, gradient = (
                            leaves[2 * layer + component],
                            gradients[2 * layer + component],
                        )
                        delta = original[component].double() - leaf.detach().double()
                        gain -= (gradient.double() * delta).sum((0, 1, 3))
                    scores[layer] = gain.detach().float().cpu().contiguous()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phases["signed_row_reduction_seconds"] = time.perf_counter() - begin
            report = dict(
                kind=KIND,
                horizon=horizon,
                KL_normalization=KL_SCALE,
                decision_margin_normalization=MARGIN_SCALE,
                objective="sum_t KL(FullKV||parent)/0.02 + relu(FullKV_margin-parent_margin)/1.0",
                score="negative_sum_heads_and_width(cache_gradient*original_minus_parent_KV)",
                reference_logits_sha256=_digest(reference),
                parent_logits_sha256=_digest(logits),
                reference_logits_fp32=reference.detach().float().cpu().tolist(),
                parent_logits_fp32=logits.detach().float().cpu().tolist(),
                reference_logits_shape=list(reference.shape),
                parent_logits_shape=list(logits.shape),
                draft_input_token_ids=[int(token.item()) for token in inputs],
                draft_predicted_token_ids=reference.argmax(-1).detach().cpu().tolist(),
                cache_dependent_query_prefixes=horizon,
                query_vectors_captured=False,
                reference_KL_per_prefix=kl.detach().cpu().tolist(),
                reference_greedy_margins=ref_margin.detach().cpu().tolist(),
                parent_reference_token_margins=source_margin.detach().cpu().tolist(),
                margin_shortfalls=margin_shortfall.detach().cpu().tolist(),
                objective_value=float(objective.detach()),
                positive_row_gains=sum(int((s > 0).sum()) for s in scores.values()),
                negative_row_gains=sum(int((s < 0).sum()) for s in scores.values()),
                zero_row_gains=sum(int((s == 0).sum()) for s in scores.values()),
                nonzero_gradient_scalars=sum(int((g != 0).sum()) for g in gradients),
                BF16_cache_leaf_gradients=True,
                prompt_leaf_KV_backing_bytes=backing_bytes(tuple(leaves)),
                autograd_saved_nonweight_peak_backing_bytes=audit.peak,
                autograd_saved_tensor_observations=audit.pack_calls,
                cumulative_GPU_allocator_peak_before_bytes=cumulative_peak_before,
                cumulative_GPU_allocator_peak_after_bytes=(
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                ),
                GPU_allocator_allocated_before_bytes=allocator_before,
                phase_wall_seconds=phases,
                total_wall_seconds=time.perf_counter() - started,
                torch_version=torch.__version__,
                transformers_version=__import__("transformers").__version__,
                native_adapter_class=type(adapter).__name__,
                attention="eager",
                source_shape=list(shape),
                no_second_prefill=True,
                original_cache_unchanged=True,
                model_weights_unchanged=True,
                model_parameter_gradients_unchanged=True,
                requires_grad_flags_unchanged=True,
                private_cache_only=True,
                returns_cpu_scores_only=True,
                parameter_gradients_requested=False,
                candidates_model_probed=0,
                no_saved_long_horizon_oracle_read=True,
                benchmark_answers_accessed=False,
                signed_first_order_prediction_not_finite_change_guarantee=True,
                whole_request_KV_peak_certified=False,
                opaque_BLAS_workspace_certified=False,
                latency_benchmark=False,
                larger_campaign_authorized=False,
            )
            return scores, report
        finally:
            if reference_state is not None:
                reference_state.past_key_values = None
            if source_state is not None:
                source_state.past_key_values = None
            leaves.clear()
            values.clear()
            inputs.clear()
            gradients, reference, logits, objective = (), None, None, None
            out = token = position = None
            # Clear partial construction owners even in an exception traceback.
            cache = targets = pair = None  # noqa: F841
            kl = ref_margin = source_margin = margin_shortfall = None
            leaf = gradient = delta = gain = None
            audit.close()
            if _prefill_signature(adapter, prefill) != before_prefill:
                raise RuntimeError(
                    "Short-horizon scoring mutated the caller's original prefill/cache"
                )
            if _model_signature(adapter.model) != before_model:
                raise RuntimeError(
                    "Short-horizon scoring mutated model weights, gradients, flags or buffers"
                )
