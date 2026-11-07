"""Bounded last-layer counterfactual: actual frozen suffix, NOT a JVP.

Diagnostic only until model-level identity/candidate controls pass bitwise.
All snapshots are CPU-owned; no model/cache owner is retained by capture.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch

_capture = ContextVar("evicastkv_last_layer_readout_capture", default=None)


def _snapshot(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def record_component(layer, component, *, parent):
    capture = _capture.get()
    if capture is not None and layer.attention.layer_idx == capture.layer:
        value = layer.leased[0][..., : layer.get_seq_length(), :]
        key = ("parent_" if parent else "active_") + ("key" if component == 0 else "value")
        if key not in capture.data:
            tensor = value[:, capture.head : capture.head + 1] if parent else value
            capture.data[key] = _snapshot(tensor)
            capture.data[key + "_stride"] = tuple(tensor.stride())


class ReadoutCapture:
    def __init__(self, head, layer=31):
        if type(head) is not int or not 0 <= head < 32 or layer != 31:
            raise ValueError("pinned last-layer, full-MHA routed head required")
        self.head, self.layer, self.data = head, layer, {}

    @contextmanager
    def use(self):
        from mosaickv.evaluation.post_attention_segment import Transport
        from mosaickv.evaluation.uniform_core_transport import Binding

        score, values, epilogue = Binding.scores, Binding.values, Transport.run
        owner = self

        def observe_score(binding, query, key, mask, scaling):
            if binding.calls % 32 == owner.layer:
                if "query" in owner.data:
                    raise RuntimeError("bounded readout capture permits one decode update")
                owner.data["query"] = _snapshot(query)
                owner.data["query_stride"] = tuple(query.stride())
                owner.data["mask"] = None if mask is None else _snapshot(mask)
                owner.data["scaling"] = float(scaling)
            return score(binding, query, key, mask, scaling)

        def observe_values(binding, length):
            result = values(binding, length)
            if (binding.calls - 1) % 32 == owner.layer:
                owner.data["head_outputs"] = _snapshot(result).reshape(1, 1, -1)
            return result

        def observe_epilogue(transport, layer, index, residual, attention):
            if index == owner.layer:
                # Input reuses the preceding epilogue's output buffer: copy
                # BEFORE replay overwrites it, never retain a detached view.
                owner.data["layer_residual"] = _snapshot(residual)
                owner.data["projected_attention"] = _snapshot(attention)
            return epilogue(transport, layer, index, residual, attention)

        token = _capture.set(self)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(Binding, "scores", observe_score))
                stack.enter_context(patch.object(Binding, "values", observe_values))
                stack.enter_context(patch.object(Transport, "run", observe_epilogue))
                yield self
        finally:
            _capture.reset(token)

    def validate(self, prompt_length):
        required = {
            "query",
            "active_key",
            "active_value",
            "parent_key",
            "parent_value",
            "head_outputs",
            "layer_residual",
            "projected_attention",
            "mask",
            "scaling",
        }
        if not required <= self.data.keys():
            raise RuntimeError(
                f"missing real readout observations: {sorted(required - self.data.keys())}"
            )
        if self.data["active_key"].shape != (1, 32, prompt_length + 1, 128):
            raise RuntimeError("actual one-update source geometry required")
        tensors = tuple(v for v in self.data.values() if isinstance(v, torch.Tensor))
        if any(t.device.type != "cpu" for t in tensors):
            raise RuntimeError("readout capture must not keep GPU source/cache owners")
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        return dict(
            temporary_cpu_capture_bytes=backing_bytes(tensors),
            gpu_capture_owners=0,
            actual_post_rope_query=True,
            actual_parent_and_active_kv=True,
        )


def suffix(layer, final_norm, output_head, residual, head_outputs):
    projected = layer.self_attn.o_proj(head_outputs)
    hidden = residual + projected
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return output_head(final_norm(hidden)).float().reshape(-1)


def attention_output(query, key, value, mask, scaling):
    score = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        score = score + mask
    probabilities = score.softmax(-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous().reshape(1, 1, -1)


def _cuda_copy(value, stride=None):
    if stride is None:
        return value.to("cuda")
    result = torch.empty_strided(value.shape, stride, dtype=value.dtype, device="cuda")
    result.copy_(value)
    return result


@torch.inference_mode()
def verify_controls(adapter, capture, overlay, original_prompt, reference, probe):
    """Verify predicted versus ACTUAL isolated model probes, then restore buffers.

    No ranking, no quality tuning, no repair-event fabrication. Restoring
    members here is an explicitly labeled causal control.
    """
    if tuple(adapter._language_layers())[-1].self_attn.layer_idx != 31:
        raise NotImplementedError("only the actual final LLaVA layer is supported")
    layer = adapter._language_layers()[-1]
    try:
        norm = adapter.model.model.language_model.norm
    except AttributeError as error:
        raise NotImplementedError("pinned LLaVA final norm path required") from error
    head = adapter.model.get_output_embeddings()
    if adapter.model.training or any(m.training for m in (layer, norm, head)):
        raise ValueError("frozen eval modules required")
    versions = tuple(p._version for p in adapter.model.parameters())
    n = original_prompt[0].shape[-2]
    evidence = capture.validate(n)
    data = capture.data
    from time import perf_counter

    torch.cuda.synchronize()
    started = perf_counter()
    q = _cuda_copy(data["query"], data["query_stride"])
    key = _cuda_copy(data["active_key"], data["active_key_stride"])
    value = _cuda_copy(data["active_value"], data["active_value_stride"])
    residual, actual_heads = _cuda_copy(data["layer_residual"]), _cuda_copy(data["head_outputs"])
    mask = None if data["mask"] is None else _cuda_copy(data["mask"])
    original = tuple(t.to("cuda") for t in original_prompt)
    expected = reference.to("cuda").float().reshape(-1)
    recomputed_heads = attention_output(q, key, value, mask, data["scaling"])
    baseline = suffix(layer, norm, head, residual, actual_heads)
    projected = layer.self_attn.o_proj(actual_heads)
    projected_bitwise = torch.equal(projected.cpu(), data["projected_attention"])
    score_bitwise = torch.equal(recomputed_heads, actual_heads)
    suffix_bitwise = torch.equal(baseline, expected)
    data["recomputed_head_outputs"] = _snapshot(recomputed_heads)
    data["recomputed_baseline_logits"] = _snapshot(baseline)
    evidence.update(
        attention_output_bitwise=score_bitwise,
        attention_output_max_difference=float((recomputed_heads - actual_heads).abs().max()),
        output_projection_bitwise=projected_bitwise,
        baseline_suffix_bitwise=suffix_bitwise,
        baseline_logit_max_difference=float((baseline - expected).abs().max()),
    )
    if not (score_bitwise and suffix_bitwise and projected_bitwise):
        torch.cuda.synchronize()
        evidence.update(
            passed=False,
            reason="readout_numerical_control_failed",
            candidate_controls=[],
            no_threshold_relaxation=True,
            diagnostic_wall_seconds=perf_counter() - started,
        )
        return evidence
    old_ids = overlay.exact_ids.cpu().clone()
    old_pair = tuple(t.cpu().clone() for t in overlay.exact_pair)
    old_promotion = tuple(t.cpu().clone() for t in overlay.promotion)
    old_flag = bool(overlay.promoted)
    protected = set(overlay.evidence["original_protected_positions"])
    members = overlay.positions.cpu().tolist()
    receiver = min(set(range(n)) - protected - set(members) - set(old_ids.tolist()))
    donor = int(old_ids[0])
    controls = []
    try:
        for target, full in zip((key, value), original, strict=True):
            target[0, overlay.head, :n] = full
        conditional = suffix(
            layer, norm, head, residual, attention_output(q, key, value, mask, data["scaling"])
        )
        data["conditional_reference_logits"] = _snapshot(conditional)
        for name in ("identity", "restore_members", "equal_capacity_substitution"):
            # Immutable captured data, not an aliased previous candidate.
            key.copy_(data["active_key"])
            value.copy_(data["active_value"])
            overlay.exact_ids.copy_(old_ids)
            for target, old in zip(overlay.exact_pair, old_pair, strict=True):
                target.copy_(old)
            overlay.promoted.fill_(old_flag)
            if name == "restore_members":
                for target, source in zip(overlay.promotion, overlay.residual_pair, strict=True):
                    target.copy_(source.reshape_as(target))
                overlay.promoted.fill_(True)
                for target, full in zip((key, value), original, strict=True):
                    target[0, overlay.head, members] = full[members]
            elif name == "equal_capacity_substitution":
                revised = sorted((set(old_ids.tolist()) - {donor}) | {receiver})
                overlay.exact_ids.copy_(torch.tensor(revised))
                for target, full in zip(overlay.exact_pair, original, strict=True):
                    target.copy_(full[revised])
                for target, full, parent in zip(
                    (key, value), original, (data["parent_key"], data["parent_value"]), strict=True
                ):
                    target[0, overlay.head, donor] = parent[0, 0, donor].to("cuda")
                    target[0, overlay.head, receiver] = full[receiver]
            predicted = suffix(
                layer, norm, head, residual, attention_output(q, key, value, mask, data["scaling"])
            )
            actual = probe()[1].float().reshape(-1)
            data[f"predicted_logits_{name}"] = _snapshot(predicted)
            data[f"actual_logits_{name}"] = _snapshot(actual)
            target_log = conditional.double().log_softmax(-1)
            divergence = (
                target_log.exp() * (target_log - predicted.double().log_softmax(-1))
            ).sum()
            controls.append(
                dict(
                    control=name,
                    logits_bitwise=torch.equal(predicted, actual),
                    maximum_logit_difference=float((predicted - actual).abs().max()),
                    token_agreement=int(predicted.argmax()) == int(actual.argmax()),
                    fixed_original_positions=True,
                    exact_capacity_unchanged=True,
                    benchmark_labels_used=False,
                    kl_to_routed_head_full_precision_target=float(divergence),
                )
            )
    finally:
        overlay.exact_ids.copy_(old_ids)
        for target, old in zip(overlay.exact_pair, old_pair, strict=True):
            target.copy_(old)
        for target, old in zip(overlay.promotion, old_promotion, strict=True):
            target.copy_(old)
        overlay.promoted.fill_(old_flag)
    restored = (
        torch.equal(overlay.exact_ids.cpu(), old_ids)
        and all(
            torch.equal(t.cpu(), old) for t, old in zip(overlay.exact_pair, old_pair, strict=True)
        )
        and all(
            torch.equal(t.cpu(), old)
            for t, old in zip(overlay.promotion, old_promotion, strict=True)
        )
        and bool(overlay.promoted) == old_flag
    )
    weights_unchanged = versions == tuple(p._version for p in adapter.model.parameters())
    torch.cuda.synchronize()
    evidence.update(
        passed=all(c["logits_bitwise"] for c in controls) and restored and weights_unchanged,
        candidate_controls=controls,
        original_overlay_restored=restored,
        model_weights_unchanged=weights_unchanged,
        persistent_gpu_bytes_unchanged=True,
        controlled_restoration_is_not_natural_repair=True,
        superiority_established=False,
        diagnostic_wall_seconds=perf_counter() - started,
        whole_request_peak_certified=False,
    )
    return evidence
