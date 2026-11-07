"""Last-layer, live-parent decision guard; never an oracle or FullKV fallback.

The parent is the existing lossy storage variant, not FullKV. Only its final
head's prompt reads are changed by the overlay. With identical emitted tokens,
all preceding layers and generated K/V remain identical by induction. This
must be established by actual pretrained counterfactual parity, not assumed.
"""

from contextlib import ExitStack
from contextvars import ContextVar
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_current = ContextVar("evicastkv_live_anchor_guard", default=None)


def record_parent_component(layer, component):
    guard = _current.get()
    if guard is not None and layer.attention.layer_idx == 31:
        guard.record_component(layer, component)


def decision(parent, candidate):
    """Unchanged .02 KL / 1.0 logit / greedy-token thresholds, per real step."""
    if parent.shape != candidate.shape or parent.ndim != 1:
        raise ValueError("aligned actual vocabulary vectors required")
    p, c = parent.double(), candidate.double()
    finite = p.isfinite().all() & c.isfinite().all()
    if not bool(finite):
        raise ValueError("nonfinite logits are an execution failure, not a successful veto")
    return decision_tensors(p, c, finite)


def decision_tensors(p, c, finite):
    """The original FP64 operations, without a host synchronization in capture."""
    kl = (p.softmax(-1) * (p.log_softmax(-1) - c.log_softmax(-1))).sum()
    maximum = (p - c).abs().max()
    same = p.argmax() == c.argmax()
    passed = finite & same & (kl <= 0.02) & (maximum <= 1.0)
    return passed, torch.stack((kl, maximum, same.double(), passed.double()))


class LiveAnchorGuard:
    def __init__(self, adapter, overlay, report, *, policy=None, inline_owner=None):
        self.adapter, self.overlay, self.report = adapter, overlay, report
        self.policy, self.inline_owner = policy, inline_owner
        self.repair_enabled = bool(policy and policy.anchor_risk_repair)
        self.single_head = bool(policy and policy.anchor_counterfactual == "single_head")
        self.readout = None
        self.tail = None
        self.suffix_replays = self.suffix_verifications = self.suffix_activation_bytes = 0
        self.stack, self.records, self.parent_logits = ExitStack(), [], []
        self.parent_key = self.parent_value = self.probabilities = self.heads = None
        self.layer = self.query = self.key = self.value = None
        self.peak_temporary_kv_bytes = 0
        self.active_key = self.active_value = self.active_heads = None
        self.repair_events = []
        self.coexisting_temporary_kv_peak_bytes = 0

    def charge(self, *extra):
        tensors = tuple(
            t
            for t in (
                self.parent_key,
                self.parent_value,
                self.active_key,
                self.active_value,
                *extra,
            )
            if t is not None
        )
        self.peak_temporary_kv_bytes = max(self.peak_temporary_kv_bytes, backing_bytes(tensors))
        coexisting = getattr(self.inline_owner, "key", None)
        combined = (*tensors, *((coexisting,) if coexisting is not None else ()))
        self.coexisting_temporary_kv_peak_bytes = max(
            self.coexisting_temporary_kv_peak_bytes, backing_bytes(combined)
        )
        self.layer.observe(combined)

    def record_component(self, layer, component):
        if component not in (0, 1) or layer.attention.layer_idx != 31:
            raise ValueError("registered last-layer K/V read required")
        self.layer = layer
        view = layer.leased[0][..., : layer.get_seq_length(), :]
        captured = view[:, self.overlay.head : self.overlay.head + 1].clone()
        if component == 0:
            if self.parent_key is not None:
                raise RuntimeError("previous parent K capture not consumed")
            self.parent_key = captured
        else:
            self.parent_value = captured
            self.value = view
        self.charge()

    def __enter__(self):
        from mosaickv.evaluation.eager_core_operator import reference_scores
        from mosaickv.evaluation.inline_first_step_repair import hidden_after_attention
        from mosaickv.evaluation.post_attention_segment import Transport
        from mosaickv.evaluation.uniform_core_transport import Binding

        if _current.get() is not None:
            raise RuntimeError("live parent scopes cannot nest")
        self.token = _current.set(self)
        score, values, epilogue = Binding.scores, Binding.values, Transport.run
        owner = self
        if self.policy and self.policy.anchor_tail != "reference":
            from mosaickv.evaluation.guard_tail_graph import GuardTail

            self.tail = GuardTail(
                self.adapter.model.model.language_model.norm,
                self.adapter.model.get_output_embeddings(),
                capture_mode="manual_shared_stream"
                if self.policy.anchor_tail == "manual_shared_stream"
                else "context_manager",
            )
        if self.policy and self.policy.reuse_anchor_readout:
            from mosaickv.evaluation.guard_readout_reuse import ReadoutReuse

            self.readout = self.stack.enter_context(
                ReadoutReuse(
                    self.adapter.model.model.language_model.norm,
                    self.adapter.model.get_output_embeddings(),
                )
            )

        def observe_scores(binding, query, key, mask, scaling):
            if binding.calls % 32 != 31:
                return score(binding, query, key, mask, scaling)
            if owner.parent_key is None or query.shape != (1, 32, 1, 128):
                raise RuntimeError("actual original last-layer observations required")
            h = owner.overlay.head
            active = None
            if not owner.single_head or owner.repair_enabled:
                active = key[:, h : h + 1].clone()
                owner.charge(active)
            if owner.repair_enabled:
                owner.active_key, owner.query, owner.mask, owner.scaling = (
                    active,
                    query,
                    mask,
                    scaling,
                )
            if owner.single_head:
                owner.probabilities = reference_scores(
                    query[:, h : h + 1], owner.parent_key, head_mask(mask, h), scaling
                )
            else:
                try:
                    key[:, h : h + 1].copy_(owner.parent_key)
                    owner.probabilities = reference_scores(query, key, mask, scaling)
                finally:
                    key[:, h : h + 1].copy_(active)
            owner.parent_key = None
            weights = score(binding, query, key, mask, scaling)
            if owner.repair_enabled:
                owner.prototype_mass = (
                    weights[0, h, 0].index_select(0, owner.overlay.positions).float().sum()
                )
            return weights

        def observe_values(binding, length):
            result = values(binding, length)
            if (binding.calls - 1) % 32 != 31:
                return result
            if owner.parent_value is None or owner.probabilities is None:
                raise RuntimeError("complete parent probability/V transaction required")
            h = owner.overlay.head
            active = None
            if not owner.single_head or owner.repair_enabled:
                active = owner.value[:, h : h + 1].clone()
                owner.charge(active)
            if owner.repair_enabled:
                owner.active_value = active
                owner.active_heads = result.reshape(1, 1, -1).clone()
            if owner.single_head:
                owner.heads = replace_head_output(
                    result.reshape(1, 1, -1), owner.probabilities, owner.parent_value, h
                )
            else:
                try:
                    owner.value[:, h : h + 1].copy_(owner.parent_value)
                    owner.heads = torch.matmul(owner.probabilities, owner.value).transpose(1, 2)
                    owner.heads = owner.heads.contiguous().reshape(1, 1, -1)
                finally:
                    owner.value[:, h : h + 1].copy_(active)
            owner.parent_value = owner.value = owner.probabilities = None
            return result

        def observe_epilogue(transport, layer, index, residual, attention):
            if index != 31:
                return epilogue(transport, layer, index, residual, attention)
            if owner.heads is None:
                raise RuntimeError("parent attention counterfactual did not engage")
            original_residual = residual.clone()
            active_hidden = epilogue(transport, layer, index, residual, attention)
            if owner.policy and owner.policy.anchor_suffix == "shared_graph":
                from mosaickv.evaluation.guard_suffix_replay import replay

                verify = owner.policy.verify_anchor_suffix and len(owner.records) in (0, 63, 126)
                reference_hidden = (
                    hidden_after_attention(layer, original_residual, owner.heads)
                    if verify
                    else None
                )
                active_hidden, parent_hidden, extra = replay(
                    transport, layer, original_residual, owner.heads, active_hidden
                )
                owner.suffix_replays += 1
                owner.suffix_activation_bytes = max(owner.suffix_activation_bytes, extra)
                if verify:
                    if not torch.equal(parent_hidden, reference_hidden):
                        raise RuntimeError("parent shared-graph suffix differs bitwise from native")
                    owner.suffix_verifications += 1
            else:
                parent_hidden = hidden_after_attention(layer, original_residual, owner.heads)
            norm = owner.adapter.model.model.language_model.norm
            head = owner.adapter.model.get_output_embeddings()
            if owner.tail is not None:
                # The opt-in graph path forbids repair/readout-reuse branches.
                # It retains their original reference path when disabled.
                output, parent_logits, record = owner.tail.run(parent_hidden, active_hidden)
                owner.records.append(record)
                owner.parent_logits.append(parent_logits.detach().cpu())
                owner.heads = None
                owner.active_key = owner.active_value = owner.active_heads = None
                owner.query = owner.mask = owner.prototype_mass = None
                return output
            parent_norm, active_norm = norm(parent_hidden), norm(active_hidden)
            parent_native, active_native = head(parent_norm), head(active_norm)
            parent_logits = parent_native.float().reshape(-1)
            active_logits = active_native.float().reshape(-1)
            passed, record = decision(parent_logits, active_logits)
            forced = bool(owner.policy and owner.policy.anchor_repair_stress and not owner.records)
            eligible = (
                owner.repair_enabled
                and not owner.overlay.repair_count
                and owner.policy.component_mode
                not in ("no_repair", "exact_members", "parent_reserved")
            )
            if eligible and (forced or not bool(passed)):
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                event = owner.overlay.promote()
                begin.record()
                for target, source in zip(
                    (owner.active_key, owner.active_value), owner.overlay.promotion, strict=True
                ):
                    target[0, 0].index_copy_(0, owner.overlay.positions, source)
                revised_heads = repaired_head_outputs(
                    owner.query,
                    owner.active_key,
                    owner.active_value,
                    owner.mask,
                    owner.scaling,
                    owner.active_heads,
                    owner.overlay.head,
                )
                repaired_hidden = hidden_after_attention(layer, original_residual, revised_heads)
                repaired_norm = norm(repaired_hidden)
                repaired_native = head(repaired_norm)
                repaired_logits = repaired_native.float().reshape(-1)
                revised_passed, revised_record = decision(parent_logits, repaired_logits)
                end.record()
                end.synchronize()
                detail = dict(
                    event,
                    layer=31,
                    trigger_step=len(owner.records) + 1,
                    trigger_reason="controlled_anchor_repair"
                    if forced
                    else "live_parent_decision_risk",
                    stress_control=forced,
                    current_token_recomputed_once=True,
                    whole_model_redecode_performed=False,
                    re_decode_cuda_ms=begin.elapsed_time(end),
                    before_parent_kl=float(record[0]),
                    after_parent_kl=float(revised_record[0]),
                    before_parent_logit_difference=float(record[1]),
                    after_parent_logit_difference=float(revised_record[1]),
                    parent_token_recovered=not bool(record[2]) and bool(revised_record[2]),
                    quality_recovery_metric="parent_token_not_benchmark_score",
                    repaired_plan_passed=bool(revised_passed),
                    repair_logit_max_change=float((repaired_logits - active_logits).abs().max()),
                    before_token_id=int(active_logits.argmax()),
                    repaired_token_id=int(repaired_logits.argmax()),
                    parent_token_id=int(parent_logits.argmax()),
                    token_changed=int(active_logits.argmax()) != int(repaired_logits.argmax()),
                    prototype_cluster_anchor=owner.overlay.anchor,
                    restored_modality="image",
                    prototype_attention_mass=float(owner.prototype_mass),
                    prototype_attention_mass_times_dispersion=float(owner.prototype_mass)
                    * owner.overlay.dispersion,
                )
                owner.report["repair_events"].append(detail)
                owner.repair_events.append(detail)
                active_hidden, passed, record = repaired_hidden, revised_passed, revised_record
                active_norm, active_native = repaired_norm, repaired_native
            # Report the safety veto separately; never call it a FullKV repair.
            output = torch.where(passed, active_hidden, parent_hidden)
            if owner.readout is not None:
                owner.readout.stage(passed, parent_norm, active_norm, parent_native, active_native)
            owner.records.append(record.detach().cpu())
            owner.parent_logits.append(parent_logits.detach().cpu())
            owner.heads = None
            owner.active_key = owner.active_value = owner.active_heads = None
            owner.query = owner.mask = owner.prototype_mass = None
            return output

        try:
            self.stack.enter_context(patch.object(Binding, "scores", observe_scores))
            self.stack.enter_context(patch.object(Binding, "values", observe_values))
            self.stack.enter_context(patch.object(Transport, "run", observe_epilogue))
            return self
        except BaseException:
            self.stack.close()
            _current.reset(self.token)
            raise

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            _current.reset(self.token)
            stats = torch.stack(self.records) if self.records else torch.empty(0, 4)
            self.report["live_anchor_guard"] = dict(
                steps=len(self.records),
                reference="live_parent_storage_variant_not_FullKV",
                counterfactual_scope="single_head" if self.single_head else "full_heads",
                readout_reuse=self.readout.report() if self.readout is not None else None,
                tail_graph=self.tail.report() if self.tail is not None else None,
                suffix_graph_replays=self.suffix_replays,
                early_middle_late_native_suffix_verifications=self.suffix_verifications,
                suffix_saved_hidden_activation_bytes=self.suffix_activation_bytes,
                suffix_additional_persistent_KV_bytes=0,
                kl_threshold=0.02,
                logit_threshold=1.0,
                greedy_token_required=True,
                prototype_or_exact_changes_vetoed=int((stats[:, 3] == 0).sum()),
                accepted_overlay_steps=int(stats[:, 3].sum()),
                observed=stats.tolist(),
                peak_temporary_kv_backing_bytes=self.peak_temporary_kv_bytes,
                cpu_logit_diagnostic_bytes=backing_bytes(tuple(self.parent_logits)),
                fullkv_fallback=False,
                benchmark_labels_accessed=False,
                parent_counterfactual_parity_not_assumed=True,
                whole_request_peak_certified=False,
                decision_risk_repair_enabled=self.repair_enabled,
                decision_risk_repair_events=self.repair_events,
                coexisting_temporary_kv_peak_bytes=self.coexisting_temporary_kv_peak_bytes,
            )
            self.parent_key = self.parent_value = self.value = self.probabilities = self.heads = (
                None
            )
            self.layer = None
            self.active_key = self.active_value = self.active_heads = self.query = None
            self.prototype_mass = None
            if self.tail is not None:
                self.tail.close()


def head_mask(mask, head):
    """Preserve shared masks, narrow real per-head masks; reject ambiguity."""
    if mask is None:
        return None
    if mask.ndim != 4 or mask.shape[1] not in (1, 32):
        raise ValueError("audited shared or full-MHA four-dimensional mask required")
    return mask if mask.shape[1] == 1 else mask[:, head : head + 1]


def replace_head_output(active, probabilities, value, head):
    if (
        type(head) is not int
        or not 0 <= head < 32
        or active.shape != (1, 1, 4096)
        or probabilities.ndim != 4
        or probabilities.shape[:3] != (1, 1, 1)
        or value.shape != (1, 1, probabilities.shape[-1], 128)
        or any(t.dtype != active.dtype or t.device != active.device for t in (probabilities, value))
    ):
        raise ValueError("complete matching actual attention head/output required")
    revised = torch.matmul(probabilities, value).reshape(1, 1, 128)
    result = active.clone().view(1, 1, 32, 128)
    result[:, :, head] = revised
    return result.reshape(1, 1, 4096)


def repaired_head_outputs(query, key, value, mask, scaling, active_heads, head):
    """Current routed-head attention; untouched heads stay bitwise unchanged."""
    from mosaickv.evaluation.eager_core_operator import reference_scores

    if (
        type(head) is not int
        or not 0 <= head < 32
        or query.shape != (1, 32, 1, 128)
        or key.ndim != 4
        or key.shape[:2] != (1, 1)
        or key.shape[-1] != 128
        or value.shape != key.shape
        or active_heads.shape != (1, 1, 4096)
        or any(
            t.dtype != torch.bfloat16 or t.device != query.device
            for t in (query, key, value, active_heads)
        )
    ):
        raise ValueError(
            "actual BF16 last-layer head, complete routed KV and native outputs required"
        )
    probability = reference_scores(query[:, head : head + 1], key, mask, scaling)
    revised = torch.matmul(probability, value).reshape(1, 1, -1)
    result = active_heads.clone().view(1, 1, 32, 128)
    result[:, :, head] = revised
    return result.reshape(1, 1, -1)


@torch.inference_mode()
def repair_operator_controls(device="cuda"):
    """Before model loading: prove single-head and full-MHA attention agree."""
    from mosaickv.evaluation.eager_core_operator import reference_scores

    generator = torch.Generator(device=device).manual_seed(20261107)
    cases = []
    for length in (603, 604, 730):
        q = torch.randn(1, 32, 1, 128, generator=generator, device=device).bfloat16()
        key, value = tuple(
            torch.randn(1, 32, length, 128, generator=generator, device=device).bfloat16()
            for _ in range(2)
        )
        for dtype in (None, torch.bfloat16, torch.float32):
            mask = (
                None if dtype is None else torch.zeros(1, 1, 1, length, device=device, dtype=dtype)
            )
            if mask is not None:
                mask[..., 2] = -torch.inf
            weights = reference_scores(q, key, mask, 128**-0.5)
            expected = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(1, 1, -1)
            observed = repaired_head_outputs(
                q, key[:, 4:5], value[:, 4:5], mask, 128**-0.5, expected, 4
            )
            passed = torch.equal(expected, observed)
            cases.append(dict(length=length, mask_dtype=str(dtype), bitwise=passed))
            if not passed:
                raise RuntimeError(
                    "single-head repair attention differs from actual full-MHA operator"
                )
    return dict(passed=True, cases=cases, model_or_labels_used=False)
