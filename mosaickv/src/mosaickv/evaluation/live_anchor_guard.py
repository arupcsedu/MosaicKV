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
    kl = (p.softmax(-1) * (p.log_softmax(-1) - c.log_softmax(-1))).sum()
    maximum = (p - c).abs().max()
    same = p.argmax() == c.argmax()
    passed = finite & same & (kl <= 0.02) & (maximum <= 1.0)
    return passed, torch.stack((kl, maximum, same.double(), passed.double()))


class LiveAnchorGuard:
    def __init__(self, adapter, overlay, report):
        self.adapter, self.overlay, self.report = adapter, overlay, report
        self.stack, self.records, self.parent_logits = ExitStack(), [], []
        self.parent_key = self.parent_value = self.probabilities = self.heads = None
        self.layer = self.query = self.key = self.value = None
        self.peak_temporary_kv_bytes = 0

    def charge(self, *extra):
        tensors = tuple(t for t in (self.parent_key, self.parent_value, *extra) if t is not None)
        self.peak_temporary_kv_bytes = max(self.peak_temporary_kv_bytes, backing_bytes(tensors))
        self.layer.observe(tensors)

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

        def observe_scores(binding, query, key, mask, scaling):
            if binding.calls % 32 != 31:
                return score(binding, query, key, mask, scaling)
            if owner.parent_key is None or query.shape != (1, 32, 1, 128):
                raise RuntimeError("actual original last-layer observations required")
            h = owner.overlay.head
            active = key[:, h : h + 1].clone()
            owner.charge(active)
            try:
                key[:, h : h + 1].copy_(owner.parent_key)
                owner.probabilities = reference_scores(query, key, mask, scaling)
            finally:
                key[:, h : h + 1].copy_(active)
            owner.parent_key = None
            return score(binding, query, key, mask, scaling)

        def observe_values(binding, length):
            result = values(binding, length)
            if (binding.calls - 1) % 32 != 31:
                return result
            if owner.parent_value is None or owner.probabilities is None:
                raise RuntimeError("complete parent probability/V transaction required")
            h = owner.overlay.head
            active = owner.value[:, h : h + 1].clone()
            owner.charge(active)
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
            parent_hidden = hidden_after_attention(layer, original_residual, owner.heads)
            norm = owner.adapter.model.model.language_model.norm
            head = owner.adapter.model.get_output_embeddings()
            parent_logits = head(norm(parent_hidden)).float().reshape(-1)
            active_logits = head(norm(active_hidden)).float().reshape(-1)
            passed, record = decision(parent_logits, active_logits)
            # Report the safety veto separately; never call it a FullKV repair.
            output = torch.where(passed, active_hidden, parent_hidden)
            owner.records.append(record.detach().cpu())
            owner.parent_logits.append(parent_logits.detach().cpu())
            owner.heads = None
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
            )
            self.parent_key = self.parent_value = self.value = self.probabilities = self.heads = (
                None
            )
            self.layer = None
