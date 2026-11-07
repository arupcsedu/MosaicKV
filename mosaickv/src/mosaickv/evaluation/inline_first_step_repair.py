"""Same entropy policy, one committed decode: no disposable whole-model probe.

Only the already verified final-layer suffix may be recomputed. K/V history
for this token is unchanged by its final-layer attention output. All original
positions, exact/prototype allocations and the 128-prediction loop remain.
"""

from contextlib import ExitStack
from unittest.mock import patch

import torch

from mosaickv.evaluation.exact_last_layer_readout import attention_output
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def hidden_after_attention(layer, residual, head_outputs):
    hidden = residual + layer.self_attn.o_proj(head_outputs)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class InlineRepair:
    def __init__(self, adapter, overlay, policy, report):
        if policy.layers != (31,) or policy.readout_selection != "none" or policy.readout_control:
            raise NotImplementedError("isolated last-layer entropy-policy transport required")
        self.adapter, self.overlay, self.policy, self.report = adapter, overlay, policy, report
        self.stack, self.done = ExitStack(), False
        self.key = self.value = self.query = self.mask = self.heads = None
        self.scaling = None
        self.evidence = dict(
            enabled=True,
            separate_compressed_probe_updates=0,
            first_committed_update_only=True,
            suffix_recomputations=0,
            temporary_key_clone_backing_bytes=0,
            original_logical_positions_preserved=True,
            whole_request_peak_certified=False,
        )

    def __enter__(self):
        from mosaickv.evaluation.post_attention_segment import Transport
        from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
        from mosaickv.evaluation.uniform_core_transport import Binding

        fill, scores, values, epilogue = (
            SequentialSourceHistoryLayer.materialize_component,
            Binding.scores,
            Binding.values,
            Transport.run,
        )
        owner = self

        def observe_fill(layer, component):
            result = fill(layer, component)
            if not owner.done and layer.attention.layer_idx == 31:
                view = layer.leased[0][..., : layer.get_seq_length(), :]
                if component == 0:
                    if owner.key is not None or torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("one outside-capture first-step key snapshot required")
                    owner.key = torch.empty_strided(
                        view.shape, view.stride(), dtype=view.dtype, device=view.device
                    )
                    owner.key.copy_(view)
                    owner.evidence["temporary_key_clone_backing_bytes"] = backing_bytes(
                        (owner.key,)
                    )
                    layer.observe((owner.key,))
                else:
                    # Existing charged shared lease; no extra V allocation.
                    owner.value = view
            return result

        def observe_scores(binding, query, key, mask, scaling):
            result = scores(binding, query, key, mask, scaling)
            if not owner.done and (binding.calls - 1) % 32 == 31:
                owner.query, owner.mask, owner.scaling = binding.core.q, mask, scaling
            return result

        def observe_values(binding, length):
            result = values(binding, length)
            if not owner.done and (binding.calls - 1) % 32 == 31:
                owner.heads = result
            return result

        def observe_epilogue(transport, layer, index, residual, attention):
            if owner.done or index != 31:
                return epilogue(transport, layer, index, residual, attention)
            # The native epilogue reuses its preceding output buffer. Preserve
            # the actual input residual before calling it, not a detached alias.
            original_residual = residual.clone()
            output = epilogue(transport, layer, index, residual, attention)
            return owner.finish(transport, layer, original_residual, output)

        self.stack.enter_context(
            patch.object(SequentialSourceHistoryLayer, "materialize_component", observe_fill)
        )
        self.stack.enter_context(patch.object(Binding, "scores", observe_scores))
        self.stack.enter_context(patch.object(Binding, "values", observe_values))
        self.stack.enter_context(patch.object(Transport, "run", observe_epilogue))
        return self

    def finish(self, transport, layer, residual, output):
        from mosaickv.repair.core import normalized_next_token_entropy

        if (
            any(t is None for t in (self.key, self.value, self.query, self.heads))
            or self.scaling is None
        ):
            raise RuntimeError("actual original reader observations missing; no simulated repair")
        if tuple(self.query.shape) != (1, 32, 1, 128) or layer.self_attn.layer_idx != 31:
            raise NotImplementedError("registered full-MHA first-update geometry required")
        norm = self.adapter.model.model.language_model.norm
        head = self.adapter.model.get_output_embeddings()
        original_logits = head(norm(output)).float().reshape(-1)
        entropy = normalized_next_token_entropy(original_logits)
        self.report["provisional_entropy"] = entropy
        forced = self.policy.force_repair or self.policy.component_mode == "forced_repair"
        should_repair = self.policy.component_mode not in (
            "no_repair",
            "exact_members",
            "parent_reserved",
        ) and (forced or entropy >= self.policy.entropy_threshold)
        self.evidence["provisional_entropy"] = entropy
        self.evidence["private_model_probe_performed"] = False
        self.evidence["temporary_attention_residual_backing_bytes"] = backing_bytes((residual,))
        transport.cache.observe((self.key,))
        if should_repair:
            event = self.overlay.promote()
            for destination, source in zip(
                (self.key, self.value), self.overlay.promotion, strict=True
            ):
                destination[0, self.overlay.head].index_copy_(0, self.overlay.positions, source)
            projected = attention_output(self.query, self.key, self.value, self.mask, self.scaling)
            output = hidden_after_attention(layer, residual, projected)
            revised_logits = head(norm(output)).float().reshape(-1)
            self.report["repair_logit_max_change"] = float(
                (revised_logits - original_logits).abs().max()
            )
            self.report["repair_token_changed"] = int(revised_logits.argmax()) != int(
                original_logits.argmax()
            )
            self.evidence["suffix_recomputations"] = 1
            self.report["repair_events"].append(
                dict(
                    event,
                    layer=31,
                    trigger_reason="controlled_entropy_zero" if forced else "entropy",
                    stress_control=forced,
                    current_token_recomputed_once=True,
                    whole_model_redecode_performed=False,
                )
            )
        else:
            self.report["repair_logit_max_change"] = 0.0
            self.report["repair_token_changed"] = False
        self.done = True
        self.key = self.value = self.query = self.mask = self.heads = None
        return output

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.key = self.value = self.query = self.mask = self.heads = None
            self.evidence["completed"] = self.done
            self.evidence["retained_private_tensor_owners"] = 0
            self.report["inline_first_step_repair"] = self.evidence
