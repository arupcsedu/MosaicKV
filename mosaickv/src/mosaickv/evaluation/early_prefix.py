"""Read-only, consumed-input FullKV-prefix replay; never an online selector."""

from contextlib import ExitStack, nullcontext
from unittest.mock import patch

from mosaickv.backends.hf_prefix_outputs import PrefixOutputRecorder


def position_metadata(position, spans):
    """Only authoritative modality/grid coordinates; no inferred OCR semantics."""
    for span in spans:
        if span["start"] <= position < span["end"]:
            row = dict(
                modality=span["modality"],
                image_index=span.get("image_index"),
                grid_row=None,
                grid_column=None,
            )
            grid = span.get("grid_shape")
            if (
                span["modality"] == "image"
                and grid
                and span["end"] - span["start"] == grid[0] * grid[1]
            ):
                row["grid_row"], row["grid_column"] = divmod(position - span["start"], grid[1])
            return row
    return dict(modality="unknown", image_index=None, grid_row=None, grid_column=None)


def retained_positions(packed):
    """Recover original per-head prompt positions, never renumber compact slots."""
    layers = [[[] for _ in range(x.key.shape[1])] for x in packed.snapshot.layers]
    for r in packed.slot_records:
        if r["tier"] != "exact":
            raise ValueError("early-prefix replay excludes transformed cache entries")
        layers[r["layer"]][r["kv_head"]].append((r["slot"], r["logical_position"]))
    result = [[[p for _, p in sorted(head)] for head in layer] for layer in layers]
    for layer, tensor in zip(result, packed.snapshot.layers, strict=True):
        if any(len(h) != tensor.key.shape[-2] or len(set(h)) != len(h) for h in layer):
            raise ValueError("incomplete or padded exact prompt mapping")
    return result


def omission_decomposition(probabilities, values, positions, source_length):
    """FP32 frozen-FullKV-query decomposition, NOT a candidate-forward prediction.

    o_full - o_kept = sum_omitted p_i * (v_i - o_kept), where o_kept
    renormalizes across selected prompt AND all generated positions. Actual
    compressed queries and generated values may differ; record those separately.
    """
    import torch

    if probabilities.ndim != 4 or probabilities.shape[0:1] != (1,) or probabilities.shape[-2] != 1:
        raise ValueError("one-query eager probabilities required")
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[-2] != probabilities.shape[-1]:
        raise ValueError("attention/value geometry mismatch")
    p = probabilities[0, :, 0].float()
    heads, kv_heads = p.shape[0], values.shape[1]
    if heads % kv_heads or len(positions) != kv_heads or not 0 < source_length <= p.shape[-1]:
        raise ValueError("invalid GQA or source prompt boundary")
    v = values[0].float().repeat_interleave(heads // kv_heads, dim=0)
    mask = torch.zeros_like(p, dtype=torch.bool)
    for head, selected in enumerate(positions):
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(not 0 <= x < source_length for x in selected)
        ):
            raise ValueError("invalid exact prompt positions")
        mask[head * (heads // kv_heads) : (head + 1) * (heads // kv_heads), selected] = True
    mask[:, source_length:] = True
    mass = (p * mask).sum(-1)
    if not bool(torch.isfinite(p).all() and torch.isfinite(v).all() and (mass > 0).all()):
        raise ValueError("invalid probabilities/retained mass")
    full = torch.einsum("hn,hnd->hd", p, v)
    kept = torch.einsum("hn,hnd->hd", p * mask, v) / mass[:, None]
    delta = full - kept
    omitted = (~mask).to(p.dtype)
    contribution = p[..., None] * omitted[..., None] * (v - kept[:, None])
    # Softmax probabilities are stored in cache dtype and need not sum to
    # exactly one after casting. Preserve that observed rounding term explicitly.
    rounding = kept * (p.sum(-1) - 1)[:, None]
    reconstruction = contribution.sum(1) + rounding
    error = float((delta - reconstruction).abs().max().item())
    if error > 1e-4 * max(1.0, float(full.abs().max().item())):
        raise RuntimeError("frozen-query decomposition identity failed")
    norm = delta.norm(dim=-1)
    direction = delta / norm.clamp_min(1e-12)[:, None]
    signed = (contribution * direction[:, None]).sum(-1)
    return dict(
        full_head_output=full.detach().cpu(),
        retained_head_output=kept.detach().cpu(),
        omitted_signed_output_contribution=signed[:, :source_length].detach().cpu(),
        prompt_probabilities=p[:, :source_length].detach().cpu(),
        retained_prompt_mask=mask[:, :source_length].detach().cpu(),
        retained_mass=mass.detach().cpu(),
        output_error_norm=norm.detach().cpu(),
        softmax_rounding_output=rounding.detach().cpu(),
        identity_max_abs_error=error,
        interpretation="frozen_FullKV_query_algebra_not_actual_compressed_forward_or_causal_restore_test",
    )


class EarlyPrefixCapture:
    """Scoped hooks; label-free forced tokens; isolated roles run sequentially."""

    def __init__(
        self,
        adapter,
        runtime_module,
        targets,
        *,
        role,
        observed,
        reference_positions=None,
        capture_predictions=(2, 4, 8, 16, 20, 21, 22, 23, 24, 32),
    ):
        if not targets or len(targets) > 32 or any(type(t) is not int or t < 0 for t in targets):
            raise ValueError("bounded saved FullKV token prefix required")
        self.adapter, self.module, self.targets = adapter, runtime_module, tuple(targets)
        self.role, self.observed = role, observed
        self.reference_positions = reference_positions or {}
        self.capture_predictions = tuple(p for p in capture_predictions if p <= len(targets))
        self.logits, self.positions, self.decompositions = [], None, {}
        self.source_length, self.modality_map = None, ()
        self.recorder = PrefixOutputRecorder(adapter)
        self.stack = ExitStack()

    def score(self, logits):
        if len(self.logits) >= len(self.targets):
            raise RuntimeError("unexpected extra forward")
        self.logits.append(logits.detach().reshape(-1, logits.shape[-1])[-1].float().cpu().clone())

    def __enter__(self):
        original_prefill, original_decode = self.adapter.prefill, self.adapter.decode_one_token
        original_pack = self.module._state_from_packed

        def packed(adapter, state, choice):
            if self.positions is not None:
                raise RuntimeError("multiple prompt constructions in fixed-prefix replay")
            self.positions = retained_positions(choice)
            return original_pack(adapter, state, choice)

        def prefill(*args, **kwargs):
            if self.logits:
                raise RuntimeError("second prefill in one replay")
            output = original_prefill(*args, **kwargs)
            self.source_length = output.state.logical_sequence_length
            self.modality_map = output.state.modality_map
            self.score(output.logits)
            return output

        def decode(token, state, *args, **kwargs):
            index = len(self.logits)
            if not index or int(token.item()) != self.targets[index - 1]:
                raise RuntimeError("replay prefix differs from saved FullKV tokens")
            prediction = index + 1
            capture = self.observed and prediction in self.capture_predictions
            decompose = capture and prediction == 23 and bool(self.reference_positions)
            if decompose:
                kwargs["capture_attentions"] = True
            with self.recorder.capture(self.role, prediction) if capture else nullcontext():
                output = original_decode(token, state, *args, **kwargs)
            self.score(output.logits)
            if any(getattr(x, "eviction_count", 0) for x in output.state.past_key_values.layers):
                raise RuntimeError("history eviction confounds early-prefix replay")
            if decompose:
                values = self.adapter._legacy_layers(output.state.past_key_values)[0]
                for role, positions in self.reference_positions.items():
                    self.decompositions[role] = [
                        omission_decomposition(p, v, selected, self.source_length)
                        for p, (_k, v), selected in zip(
                            output.attention_weights, values, positions, strict=True
                        )
                    ]
            return output

        self.stack.enter_context(patch.object(self.module, "_state_from_packed", packed))
        self.stack.enter_context(patch.object(self.adapter, "prefill", prefill))
        self.stack.enter_context(patch.object(self.adapter, "decode_one_token", decode))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def validate_complete(self):
        if len(self.logits) != len(self.targets):
            raise RuntimeError("incomplete saved-prefix replay")
        if self.observed and set(self.recorder.records[self.role]) != set(self.capture_predictions):
            raise RuntimeError("missing layer-output captures")


def compare_layer_outputs(full, compressed):
    """Actual model outputs; error growth is descriptive, not a causal attribution."""
    import torch

    if set(full) != set(compressed):
        raise ValueError("paired prediction captures required")
    rows = []
    for prediction in sorted(full):
        for layer, (a, b) in enumerate(zip(full[prediction], compressed[prediction], strict=True)):
            for field in PrefixOutputRecorder.FIELDS:
                x, y = a[field].double(), b[field].double()
                if x.shape != y.shape or not bool(
                    torch.isfinite(x).all() and torch.isfinite(y).all()
                ):
                    raise ValueError("invalid paired layer outputs")
                rows.append(
                    dict(
                        prediction=prediction,
                        layer=layer,
                        field=field,
                        maximum_absolute_error=float((x - y).abs().max()),
                        l2_error=float((x - y).norm()),
                        relative_l2_error=float((x - y).norm() / x.norm().clamp_min(1e-12)),
                    )
                )
    return rows
