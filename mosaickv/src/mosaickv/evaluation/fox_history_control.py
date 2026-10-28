"""Evaluation-only diagnosis of geometry error versus accumulated KV-history drift.

Never imported by the online compressor. A synchronized reference is not repair,
nor a deployable generation method, nor benchmark-quality evidence.
"""

from __future__ import annotations

from mosaickv.evaluation.fox_numerics import _step, independent_state


def padded_same_history(adapter, full, *, gap, padding):
    import torch

    source = adapter._legacy_layers(full.past_key_values)[0]
    positions = {
        (layer, head): tuple(range(key.shape[2]))
        for layer, (key, value) in enumerate(source)
        for head in range(key.shape[1])
    }
    result, masks = independent_state(
        adapter, full, positions, original_layout=True, padding=padding
    )
    destination = adapter._legacy_layers(result.past_key_values)[0]
    with torch.inference_mode():
        for (key, value), (target_key, target_value), mask in zip(
            source, destination, masks, strict=True
        ):
            target_key[:, :, gap:] = 0
            target_value[:, :, gap:] = 0
            target_key[:, :, gap + padding :] = key[:, :, gap:]
            target_value[:, :, gap + padding :] = value[:, :, gap:]
            mask[:, gap:] = False
            mask[:, gap + padding :] = True
    return result, masks


def identical_history(adapter, full, padded, *, gap, padding):
    import torch

    source = adapter._legacy_layers(full.past_key_values)[0]
    target = adapter._legacy_layers(padded.past_key_values)[0]
    if full.next_decode_position != padded.next_decode_position:
        return False
    for (key, value), (pk, pv) in zip(source, target, strict=True):
        indices = torch.cat(
            (
                torch.arange(gap, device=key.device),
                torch.arange(gap + padding, pk.shape[2], device=key.device),
            )
        )
        if (
            key.dtype != pk.dtype
            or key.device != pk.device
            or not torch.equal(key, pk.index_select(2, indices))
            or not torch.equal(value, pv.index_select(2, indices))
        ):
            return False
    return True


def diagnose_history(adapter, original, first_token, *, steps=16):
    import torch

    from mosaickv.backends.hf_runtime import _clone_decode_state

    full = _clone_decode_state(adapter, original)
    gap = original.active_cache_length
    padding = max(1, gap // 3)
    persistent, persistent_masks = padded_same_history(adapter, full, gap=gap, padding=padding)
    synchronized, synchronized_masks = padded_same_history(adapter, full, gap=gap, padding=padding)
    # The unpadded source uses its own independently generated all-valid mask.
    all_positions = {
        (layer, head): tuple(range(k.shape[2]))
        for layer, (k, value) in enumerate(adapter._legacy_layers(full.past_key_values)[0])
        for head in range(k.shape[1])
    }
    full, full_masks = independent_state(adapter, full, all_positions, original_layout=True)
    token = torch.tensor([[int(first_token)]], device=adapter.device, dtype=torch.long)
    records = []
    for step in range(steps):
        rebased, rebased_masks = padded_same_history(adapter, full, gap=gap, padding=padding)
        if not identical_history(
            adapter, full, rebased, gap=gap, padding=padding
        ) or not identical_history(adapter, full, synchronized, gap=gap, padding=padding):
            raise AssertionError("same-source history control is not bitwise identical")
        reference = _step(adapter, token, full, full_masks)
        natural = _step(adapter, token, persistent, persistent_masks)
        independent = _step(adapter, token, rebased, rebased_masks)
        synchronized_output = _step(adapter, token, synchronized, synchronized_masks)
        norm = max(1.0, float(reference.logits.float().abs().max().item()))
        item = dict(
            step=step,
            identical_source_history_before_forward=True,
            rebased_synchronized_max_absolute=float(
                (independent.logits.float() - synchronized_output.logits.float()).abs().max().item()
            ),
        )
        for name, out in (
            ("persistent", natural),
            ("rebased", independent),
            ("synchronized", synchronized_output),
        ):
            if not bool(torch.isfinite(out.logits).all()):
                raise AssertionError("nonfinite numerical diagnostic")
            item[name + "_normalized_error"] = (
                float((reference.logits.float() - out.logits.float()).abs().max().item()) / norm
            )
            item[name + "_greedy_equal"] = bool(
                torch.equal(reference.next_token_id, out.next_token_id)
            )
        records.append(item)
        # Evaluation-only history replacement, AFTER reporting the current
        # logits. This is never used by model evaluation or online generation.
        with torch.inference_mode():
            for (key, value), (pk, pv) in zip(
                adapter._legacy_layers(full.past_key_values)[0],
                adapter._legacy_layers(synchronized.past_key_values)[0],
                strict=True,
            ):
                pk[:, :, gap + padding :] = key[:, :, gap:]
                pv[:, :, gap + padding :] = value[:, :, gap:]
        token = reference.next_token_id
    return dict(
        records=records,
        steps=steps,
        padding_slots_per_head=padding,
        same_source_controls_evaluation_only=True,
        benchmark_quality_evidence=False,
        cache_dtype=str(adapter._legacy_layers(full.past_key_values)[0][0][0].dtype),
    )
