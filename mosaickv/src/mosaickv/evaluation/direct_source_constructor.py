"""Online shared-source construction without building a discarded shadow codec.

This is the same explicitly lossy source VARIANT, not mosaickv_exact. Selection
uses only current prompt probabilities and the pinned profile. No annotations,
reference outputs, saved selections or second prefill are accepted.
"""

from dataclasses import dataclass

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.request_precision import online_frozen
from mosaickv.evaluation.shared_source_format import construct as construct_layer
from mosaickv.evaluation.shared_source_format import fit_transport
from mosaickv.evaluation.source_retile import retile
from mosaickv.evaluation.text_source_storage import optimize


@dataclass(frozen=True)
class DirectSourcePolicy:
    ratio: float = 0.5
    head_group: int = 4
    lossless_minimum: bool = True

    def __post_init__(self):
        if (
            type(self.ratio) is not float
            or self.ratio != 0.5
            or type(self.head_group) is not int
            or self.head_group not in (1, 2, 4)
            or type(self.lossless_minimum) is not bool
        ):
            raise ValueError(
                "explicit frozen half-source selection, audited tiling and header policy required"
            )


DEFAULT_POLICY = DirectSourcePolicy()


@torch.inference_mode()
def construct(
    adapter,
    prepared,
    prefill,
    source_capture,
    text_capture,
    phase_capture,
    profile,
    cap_bytes,
    *,
    policy=DEFAULT_POLICY,
    priors=None,
):
    if type(cap_bytes) is not int or cap_bytes <= 0 or not isinstance(policy, DirectSourcePolicy):
        raise ValueError("positive hard physical cap and explicit source policy required")
    frozen, online = online_frozen(
        prefill,
        tuple(profile),
        policy.ratio,
        method="mosaickv_exact",
        allocation_mode="balanced",
        priors=priors,
    )
    memberships = frozen["uniform_0.5_control"]["anchor"]["positions"]
    return _encode_selected(
        adapter,
        prepared,
        prefill,
        source_capture,
        text_capture,
        phase_capture,
        memberships,
        cap_bytes,
        policy=policy,
        online=online,
    )


@torch.inference_mode()
def _encode_selected(
    adapter,
    prepared,
    prefill,
    source_capture,
    text_capture,
    phase_capture,
    memberships,
    cap_bytes,
    *,
    policy,
    online,
):
    """Shared implementation for immediate-parent equality controls only."""
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    h, n = original[0][0].shape[1:3]
    if (
        len(source_capture.records) != len(original)
        or len(text_capture.records) != len(original)
        or len(adapter._language_layers()) != len(original)
        or any(tuple(pair[0].shape) != (1, h, n, 128) for pair in original)
        or prepared.logical_sequence_length != prefill.state.logical_sequence_length
    ):
        raise ValueError(
            "complete same-prefill full-MHA source, text, phases and logical metadata required"
        )
    identity = (
        torch.arange(n, device=adapter.device, dtype=torch.int32),
        torch.arange(n, device=adapter.device, dtype=torch.int16)[None].expand(h, n),
    )
    packets = []
    for i, (pair, block, heads) in enumerate(
        zip(original, adapter._language_layers(), memberships, strict=True)
    ):
        packet = construct_layer(
            source_capture.records[i],
            pair,
            heads,
            text_capture.positions,
            phase_capture.phases,
            identity,
            exclude_global_exact=True,
        )
        packet, _ = fit_transport(packet, block.self_attn)
        packet, _ = optimize(
            packet,
            text_capture.records[i],
            (tuple(range(n)),) * h,
            text_capture.positions,
            block.self_attn,
        )
        packet = retile(packet, policy.head_group)
        if policy.lossless_minimum:
            from mosaickv.evaluation.lossless_affine_minimum import encode_format

            packet = encode_format(packet)
        packets.append(packet)
    packets = tuple(packets)
    size = backing_bytes(tuple(t for p in packets for t in p.tensors()))
    if size > cap_bytes:
        raise RuntimeError(f"source construction exceeds frozen hard cap: {size}>{cap_bytes}")
    return packets, dict(
        source_memberships=memberships,
        online=online,
        physical_bytes=size,
        cap_bytes=cap_bytes,
        source_prefill_reused=True,
        no_discarded_shadow_codec_constructed=True,
        no_benchmark_label_or_reference_access=True,
        all_six_original_mechanisms_replaced_by_source_precision_variant=True,
        original_logical_length=prefill.state.logical_sequence_length,
        whole_request_peak_certified=False,
    )
