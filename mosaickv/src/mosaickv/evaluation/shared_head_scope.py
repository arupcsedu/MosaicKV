"""Isolated all-head source-prototype ownership diagnostic; DEFAULT OFF.

Not a silent promotion of the single-head six-component overlay. The planner
uses actual prompt demand and source reconstruction error. It substitutes one
source group per layer before numerical transport fitting, retaining every
logical position and original exact K/V protection. Model guards remain external.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from unittest.mock import patch

import torch

from mosaickv.evaluation.shared_head_prototypes import construct, integers

_demand = ContextVar("shared_head_prompt_demand", default=None)


@torch.inference_mode()
def choose_group(body, text, demand, size):
    n, width = body.shape[2:]
    if (
        type(size) is not int
        or size not in (4, 8, 16)
        or demand.shape != (n,)
        or not bool(torch.isfinite(demand).all())
        or bool((demand < 0).any())
    ):
        raise ValueError("registered group size and actual nonnegative prompt demand required")
    protected = set(text) | {n - 1}
    candidates = [
        tuple(range(start, start + size))
        for start in range(0, n - size + 1, size)
        if start // 128 == (start + size - 1) // 128
        and not protected.intersection(range(start, start + size))
    ]
    if not candidates:
        raise ValueError("no feasible nontext source prototype group")
    codes = integers(body.codes, n, width).float()
    ids = torch.tensor(candidates, device=codes.device, dtype=torch.long)
    values = codes[ids]
    center = values.mean(1, keepdim=True).round()
    # Account for actual quantizer scale, not just integer-code distance.
    scale = body.scale.reshape(width, -1).transpose(0, 1)
    error = ((values - center) * scale[ids[:, 0] // 128, None]).square().mean(-1)
    risk = (error * demand[ids]).sum(-1)
    index = int(torch.argmin(risk))  # deterministic source-position tie break
    return candidates[index], dict(
        candidates=len(candidates),
        forecast="mean_actual_last_16_prompt_probabilities_across_all_heads",
        selection="minimum_prompt_demand_weighted_affine_source_error",
        risk=float(risk[index]),
        not_the_six_component_submodular_planner=True,
    )


@contextmanager
def use(*, group_size=8, map_policy="independent"):
    from mosaickv.evaluation import shared_source_format as formats
    from mosaickv.evaluation import streaming_source_constructor as constructor

    if map_policy not in ("independent", "first_layer_shared"):
        raise ValueError("explicit registered source-map policy required")
    original_encoder, original_construct = constructor.StreamingDirectSource, formats.construct
    records, planning = [], []
    common_group, common_map = None, None

    class Encoder(original_encoder):
        def __call__(self, index, attention, original, args, kwargs, output):
            probabilities = output[1][0, :, -16:].float().mean((0, 1))
            token = _demand.set(probabilities)
            try:
                return super().__call__(index, attention, original, args, kwargs, output)
            finally:
                _demand.reset(token)

    def build(normalized, original_pair, memberships, text, phases, identity, **kwargs):
        nonlocal common_group, common_map
        packet = original_construct(
            normalized, original_pair, memberships, text, phases, identity, **kwargs
        )
        demand = _demand.get()
        if demand is None:
            raise RuntimeError("all-head prototypes require actual constructor prompt evidence")
        if map_policy == "independent" or common_group is None:
            group, evidence = choose_group(packet.source, text, demand, group_size)
            common_group = group
        else:
            group = common_group
            evidence = dict(
                selection="frozen_first_layer_prompt_demand_group", no_second_prefill=True
            )
        transaction = construct(packet.source, (group,), heads=packet.shape[1], mandatory=text)
        if map_policy == "first_layer_shared":
            if common_map is None:
                common_map = transaction.body.row_map
            elif not torch.equal(common_map, transaction.body.row_map):
                raise RuntimeError("common source map changed physical alignment")
            transaction.body = replace(transaction.body, row_map=common_map, row_map_shared=True)
        # The codec must be allowed to retire the NEW raw prototype codes as
        # well. Keep only CPU residuals/metadata/weak original-owner evidence.
        records.append(
            (transaction.record(), transaction.residual_codes, transaction.original_owner)
        )
        planning.append(evidence)
        result = replace(packet, source=transaction.body)
        # Construction is explicitly pinned to 32 layers; retain no hidden
        # GPU map owner beyond the immutable packets after the last layer.
        if len(records) == 32:
            common_map = None
        return result

    report = dict(
        variant="all_head_source_prototype_ownership_diagnostic_v1",
        group_size=group_size,
        map_policy=map_policy,
        benchmark_labels_accessed=False,
        six_component_claim_authorized=False,
        natural_model_repair_validated=False,
    )
    with (
        patch.object(constructor, "StreamingDirectSource", Encoder),
        patch.object(formats, "construct", build),
    ):
        try:
            yield report
        finally:
            common_map = None
            report["layers"] = [
                dict(
                    metadata,
                    layer=i,
                    old_source_tensor_alive=ref() is not None,
                    planning=planning[i],
                )
                for i, (metadata, _residual, ref) in enumerate(records)
            ]
            report["all_original_source_tensor_objects_retired"] = bool(records) and all(
                ref() is None for _metadata, _residual, ref in records
            )
            report["CPU_residual_bytes"] = sum(
                metadata["CPU_residual_bytes"] for metadata, _residual, _ref in records
            )
            report["quality_preservation_not_inferred_from_execution"] = True
