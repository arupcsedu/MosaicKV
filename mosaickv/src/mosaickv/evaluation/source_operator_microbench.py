"""No model inference: numeric and randomized CUDA-event operator controls."""

import random
import statistics
from dataclasses import replace

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.native_source_chunks import new_workspace
from mosaickv.evaluation.sequential_kv_lease import fill_component
from mosaickv.evaluation.source_projection_audit import comparison

VARIANTS = ("fused_h1", "fused_h2", "fused_h4", "native_bmm_256")


@torch.inference_mode()
def run(packet, attention):
    n = packet.shape[2]
    target = torch.full(
        (*packet.shape[:2], n + 128, 128),
        -71,
        device=packet.source.codes.device,
        dtype=torch.bfloat16,
    )
    reference = packet.materialize(attention)
    before = tuple(t.clone() for t in packet.tensors())
    packets = {
        name: replace(packet, source_head_group=int(name[-1]))
        if name.startswith("fused")
        else replace(packet, projection_backend=name)
        for name in VARIANTS
    }
    workspace = new_workspace(packets["native_bmm_256"])
    numeric = {}
    unsupported = {}
    kernels = {}

    def execute(name):
        for component in (0, 1):
            fill_component(
                packets[name],
                attention,
                component,
                target,
                source_workspace=workspace if name.startswith("native") else None,
            )

    for name, fmt in packets.items():
        checks = []
        try:
            if name.startswith("fused"):
                from mosaickv.evaluation.shared_source_kernel import project

                kernels[name] = project(
                    fmt, attention, (target, target), component=0, inspect_kernel=True
                )
            for component in (0, 1):
                fill_component(
                    fmt,
                    attention,
                    component,
                    target,
                    source_workspace=workspace if name.startswith("native") else None,
                )
                value = target[..., :n, :]
                checks.append(
                    dict(
                        comparison(reference[component], value),
                        bitwise_equal=bool(
                            torch.equal(
                                value.view(torch.int16), reference[component].view(torch.int16)
                            )
                        ),
                        history_sentinel_unchanged=bool((target[..., n:, :] == -71).all()),
                    )
                )
            for _ in range(5):
                execute(name)
            torch.cuda.synchronize()
        except Exception as exc:
            torch.cuda.synchronize()  # poisoned CUDA contexts still fail the worker
            unsupported[name] = f"{type(exc).__name__}: {exc}"
            checks = [
                dict(bitwise_equal=False, history_sentinel_unchanged=False, error=unsupported[name])
            ]
        numeric[name] = checks
    torch.cuda.synchronize()
    generator = random.Random(20261101)
    trials = []
    for repetition in range(20):
        order = [name for name in VARIANTS if name not in unsupported]
        generator.shuffle(order)
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            execute(name)
            end.record()
            trials.append((repetition, name, start, end))
    torch.cuda.synchronize()
    records = [
        dict(repetition=i, variant=name, cuda_ms=start.elapsed_time(end))
        for i, name, start, end in trials
    ]
    if not all(torch.equal(a, b) for a, b in zip(before, packet.tensors(), strict=True)):
        raise RuntimeError("operator benchmark mutated encoded source or exact K/V")
    return dict(
        numerical=numeric,
        kernel_metadata=kernels,
        unsupported=unsupported,
        trials=records,
        medians_ms={
            name: statistics.median(r["cuda_ms"] for r in records if r["variant"] == name)
            for name in VARIANTS
            if name not in unsupported
        },
        p95_ms={
            name: sorted(r["cuda_ms"] for r in records if r["variant"] == name)[18]
            for name in VARIANTS
            if name not in unsupported
        },
        charged_operator_bytes={
            name: backing_bytes(
                (*packet.tensors(), target, *((workspace,) if name.startswith("native") else ()))
            )
            for name in VARIANTS
        },
        inactive_reference_bytes=backing_bytes(reference),
        inactive_immutability_control_bytes=backing_bytes(before),
        immutable_source_verified=True,
        inactive_workspace_bytes_when_fused=backing_bytes((workspace,)),
        excludes_weights_attention_and_internal_blas_workspace=True,
        all_supported_methods_have_five_warmups_twenty_randomized_trials=True,
        p95_definition="nearest_rank",
        no_model_outputs_or_annotations=True,
    )


def select_group(cases):
    names = VARIANTS
    available = {
        name: bool(cases) and all(name in c["microbenchmark"]["medians_ms"] for c in cases)
        for name in names
    }
    sums = {
        name: sum(c["microbenchmark"]["medians_ms"][name] for c in cases)
        if available[name]
        else None
        for name in names
    }
    bitwise = {
        name: bool(cases)
        and all(
            r["bitwise_equal"] and r["history_sentinel_unchanged"]
            for c in cases
            for r in c["microbenchmark"]["numerical"][name]
        )
        for name in names
    }
    eligible = [name for name in names[:3] if available[name] and bitwise[name]]
    winner = (
        min(eligible, key=lambda name: (sums[name], name))
        if eligible and available["fused_h1"]
        else None
    )
    if winner is not None and sums[winner] > 0.9 * sums["fused_h1"]:
        winner = "fused_h1" if bitwise["fused_h1"] else None
    return dict(
        summed_layer_medians_ms=sums,
        all_components_bitwise=bitwise,
        chosen_grouped_operator=winner,
        no_native_or_model_promotion=True,
    )
