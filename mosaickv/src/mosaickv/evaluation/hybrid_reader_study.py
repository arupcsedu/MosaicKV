"""Bounded operator cost table, no labels and no online model probes."""

import time
from dataclasses import asdict
from itertools import product

import torch

from mosaickv.evaluation.hybrid_reader import MODES, ReaderOption, replace_packet
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_kv_lease import fill_component


def timed(function, *, warmups=5, repetitions=20):
    if warmups != 5 or repetitions != 20:
        raise ValueError("registered five warmups and twenty repetitions required")
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    observations = []
    for _ in range(repetitions):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        host = time.perf_counter_ns()
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        observations.append(
            dict(wall_ns=time.perf_counter_ns() - host, cuda_ms=start.elapsed_time(end))
        )
    return observations


@torch.inference_mode()
def profile(packets, attentions, *, seed):
    """Operator evidence only; actual request benchmarking includes conversion.

    Three real strided layouts, randomized option order, no full-cache tensor
    retained beyond one layer. No online use of these timing observations.
    """
    import random
    from collections import Counter

    rng, routes, raw = random.Random(seed), [], []

    def identity(t):
        return (str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())

    owners = Counter(k for p in packets for k in {identity(t) for t in p.private()})
    shared = {k for k, count in owners.items() if count > 1}

    def cost(packet):
        return backing_bytes(tuple(t for t in packet.private() if identity(t) not in shared))

    for index, (packet, attention) in enumerate(zip(packets, attentions, strict=True)):
        modes = list(product(MODES, repeat=2))
        rng.shuffle(modes)
        options = []
        for mode in modes:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            candidate = replace_packet(packet, attention, mode)
            torch.cuda.synchronize()
            construction_ns = (
                time.perf_counter_ns() - started if mode != ("source", "source") else 0
            )
            observations = []
            for padding in (0, 64, 127):
                shape = (*packet.shape[:2], packet.shape[-2] + padding, packet.shape[-1])
                target = torch.full(
                    shape, -71, device=packet.private()[0].device, dtype=torch.bfloat16
                )
                reference = torch.empty_like(target)
                for component in (0, 1):
                    fill_component(packet, attention, component, reference)
                    fill_component(candidate, attention, component, target)
                    if not torch.equal(
                        reference[..., : packet.shape[-2], :].view(torch.int16),
                        target[..., : packet.shape[-2], :].view(torch.int16),
                    ) or not bool((target[..., packet.shape[-2] :, :] == -71).all()):
                        raise RuntimeError("hybrid operator changed K/V bits or history sentinels")

                def call(candidate=candidate, attention=attention, target=target):
                    for component in (0, 1):
                        fill_component(candidate, attention, component, target)

                observations.extend(dict(padding=padding, **r) for r in timed(call))
                del target, reference
            option = ReaderOption(
                mode,
                cost(candidate),
                construction_ns,
                int(sum(r["wall_ns"] for r in observations) / len(observations)),
            )
            options.append(option)
            raw.append(
                dict(
                    layer=index,
                    option=asdict(option),
                    trials=observations,
                    original_KV_bitwise=True,
                    history_sentinels_unchanged=True,
                )
            )
            del candidate
        routes.append(sorted(options, key=lambda o: o.modes))
    return dict(
        routes=[[asdict(o) for o in r] for r in routes],
        raw=raw,
        base_private_bytes=backing_bytes(tuple(t for p in packets for t in p.private())),
        cross_layer_aliases_reserved_once_not_credited_as_freed=True,
        timings_are_operator_evidence_not_request_benchmarks=True,
        construction_times_are_single_observations_not_confidence_intervals=True,
        no_labels_or_answers_accessed=True,
        independent_quality_N=0,
    )


def plans(cost_table, *, allowances=(0, 6 * 2**20)):
    from mosaickv.evaluation.hybrid_reader import choose

    routes = [
        [
            ReaderOption(tuple(x["modes"]), x["private_bytes"], x["construction_ns"], x["read_ns"])
            for x in route
        ]
        for route in cost_table["routes"]
    ]
    baseline = cost_table["base_private_bytes"]
    exclusive_baseline = sum(
        next(o for o in r if o.modes == ("source", "source")).private_bytes for r in routes
    )
    results = []
    for allowance in allowances:
        choice = choose(routes, private_cap_bytes=exclusive_baseline + allowance, decode_steps=127)
        policy = dict(
            modes=choice["modes"], base_private_bytes=baseline, maximum_added_bytes=allowance
        )
        source_time = (
            sum(next(o for o in r if o.modes == ("source", "source")).read_ns for r in routes) * 127
        )
        results.append(
            dict(
                policy=policy,
                choice=choice,
                source_predicted_ns=source_time,
                predicted_improvement_ns=source_time - choice["predicted_ns"],
                operator_prediction_is_not_measured_request_speedup=True,
            )
        )
    return results
