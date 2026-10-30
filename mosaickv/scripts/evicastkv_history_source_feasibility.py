#!/usr/bin/env python3
"""Read-only generated-history source feasibility on two consumed inputs."""

from pathlib import Path

import evicastkv_encoded_baselines as encoded

from mosaickv.evaluation.fox_campaign import write_once

base = encoded.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/history-source-packets-consumed-20261030-v1")
original_analyze = base.analyze


def prepare(root, phase):
    writer = encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            value.update(
                protocol="history_source_capture_and_exact_packet_feasibility_v1",
                no_selector_change=True,
                source_capture_horizon=128,
                history_packet_control_updates=[31, 127],
                original_model_outputs_must_be_bitwise=True,
                normalized_source_tail_additional_bytes_charged=True,
                packet_build_budget_seconds_per_method_and_horizon=300,
                prospective_fraction_gate_128=0.10,
                packet_feasibility_is_not_online_memory_measurement=True,
                capture_timing_not_a_latency_benchmark=True,
                online_runtime_integration_authorized=False,
                no_fresh_or_large_campaign=True,
            )
        writer(path, value)

    encoded.history.original_writer = register
    try:
        return encoded.prepare(root, phase)
    finally:
        encoded.history.original_writer = writer


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import torch

    from mosaickv.evaluation.history_source_feasibility import (
        HistoryProjectionCapture,
        packet_feasibility,
    )
    from mosaickv.evaluation.storage_runs import run

    pending, references = {}, {"full_kv": full_reference}
    current_method = ""
    metrics, diagnostics = [], []
    old_choices, old_timing = base.choices, base.timing

    def choices(*args, **kwargs):
        variants, proof = encoded.fused_choices(*args, **kwargs)
        pending[current_method] = next(iter(variants.values()))
        return variants, proof

    def no_timing(_adapter, _prefill, variants, reference, method, _plan):
        pending[method] = next(iter(variants.values()))
        references[method] = reference
        return []

    base.choices, base.timing = choices, no_timing
    try:
        for method in encoded.history.paired.METHODS:
            current_method = method
            folder = out / method
            folder.mkdir()
            metrics.extend(
                encoded.history.paired.original_study(
                    adapter,
                    prefill,
                    capture,
                    frozen,
                    dict(job, method=method),
                    cfg,
                    full_reference,
                    prepared,
                    folder,
                    plan,
                )
            )
            formats = pending[method]
            reference, reference_record = run(
                adapter, prefill, formats, tokens=128, audit=True, batched_history=True
            )
            with HistoryProjectionCapture(adapter) as collector:
                observed, observed_record = run(
                    adapter, prefill, formats, tokens=128, audit=True, batched_history=True
                )
            if not torch.equal(reference, observed) or not torch.equal(
                reference[:32], references[method]
            ):
                raise RuntimeError("read-only history capture changed original model output")
            records = collector.assemble(127)
            torch.save(
                {
                    layer: {name: tensor.cpu() for name, tensor in row.items()}
                    for layer, row in records.items()
                },
                folder / "history_source_captures.pt",
            )
            torch.save(dict(reference=reference, observed=observed), folder / "history_logits.pt")
            packet_records = []
            for updates in plan["history_packet_control_updates"]:
                packet_records.append(
                    packet_feasibility(
                        adapter,
                        records,
                        updates=updates,
                        maximum_seconds=plan["packet_build_budget_seconds_per_method_and_horizon"],
                    )
                )
            row = dict(
                method=method,
                bitwise_128_capture_control=True,
                prefix_32_bitwise_original_reference=True,
                reference_transport=reference_record,
                capture_transport=observed_record,
                capture_workspace_excluded_from_online_storage_claim=True,
                capture_timing_not_latency_benchmark=True,
                packets=packet_records,
            )
            write_once(folder / "history_source_feasibility.json", row)
            diagnostics.append(row)
            del collector, records
    finally:
        base.choices, base.timing = old_choices, old_timing
    return [
        dict(
            baseline_metrics=metrics,
            history_packet_diagnostics=diagnostics,
            no_online_saving_or_quality_superiority_claim=True,
        )
    ]


def analyze(root, expected):
    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"]:
        return
    rows = [
        dict(task=worker["task"], method=method["method"], **packet)
        for worker in gate["controls"]
        for method in worker["history_packet_diagnostics"]
        for packet in method["packets"]
    ]
    controls = [row for row in rows if row["updates"] == 127]
    if len(rows) != 24 or len(controls) != 12:
        raise ValueError("two complete inputs, six methods, two horizons required")
    write_once(
        root / "analysis/history_source_packet_decision.json",
        dict(
            rows=rows,
            prospective_feasible=all(
                row["prospective_packet_saving_fraction"] >= 0.10 for row in controls
            ),
            no_online_runtime_or_request_peak_claim=True,
            no_selection_or_independent_quality_claim=True,
            no_automatic_integration_or_new_campaign=True,
        ),
    )


if __name__ == "__main__":
    encoded.history.paired.install()
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_history_source_feasibility.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    base.main()
