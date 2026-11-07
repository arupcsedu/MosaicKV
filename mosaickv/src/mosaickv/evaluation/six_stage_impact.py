"""Separate profiled phase costs, matched-allocation ablations and KV bytes."""

import math
from statistics import mean

STAGES = (
    ("future_query_forecast", ("prompt_queries_and_forecast",), "no_forecast"),
    ("evidence_graph", ("sparse_graph",), "no_graph"),
    ("value_aware_utility", ("value_utility",), "no_value"),
    ("budgeted_selection", ("budgeted_selection",), "modular_selection"),
    (
        "exact_prototype_residual_tiers",
        ("prototype_and_residual_construction", "precision_transaction_cpu_and_buffer_updates"),
        "exact_members",
    ),
    ("uncertainty_repair", (), "no_repair"),
)


def build(profiles, panel, latency):
    if len(profiles) != 2 or panel.get("execution_passed") is not True:
        raise ValueError("two complete profiles and the completed consumed ablation required")
    if latency.get("complete") is not True:
        raise ValueError("complete paired unprofiled latency artifact required")
    phase_rows, overhead, repair_buffers, guard_buffers = [], [], [], []
    for p in profiles:
        if p.get("all_128_logits_bitwise") is not True or p.get("labels_accessed") is not False:
            raise ValueError("numerically valid, label-blind phase profiles required")
        evidence = p["mechanism_evidence"]
        times = evidence["layers"]["31"]["construction_phase_seconds"]
        if any(not math.isfinite(v) or v < 0 for v in times.values()):
            raise ValueError("finite nonnegative measured phase times required")
        if (
            type(evidence["extra_gpu_bytes"]) is not int
            or type(evidence["residual_cpu_bytes"]) is not int
        ):
            raise ValueError("measured overlay/residual storage required")
        phase_rows.append(times)
        overhead.append(
            dict(gpu_bytes=evidence["extra_gpu_bytes"], cpu_bytes=evidence["residual_cpu_bytes"])
        )
        inline = evidence.get("inline_first_step_repair", {})
        repair_buffers.append(
            dict(
                first_update_key_snapshot_GPU_bytes=inline.get("temporary_key_clone_backing_bytes"),
                first_update_residual_GPU_bytes=inline.get(
                    "temporary_attention_residual_backing_bytes"
                ),
                stage_peak_certified=False,
            )
        )
        tail = evidence.get("live_anchor_guard", {}).get("tail_graph", {})
        guard_buffers.append(
            dict(
                record_activation_bytes=tail.get("record_buffer_activation_bytes"),
                explicit_activation_backing_bytes=tail.get("explicit_activation_backing_bytes"),
                observed_graph_allocator_increment_bytes=tail.get(
                    "observed_graph_allocator_increment_bytes"
                ),
                not_KV_bytes=True,
            )
        )
    rows = []
    for name, phases, ablation in STAGES:
        effect = panel["effects"][ablation]
        values = [sum(p[k] for k in phases) for p in phase_rows] if phases else []
        rows.append(
            dict(
                stage=name,
                profiled_seconds=values or None,
                mean_profiled_milliseconds=1000 * mean(values) if values else None,
                timing_is_causal_ablation_delta=False,
                separately_audited_stage_KV_bytes=None,
                whole_overlay_storage=overhead
                if name == "exact_prototype_residual_tiers"
                else None,
                known_first_step_repair_buffers=repair_buffers
                if name == "uncertainty_repair"
                else None,
                historical_16_image_unguarded_ablation=dict(
                    compared_arm=ablation,
                    full_minus_ablated_cider=effect["full_minus_arm_cider"],
                    interval=effect["retrospective_unadjusted_95_interval"],
                    changed_captions=effect["full_minus_arm_caption_changes"],
                    same_version_as_latency_run=False,
                    independent_quality_evidence=False,
                ),
            )
        )
    return dict(
        measurement_type="derived_consumed_engineering_profiles_and_historical_ablation",
        result_category="unified_comparison",
        stages=rows,
        routing_descriptor_mean_ms=1000 * mean(p["route_and_descriptors"] for p in phase_rows),
        natural_repair_events_in_historical_panel=panel["natural_repair_events"],
        latest_paired_result=latency,
        stage_memory_not_separately_audited=True,
        per_token_guard_not_a_six_component_construction_phase=True,
        per_input_guard_activation_buffers=guard_buffers,
        profile_costs_not_valid_timed_trial_latencies=True,
        historical_ablation_not_a_gate_for_current_guarded_variant=True,
        whole_request_peak_certified=False,
        superiority_established=False,
    )


def markdown(audit):
    lines = [
        "# Six-stage measured impact and limits",
        "",
        "Two consumed-input synchronized/cProfile construction costs are not causal latency "
        "deltas. Quality deltas below come from a different, earlier unguarded 16-image "
        "ablation. They must not be presented as effects of the latest guarded runtime.",
        "",
        "| Stage | Profiled mean ms | Separately audited stage KV | "
        "Historical full-minus-ablation CIDEr |",
        "| --- | ---: | --- | ---: |",
    ]
    for r in audit["stages"]:
        ms = r["mean_profiled_milliseconds"]
        memory = "not isolated"
        if r["whole_overlay_storage"]:
            entries = r["whole_overlay_storage"]
            memory = "; ".join(
                f"input {i}: +{v['gpu_bytes']} GPU / +{v['cpu_bytes']} CPU bytes"
                for i, v in enumerate(entries)
            )
        if r.get("known_first_step_repair_buffers"):
            known = [
                v["first_update_key_snapshot_GPU_bytes"]
                for v in r["known_first_step_repair_buffers"]
            ]
            if all(type(v) is int for v in known):
                memory = f"known key snapshots {known} GPU bytes; not stage peak"
        lines.append(
            f"| {r['stage']} | {ms:.3f} | {memory} | "
            f"{r['historical_16_image_unguarded_ablation']['full_minus_ablated_cider']:+.5f} |"
            if ms is not None
            else f"| {r['stage']} | not isolated | {memory} | "
            f"{r['historical_16_image_unguarded_ablation']['full_minus_ablated_cider']:+.5f} |"
        )
    lines += [
        "",
        f"Routing/descriptors add {audit['routing_descriptor_mean_ms']:.3f} profiled ms. "
        "The per-token live-parent guard is additional to these six construction stages. "
        "Temporary CPU graph/descriptor/score peaks were not separately audited.",
        "The repair key snapshot is a known temporary buffer, not the complete repair-stage "
        "or whole-request peak. Guard-record buffers are activations, not KV; their explicit "
        "backing and observed allocator increments remain separately recorded in audit.json.",
        "",
        "Negative full-minus-ablation CIDEr means the full arm scored lower. "
        "Natural repair did not occur in the historical panel; forced restoration tested "
        "mechanics, not quality recovery. Matched reserved allocations do not prove "
        "individual mechanisms save KV bytes.",
        "",
        "# Latest unprofiled paired request measurements",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Decode-owned peak MiB | Mean request s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, result in audit["latest_paired_result"]["results"].items():
        for arm, s in result["storage"].items():
            values = [
                max(s[k]) / 2**20
                for k in (
                    "initial_encoded_bytes",
                    "final_encoded_bytes",
                    "reported_counter_peak_bytes",
                )
            ]
            lines.append(
                f"| {index} | {arm} | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {s['mean_seconds']:.4f} |"
            )
    lines += [
        "",
        "Common source coding must also be applied to storage-only. The overlay is additive; "
        "these measurements do not establish an all-six quality, storage or latency advantage. "
        "Peak is decode-owned, not a whole-request GPU/KV peak certificate.",
        "",
        "All source paths and hashes are in audit.json; original artifacts are unchanged.",
    ]
    return "\n".join(lines) + "\n"
