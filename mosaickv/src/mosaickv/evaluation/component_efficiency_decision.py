"""Derived component decision from complete consumed engineering evidence."""

import math


def build(admission, uniform, facility, gray):
    if any(
        p.get("complete") is not True or p.get("errors")
        for p in (admission, uniform, facility, gray)
    ):
        raise ValueError("complete failure-preserving source and model analyses required")
    if set(admission["results"]) != {"0", "1"}:
        raise ValueError("both registered consumed timing inputs required")
    comparisons, timing = [], []
    for i, r in admission["results"].items():
        if r["all_logits_bitwise"] is not True or r["all_greedy_parent_agreement"] is not True:
            raise ValueError("actual complete numerical controls required")
        for boundary in (
            "initial_encoded_bytes",
            "final_encoded_bytes",
            "reported_counter_peak_bytes",
        ):
            first = r["storage"]["common_storage_wrapped"][boundary]
            second = r["storage"]["common_storage_hook_free"][boundary]
            if first != second or len(first) != 1 or type(first[0]) is not int or first[0] <= 0:
                raise ValueError("physical cache boundaries must match between source-only arms")
        pair = r["comparisons"]["common_storage_hook_free_versus_retained_hybrid"]
        comparisons.append(dict(input=i, versus="retained_six_stage_hybrid", **pair))
        comparisons.append(
            dict(
                input=i,
                versus="same_codec_storage_only",
                **r["comparisons"]["common_storage_hook_free_versus_common_storage_wrapped"],
            )
        )
        for arm in ("common_storage_wrapped", "common_storage_hook_free", "retained_hybrid"):
            reference = r["comparisons"][
                "retained_hybrid_versus_common_storage_wrapped"
                if arm == "retained_hybrid"
                else "common_storage_hook_free_versus_common_storage_wrapped"
            ]
            seconds = reference["original" if arm == "common_storage_wrapped" else "inline"]["mean"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("measured finite positive request time required")
            timing.append(dict(input=i, arm=arm, seconds=seconds, **r["storage"][arm]))
    codecs = []
    for name, result in (
        ("uniform_source_dictionary", uniform),
        ("facility_source_dictionary", facility),
        ("gray384_source_codec", gray),
    ):
        if set(result["results"]) != {"0", "1"}:
            raise ValueError("both registered CPU input screens required")
        for i, r in result["results"].items():
            if r["original_code_bits_exact"] is not True:
                raise ValueError("every original source code must survive")
            fraction = r["savings_fraction"]
            if not math.isfinite(fraction) or fraction < 0:
                raise ValueError("finite nonnegative selected packet saving required")
            codecs.append(
                dict(
                    variant=name,
                    input=i,
                    saved_CPU_source_bytes=r["saved_bytes"],
                    CPU_source_savings_fraction=fraction,
                    source_gate_passed=result["CPU_storage_gate_passed"],
                    GPU_or_task_claim=False,
                )
            )
    return dict(
        source_only_component_elimination_gate=admission[
            "source_only_component_elimination_engineering_gate"
        ],
        measured_requests=timing,
        comparisons=comparisons,
        CPU_codec_screens=codecs,
        recommended_runtime_variant="explicit_storage_only_not_six_component_method",
        six_component_research_code_preserved=True,
        existing_validated_defaults_unchanged=True,
        new_benchmark_quality_or_all_baseline_superiority_established=False,
        whole_request_peak_certified=False,
        fresh_or_paper_campaign_authorized=False,
    )


def markdown(decision):
    lines = [
        "# Component efficiency review: implemented and verified",
        "",
        "The user permits reducing the active design when mechanisms add cost without a "
        "demonstrated useful effect. Preserve the six-component research code and historical "
        "results. The new hook-free source-only path is explicitly a reduced-component variant, "
        "not a successful all-six method.",
        "",
        "## GPU execution",
        "",
        "Two consumed COCO inputs; LLaVA-1.5-7B, pinned weights/processor, BF16 eager, "
        "128 predictions, same common source codec, six warmups and 24 balanced paired "
        "trials per input. Every trial matched all 128 appropriate frozen vocabulary vectors; "
        "all emitted greedy tokens matched the storage parent. This is engineering evidence, "
        "not independent benchmark quality. The parent's existing source quantization is "
        "unchanged: lossless coding does not imply original FullKV K/V are unquantized.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Decode-owned peak MiB | Mean request s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in decision["measured_requests"]:
        memory = [
            r[k][0] / 2**20
            for k in ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")
        ]
        lines.append(
            f"| {r['input']} | {r['arm']} | {memory[0]:.4f} | {memory[1]:.4f} | "
            f"{memory[2]:.4f} | {r['seconds']:.4f} |"
        )
    lines += [
        "",
        "| Input | Hook-free versus | Mean ratio | Adjusted upper ratio | Strict latency gate |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in decision["comparisons"]:
        lines.append(
            f"| {r['input']} | {r['versus']} | {r['paired_mean_ratio']:.6f} | "
            f"{r['adjusted_one_sided_upper_ratio']:.6f} | {r['paired_latency_engineering_gate']} |"
        )
    lines += [
        "",
        "Adjusted bounds use 10,000 paired trial bootstraps and six-comparison "
        "Bonferroni correction. Trial resampling on two fixed images is not an "
        "example-level quality interval. Equal-or-faster requires upper <=1.00; a "
        "4% win requires upper <=0.96. Do not round a 3.82% observation into 4%. "
        "Source-only cache boundaries match exactly. The hybrid's additional 6,713 GPU "
        "bytes and 1,024 pinned CPU bytes are eliminated, not a large compression gain. "
        "Whole-request KV peak remains uncertified.",
        "",
        "## Better component uses tested before omission",
        "",
        "| Candidate | Input | Saved CPU source-stream bytes | Source-stream saving | "
        "Registered storage gate |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in decision["CPU_codec_screens"]:
        lines.append(
            f"| {r['variant']} | {r['input']} | {r['saved_CPU_source_bytes']} | "
            f"{100 * r['CPU_source_savings_fraction']:.4f}% | {r['source_gate_passed']} |"
        )
    lines += [
        "",
        "These CPU captures are the consumed Description/MM-Vet diagnostics, not "
        "the two COCO timing images. Uniform dictionaries examined all 32 layers per "
        "input; the cost-aware facility screen examined fixed layers 0/8/16/24/31. "
        "Gray coding examined all 32 layers. Complete packet costs include selectors, "
        "offsets, anchors/maps when applicable, and payloads. Every original code bit "
        "roundtripped. The dictionary/graph/selector attempt admitted no net-positive "
        "replacement. Gray coding improved a few layers, but missed the registered "
        "2% code-stream threshold. No native GPU reader was built for failed formats. "
        "CPU packet savings are not active GPU KV or model-quality measurements.",
        "",
        "## Active design decision",
        "",
        "| Mechanism | Decision for the optimized storage variant | Reason |",
        "| --- | --- | --- |",
        "| Future-query forecasting | Off; preserve experimental implementation | "
        "Parent-token lock cannot create an answer gain; "
        "no independently supported marginal gain |",
        "| Cross-modal evidence graph | Off; preserve experimental implementation | "
        "Largest measured construction stage; no demonstrated gain and byte-aware dictionary "
        "use also failed |",
        "| Value-aware utility | Off; preserve experimental implementation | "
        "Earlier error/reconstruction improvements did not consistently improve answers |",
        "| Submodular selection | Off for additive precision overlay | "
        "No complete source owner is retired; cost-aware source selection correctly declined "
        "expensive anchors |",
        "| Exact/prototype/residual overlay | Off | Additional shared-source overlay state "
        "cannot reduce the parent's backing |",
        "| Uncertainty residual repair | Off | The small panel did not demonstrate "
        "benchmark-quality recovery; forced tests prove mechanics only |",
        "| Existing source reader and validated common codec | Retained | Preserve original "
        "positions, parent numerical computation and exact physical accounting |",
        "",
        "No research module was deleted or silently relabeled. Choosing "
        "`experimental_overlay` still executes the existing full implementation. "
        "`storage_preservation` requires an explicitly disabled overlay and refuses "
        "unverified source transforms. It installs no inactive hooks or private "
        "overlay repair/planning state. The parent's original prompt-attention protection, "
        "source precision and lossless numerical corrections remain unchanged. "
        "Removing inactive wrapper hooks alone did not prove "
        "a latency win over the existing storage-only path; the substantial measured "
        "saving is from not running the full adaptive overlay.",
        "",
        "No fresh calibration, larger quality sweep or paper campaign is justified "
        "by this review alone. A future adaptive design must demonstrate a useful "
        "complete source-owner replacement, with all consumers, residuals, history "
        "and temporary backing charged, before adding its runtime overhead.",
    ]
    if decision.get("earlier_stage_context"):
        lines += [
            "", "## Earlier stage context, not a new component-quality experiment", "",
            "These phase costs profile the retained hybrid on two consumed timing inputs. "
            "The CIDEr differences are from a separate, earlier unguarded 16-image panel; "
            "they are retrospective and not independent evidence for the current guarded "
            "variant. Construction time is not a causal request-latency delta.", "",
            "| Component | Profiled mean ms | Historical full-minus-ablation CIDEr |",
            "| --- | ---: | ---: |",
        ]
        for r in decision["earlier_stage_context"]:
            ms = r["mean_profiled_milliseconds"]
            if not math.isfinite(ms) or ms < 0:
                raise ValueError("finite measured stage times required")
            delta = r["historical_16_image_unguarded_ablation"]["full_minus_ablated_cider"]
            lines.append(f"| {r['stage']} | {ms:.3f} | {delta:+.5f} |")
        lines += [
            "", "Stage six times only the first-update policy check, excluding its key "
            "snapshot; no natural restoration happened in that panel. Its known temporary "
            "K backing was 5,947,904 bytes, not a certified whole-stage or whole-request peak. "
            "No observed CIDEr change in this small panel is not proof a mechanism can never "
            "help. It is insufficient reason to retain its overhead in this storage variant.",
        ]
    return "\n".join(lines) + "\n"
