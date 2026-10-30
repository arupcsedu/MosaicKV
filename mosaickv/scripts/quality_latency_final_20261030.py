"""Hash-verified consumed-diagnostic report, with failures and no paper promotion."""
# ruff: noqa: E501

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/scratch/djy8hg/runs/evicastkv")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root, inputs):
    plan = read(root / "plan.json")
    gate = read(root / "analysis/GATE.json")
    for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
        inputs[str(root / name)] = sha(root / name)
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        result_path = folder / "result.json"
        if not result_path.exists():
            continue
        result = read(result_path)
        inputs[str(result_path)] = sha(result_path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
            root / "plan.json"
        ):
            raise ValueError("producer provenance mismatch")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("producer evidence changed")
        if (folder / "nested_evidence.json").exists():
            for name, digest in read(folder / "nested_evidence.json").items():
                if sha(folder / name) != digest:
                    raise ValueError("nested producer evidence changed")
    return plan, gate


def main(out, history_root=None):
    if out.exists():
        raise FileExistsError("new derived report directory required")
    inputs = {}
    history = history_root or RUNS / "batched-history-consumed-20261030-v1"
    plan, gate = verify(history, inputs)
    if gate["status"] != "complete_consumed_engineering_diagnostic" or gate["failures"]:
        raise ValueError("complete history controls required for this report")
    rows, trials, ratios = [], [], []
    for record in gate["controls"]:
        task, sample = record["task"], record["sample_id"]
        metrics = {r["method"]: r for r in record["baseline_metrics"]}
        for control in record["history_controls"]:
            method = control["method"]
            metric = metrics[method]["controls"][0]
            storage = control["batched"]["storage"]
            rows.append(
                dict(
                    task=task,
                    sample_id=sample,
                    method=method,
                    horizon=control["horizon"],
                    initial_bytes=storage["initial_active_kv_storage_bytes"],
                    final_bytes=storage["final_active_kv_storage_bytes"],
                    observed_decode_owned_peak_bytes=storage["peak_active_kv_storage_bytes"],
                    bitwise_own_logits=control["bitwise_logits"],
                    observed_peak_not_increased=control["observed_peak_not_increased"],
                    rouge_l_32_model_reference=metric["fullkv_reference_rouge_l_f1"],
                    token_agreement_32_model_reference=metric["own_greedy_token_agreement"],
                    kl_32_forced_fullkv_prefix=metric["fullkv_prefix_mean_kl"],
                    not_annotation_quality=True,
                    source_sha=plan["source_sha"],
                    method_variant=(
                        "all_position_lossless_transport_control" if method == "full_kv" else method
                    ),
                    source_gate=str(history / "analysis/GATE.json"),
                    whole_request_peak_certified=False,
                )
            )
        trials.extend(dict(task=task, sample_id=sample, **r) for r in record["trials"])
        timed = [r for r in record["trials"] if not r["warmup"] and r["history"] == "batched"]
        evi = {
            r["trial"]: r["decode_wall_seconds"] for r in timed if r["method"] == "mosaickv_exact"
        }
        for method in METHODS[1:]:
            selected = [r for r in timed if r["method"] == method]
            if len(selected) != 20 or len(evi) != 20:
                raise ValueError("complete twenty paired repetitions required")
            samples = np.array([evi[r["trial"]] / r["decode_wall_seconds"] for r in selected])
            boot = np.median(
                samples[np.random.default_rng(20261030).integers(0, 20, size=(10000, 20))], axis=1
            )
            ratios.append(
                dict(
                    task=task,
                    method=method,
                    decode_p50_seconds=float(
                        np.median([r["decode_wall_seconds"] for r in selected])
                    ),
                    decode_p95_seconds=float(
                        np.percentile([r["decode_wall_seconds"] for r in selected], 95)
                    ),
                    evicast_paired_ratio=float(np.median(samples)),
                    ratio_ci_lower=float(np.percentile(boot, 2.5)),
                    ratio_ci_upper=float(np.percentile(boot, 97.5)),
                )
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 24 or frame.duplicated(["task", "sample_id", "method", "horizon"]).any():
        raise ValueError("all six methods at both horizons on two inputs required")
    import torch

    frozen_checks = []
    for job in plan["jobs"]:
        source = Path(job["delimiter_worker"]) / "captures.pt"
        if plan["bound_files"].get(str(source)) != sha(source):
            raise ValueError("frozen membership capture changed")
        inputs[str(source)] = sha(source)
        frozen = torch.load(source, map_location="cpu", weights_only=True)
        record = next(r for r in gate["controls"] if r["sample_id"] == job["sample_id"])
        logical = record["history_controls"][0]["original"]["original_logical_length"]
        budgets = []
        for role in ("window_topk_control", "uniform_0.5_control"):
            positions = frozen[role]["anchor"]["positions"]
            if len(positions) != 32 or any(len(heads) != 32 for heads in positions):
                raise ValueError("audited full-MHA geometry differs")
            if any(
                tuple(sorted(set(head))) != tuple(head) or head[-1] != logical - 1
                for heads in positions
                for head in heads
            ):
                raise ValueError(
                    "frozen selector dropped mandatory last token or changed positions"
                )
            raw_bytes = sum(len(head) for heads in positions for head in heads) * 128 * 2 * 2
            budgets.append(raw_bytes)
            frozen_checks.append(
                dict(
                    task=job["task"],
                    sample_id=job["sample_id"],
                    role=role,
                    original_logical_length=logical,
                    raw_selected_KV_bytes=raw_bytes,
                    mandatory_last_token_in_all_heads=True,
                    monotonic_unique_logical_positions=True,
                )
            )
        if budgets != [156008448, 156008448]:
            raise ValueError("original source-slot capacities do not match")
    constructor = RUNS / "direct-constructor-consumed-20261030-v1"
    _, constructor_gate = verify(constructor, inputs)
    ctor_summary = []
    if not constructor_gate["failures"]:
        summary_path = constructor / "constructor_analysis/summary.json"
        inputs[str(summary_path)] = sha(summary_path)
        ctor_summary = read(summary_path)["summaries"]
    dual = RUNS / "dual-projection-consumed-20261030-v1-retry3"
    _, dual_gate = verify(dual / "compact", inputs)
    _, dual_timing = verify(dual / "timing", inputs)
    if dual_gate["failures"] or dual_timing["failures"]:
        raise ValueError("dual controls missing")
    cpu = RUNS / "quality-latency-summary-20261030-v1/cpu-storage-regression.xml"
    inputs[str(cpu)] = sha(cpu)
    suites = ET.parse(cpu).getroot().findall("testsuite")
    cpu_tests = sum(int(s.attrib["tests"]) for s in suites)
    if not suites or any(int(s.attrib.get(k, "0")) for s in suites for k in ("errors", "failures")):
        raise ValueError("CPU regression did not pass")
    frozen_audit = RUNS / "quality-latency-summary-20261030-v1/frozen_membership_audit.json"
    if not read(frozen_audit)["independent_source_capacity_check_passed"]:
        raise ValueError("source capacity audit failed")
    inputs[str(frozen_audit)] = sha(frozen_audit)
    for root in (
        RUNS / "quality-latency-consumed-20261030-v1/evidence",
        RUNS / "quality-latency-compact-direct-20261030-v2/layer_addition",
    ):
        _, negative = verify(root, inputs)
        if negative["accepted_consumed_diagnostics"]:
            raise ValueError("negative-selection seal differs from the inspected evidence")
    out.mkdir()
    (out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "vector_charts_source.py").write_bytes(
        Path(__file__).with_name("storage_vector_charts_20261030.py").read_bytes()
    )
    (out / "storage_vector_charts_20261030.py").write_bytes(
        Path(__file__).with_name("storage_vector_charts_20261030.py").read_bytes()
    )
    frame.to_parquet(out / "storage_quality_controls.parquet", index=False)
    pd.DataFrame(trials).to_parquet(out / "paired_decode_trials.parquet", index=False)
    pd.DataFrame(ratios).to_parquet(out / "paired_decode_summary.parquet", index=False)
    frame.to_csv(out / "storage_quality_controls.csv", index=False)
    (out / "frozen_membership_and_mandatory_audit.json").write_text(
        json.dumps(frozen_checks, indent=2)
    )
    text = [
        "# EviCastKV quality, storage and latency engineering report",
        "",
        "## Scope and outcome",
        "",
        "The lossless transport work succeeded: selected/merged K/V and model outputs are preserved, "
        "storage remains reduced, fused projection accelerates decoding, and the common history codec "
        "has a small additional benefit. The attempted quality-selection interventions failed the "
        "unchanged guard. **No general superiority, independent quality preservation, or paper-ready "
        "30%-smaller claim is established.** No fresh cohort, calibration or large campaign was launched.",
        "",
        "All results below use exactly two consumed LLaVA-1.5-7B inputs (Description 000000532947; "
        "paper-style MM-Vet v1_197), pinned revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9, "
        "BF16 HF eager and deterministic greedy decoding. Each GPU worker uses one A100 and 12 CPUs. "
        "Every compressed selector starts with the same **148.78125 MiB original source-slot capacity**. "
        "Lossless encoded bytes differ; these are not exact-byte-matched selection-policy curves. "
        "FullKV is a quality reference and is excluded from compressed-method latency.",
        "",
        "Important naming distinction: raw `full_kv` rows in this storage diagnostic retain every "
        "position but also use the common lossless transport. Their memory values are shown as "
        "**all-position lossless controls**, not ordinary uncompressed FullKV memory. Ordinary "
        "FullKV remains the untouched model-output/parity reference. Do not use the control's "
        "encoded bytes as an uncompressed-FullKV denominator.",
        "",
        "EviCastKV uses the frozen `uniform_0.5_control` membership; Top-K uses "
        "`window_topk_control`. Selection is not recomputed in these storage experiments. "
        "The normal CLI defaults and paper candidates are not changed by these diagnostics.",
        "",
        "ROUGE-L/token agreement refer only to the model's 32-token FullKV output. They are not "
        "annotation scores, native PPL, or the large paper-native ROUGE evaluation. PrefixKV/VL-Cache "
        "rows are append-only, byte-rescaled local adaptations, not official results. LOOK-M keeps "
        "its actual merged K/V and uses the common lossless KV codec; original-source regeneration "
        "without replaying its merge would be invalid. A compound merge-replay codec is not "
        "implemented or evaluated, so this study does not prove fully optimized LOOK-M storage.",
        "",
        "## Latest common-transport comparison (32 tokens)",
        "",
        "| Input | Method | Initial MiB | Final MiB | Observed decode-owned peak MiB | Model-reference ROUGE-L | Token agreement | Decode p50 / p95 s | Evi/method paired ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lookup = {(r["task"], r["method"]): r for r in ratios}
    for r in frame[frame.horizon == 32].sort_values(["task", "method"]).to_dict("records"):
        s = lookup.get((r["task"], r["method"]))
        timing = (
            "quality reference"
            if s is None
            else f"{s['decode_p50_seconds']:.3f} / {s['decode_p95_seconds']:.3f}"
        )
        ratio = (
            "not compared"
            if s is None
            else f"{s['evicast_paired_ratio']:.3f} [{s['ratio_ci_lower']:.3f}, {s['ratio_ci_upper']:.3f}]"
        )
        label = "All-position lossless control" if r["method"] == "full_kv" else r["method"]
        text.append(
            f"| {r['task']} | {label} | {r['initial_bytes'] / 2**20:.3f} | {r['final_bytes'] / 2**20:.3f} | {r['observed_decode_owned_peak_bytes'] / 2**20:.3f} | {r['rouge_l_32_model_reference']:.3f} | {r['token_agreement_32_model_reference']:.3f} | {timing} | {ratio} |"
        )
    text.extend(
        [
            "",
            "A ratio below one means EviCastKV decoded faster in these conditional trials; "
            "a ratio above one means slower. Intervals bootstrap the twenty paired timing blocks, not "
            "benchmark examples. Neither two inputs nor their timing intervals establish general quality.",
            "The 32-token schedule uses the prefill's first prediction and 31 decode forwards; "
            "prefill is excluded from this timing boundary.",
            "",
            "PrefixKV is smaller than EviCastKV on these inputs; on Description their model-reference "
            "quality ties. On MM-Vet, EviCastKV better preserves the reference than PrefixKV/VL-Cache/LOOK-M, "
            "but this is one consumed case. Negative outcomes and faster/smaller competitors are retained.",
            "",
            "## 128-token append-only storage controls",
            "",
            "All six methods passed bitwise output equality against their unbatched-history transport at "
            "32 and 128 tokens, including independently enumerated backing storage. Initial bytes are "
            "identical under both history policies; final and observed decode-owned peak do not increase. "
            "No 128-token timing claim is made.",
            "",
            "| Input | Method | Initial MiB | Final MiB (128) | Observed decode-owned peak MiB (128) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for r in frame[frame.horizon == 128].sort_values(["task", "method"]).to_dict("records"):
        label = "All-position lossless control" if r["method"] == "full_kv" else r["method"]
        text.append(
            f"| {r['task']} | {label} | {r['initial_bytes'] / 2**20:.3f} | {r['final_bytes'] / 2**20:.3f} | {r['observed_decode_owned_peak_bytes'] / 2**20:.3f} |"
        )
    text.extend(
        [
            "",
            "## Why latency improved",
            "",
            "1. Compact replay disables reduced-precision reduction only for source reconstruction, "
            "restoring the original global policy afterward. Exact original-bit checks remain mandatory.",
            "2. Direct gather/RoPE and scatter avoid complete selected-K/V intermediate tensors.",
            "3. One dual-projection kernel reuses source input for K and V and writes directly into a "
            "layer lease. Narrow maps and exact sparse corrections count toward storage. Dynamic decode "
            "strides avoid repeated JIT compilation. Retention-one and all original-output controls pass.",
            "4. History tails are sealed in bounded four-layer groups, reducing allocation-size "
            "synchronizations from 64 to eight per 32-layer seal. Shared literal pools preserve every "
            "BF16 pattern. This provides only a small 32-token gain; it is not a major acceleration.",
            "",
            "The independently paired dual-kernel study reduced EviCastKV decode from about 2.12 s "
            "to 1.85-1.87 s (roughly 12%) and reduced its initial backing from about 61.6 MiB to "
            "60.9-61.0 MiB. It remains slower than storing encoded K/V. The table above applies the same "
            "eligible optimized transport to every selector, avoiding a one-sided implementation advantage.",
            "",
            "## Constructor differential study",
            "",
        ]
    )
    if ctor_summary:
        text.extend(
            [
                "The direct builder retains all three independent original-bit checks. It avoids "
                "redundant full-shape reconstructions and skips a second reduction policy only when zero "
                "correction bytes is already a proven lower bound. Every controlled source method has "
                "identical original logits and initial/final/observed-peak storage.",
                "",
                "| Input | Method | Reference constructor p50 s | Direct constructor p50 / p95 s | Paired ratio |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in ctor_summary:
            text.append(
                f"| {r['task']} | {r['method']} | {r['reference_p50_seconds']:.3f} | {r['direct_p50_seconds']:.3f} / {r['direct_p95_seconds']:.3f} | {r['paired_median_ratio']:.3f} |"
            )
    else:
        text.append(
            "The constructor study failed; its failure records are preserved. No speedup is inferred."
        )
    text.extend(
        [
            "",
            "The constructor differential times the base dual-format builder. It does not "
            "include the later encoded-source checkpoint/patch construction, and it must not be added "
            "to a different run's decode median as a measured end-to-end result.",
            "",
        ]
    )
    encoded_root = RUNS / "encoded-projection-consumed-20261030-v1"
    _, encoded_controls = verify(encoded_root / "compact", inputs)
    _, encoded_times = verify(encoded_root / "timing", inputs)
    decision_path = encoded_root / "timing/analysis/encoded_source_decision.json"
    decision = read(decision_path)
    if (
        not decision["engineering_eligible"]
        or encoded_controls["failures"]
        or encoded_times["failures"]
    ):
        raise ValueError("encoded-source engineering control did not pass")
    inputs[str(decision_path)] = sha(decision_path)
    text.extend(
        [
            "## Direct-from-packed-source peak-memory control",
            "",
            "The exact encoded-source kernel avoids allocating the complete decoded normalized matrix. "
            "Checkpoints cost additional persistent bytes. The independent paired control passed its "
            "registered peak and <=1.05 latency-ratio gates without changing a K/V bit or output token. "
            "It reduces decode-owned peak, but **does not reduce initial bytes relative to the dual path**.",
            "",
            "| Input | Dual initial MiB | Packed-source initial MiB | Dual final MiB | Packed-source final MiB | Dual peak MiB | Packed-source peak MiB | Paired decode ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    time_lookup = {(r["task"], r["role"]): r for r in decision["timing"]}
    for record in decision["memory"]:
        if record["role"] != "uniform_0.5_control":
            continue
        a, b = record["dual"], record["encoded"]
        t = time_lookup[record["task"], record["role"]]
        values = [
            a["initial_active_kv_storage_bytes"],
            b["initial_active_kv_storage_bytes"],
            a["final_active_kv_storage_bytes"],
            b["final_active_kv_storage_bytes"],
            a["peak_active_kv_storage_bytes"],
            b["peak_active_kv_storage_bytes"],
        ]
        text.append(
            "| "
            + record["task"]
            + " | "
            + " | ".join(f"{v / 2**20:.3f}" for v in values)
            + f" | {t['paired_median_ratio']:.3f} |"
        )
    cpu_root = RUNS / "predictive-storage-consumed-20261030-v1-cpu-retry2"
    cpu_plan = read(cpu_root / "plan.json")
    cpu_gate = read(cpu_root / "codec_analysis/GATE.json")
    if cpu_gate["failures"] or len(cpu_gate["rows"]) != 12:
        raise ValueError("codec feasibility incomplete; cannot seal its outcome")
    if cpu_gate["eligible_for_separately_versioned_GPU_decoder"] or any(
        r["prospective_reduction"] != 0 for r in cpu_gate["rows"]
    ):
        raise ValueError("row-predictor outcome differs from the registered negative seal")
    for name in ("plan.json", "codec_analysis/GATE.json", "codec_analysis/manifest.json"):
        inputs[str(cpu_root / name)] = sha(cpu_root / name)
    for job in cpu_plan["jobs"]:
        folder = cpu_root / f"codec_workers/{job['index']}"
        result = read(folder / "result.json")
        if result["source_sha"] != cpu_plan["source_sha"] or result["plan_sha256"] != sha(
            cpu_root / "plan.json"
        ):
            raise ValueError("CPU codec provenance mismatch")
        inputs[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("CPU codec evidence changed")
    pd.DataFrame(cpu_gate["rows"]).to_parquet(
        out / "CPU_codec_packet_feasibility.parquet", index=False
    )
    text.extend(
        [
            "",
            "## Stopped CPU-only row-predictor direction",
            "",
            "The same stored tensors from all six methods received six fixed predictor/codec candidates: "
            "no prediction, XOR16 or ordered-delta16, each with the existing exponent codec or a sparse "
            "high-byte packet. All bits round-tripped, but none beat the existing packet after metadata. "
            "The prospective saving was zero on both inputs for every method. The preregistered three-percent "
            "gate failed, so no predictive GPU decoder or new cohort was submitted. This negative CPU "
            "storage feasibility is separate from the successful direct packed-source projection, which "
            "uses the unchanged existing codec. Metadata-loading failures were preserved and repaired "
            "with scoped weights-only enum allowlisting; GPU captures were reused, not rerun.",
            "",
        ]
    )
    text.extend(
        [
            "",
            "## Remaining limitations",
            "",
            "All latency boundaries exclude original prefill, selection, artifact I/O and the other "
            "separately timed phases. Medians from different jobs are not summed into a claimed measured "
            "end-to-end speedup. The observed cache-owned peak includes maps, bodies, correction buffers, "
            "rotary data, history, layer leases and observed tensor scratch. It excludes inactive original "
            "diagnostic prefill/captures, weights, attention activations, internal GEMM workspace and allocator "
            "reserve. **Whole-request peak is uncertified.** Common FullKV prefill can dominate lifetime "
            "peak, so reduced decode-owned storage cannot be relabeled a lower request peak.",
            "",
            "Bulk source-sharing additions and draft-ranked one-layer additions both remained negative. "
            "They are sealed, not promoted. The software/storage success does not repair the existing "
            "selection-quality deficit. Prototype merging, residual repair, other architectures and native "
            "serving backends receive no new validated claim. The isolated source worktrees preserve the "
            "user's dirty main checkout and paper files.",
            "",
            f"A {cpu_tests}-test CPU regression run passed (CPU job 22194695), in addition to the focused new "
            "constructor tests. All source, plan, raw-result and nested evidence hashes are recorded in "
            "manifest.json. No packages were installed and all caches remain under scratch.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(text))
    from storage_vector_charts_20261030 import draw

    draw(frame, ratios, out)
    inputs[str(Path(__file__).with_name("storage_vector_charts_20261030.py"))] = sha(
        Path(__file__).with_name("storage_vector_charts_20261030.py")
    )
    source_root = Path(__file__).resolve().parents[2]
    source_git_sha = "not_git_backed_snapshot"
    source_dirty = False
    if (source_root / ".git").exists():
        source_git_sha = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"], text=True
            ).strip()
        )
    manifest = dict(
        analysis_git_sha=source_git_sha,
        analysis_source_dirty=source_dirty,
        script_sha256=sha(Path(__file__)),
        inputs=inputs,
        outputs={p.name: sha(p) for p in out.iterdir()},
        consumed_input_count=2,
        exact_encoded_byte_matching=False,
        quality_improvement_established=False,
        independent_quality_established=False,
        whole_request_peak_certified=False,
        paper_campaign_authorized=False,
        constructor_failures=constructor_gate["failures"],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-root", type=Path)
    args = parser.parse_args()
    main(args.output, args.history_root)
