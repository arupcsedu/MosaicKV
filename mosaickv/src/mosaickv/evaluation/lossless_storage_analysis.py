"""Strict paired systems analysis of a shared representation, never a selector win."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mosaickv.evaluation.fox_campaign import sha, write_once


def summarize_trials(trials: list[Any]) -> list[Any]:
    import numpy as np

    from mosaickv.evaluation.lossless_storage_benchmark import VARIANTS

    groups = {}
    for encoded, fused in VARIANTS:
        group = sorted(
            (t for t in trials if t["encoded"] == encoded and t["fused"] == fused),
            key=lambda t: t["block"],
        )
        if (
            len(group) != 25
            or [t["block"] for t in group] != list(range(-5, 20))
            or any(
                t["warmup"] != (t["block"] < 0) or not t["bitwise_reference_parity"] for t in group
            )
            or any(
                not np.isfinite(t[k]) or t[k] <= 0
                for t in group
                for k in (
                    "construction_seconds",
                    "decode_wall_seconds",
                    "decode_cuda_ms",
                )
            )
        ):
            raise ValueError(
                "complete five-warmup/twenty-trial finite bitwise-paired records required"
            )
        if any(t["storage"] != group[0]["storage"] for t in group):
            raise ValueError("storage changed between repeated trials")
        groups[encoded, fused] = group
    for encoded in (False, True):
        if groups[encoded, False][0]["storage"] != groups[encoded, True][0]["storage"]:
            raise ValueError("fusion changed physical storage boundary")
    rows = []
    rng = np.random.default_rng(20261029)
    indices = rng.integers(0, 20, size=(10000, 20))
    for variant, group in groups.items():
        encoded, fused = variant
        timed = group[5:]
        reference = groups[False, fused][5:]
        values = np.array([t["decode_cuda_ms"] / 1000 for t in timed])
        baseline = np.array([t["decode_cuda_ms"] / 1000 for t in reference])
        ratio = values.mean() / baseline.mean()
        ratios = values[indices].mean(1) / baseline[indices].mean(1)
        low = float(np.quantile(ratios, 0.025))
        high = float(np.quantile(ratios, 0.975))
        rows.append(
            dict(
                encoded=encoded,
                fused=fused,
                trials=20,
                warmups=5,
                median_decode_cuda_s=float(np.median(values)),
                p95_decode_cuda_s=float(np.quantile(values, 0.95)),
                mean_decode_cuda_s=float(values.mean()),
                mean_constructor_s=float(np.mean([t["construction_seconds"] for t in timed])),
                decode_ratio_vs_same_transport_raw=float(ratio),
                decode_ratio_ci95=[low, high],
                **timed[0]["storage"],
                within_1_05_transport_overhead=bool(high <= 1.05),
                not_selector_or_request_latency_claim=True,
            )
        )
    return rows


def analyze_timing(root: Path, plan: Any, producer_failures: list[Any]) -> None:
    import pandas as pd  # type: ignore[import-untyped]  # Common lock has no pandas-stubs.

    out = root / "transport_timing_analysis"
    out.mkdir(exist_ok=False)
    rows: list[Any] = []
    trials: list[Any] = []
    failures = list(producer_failures)
    inputs: dict[str, str] = {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        result_path = folder / "result.json"
        if not result_path.exists():
            continue  # Already explicit in producer failure ledger.
        inputs[str(result_path)] = sha(result_path)
        result = json.loads(result_path.read_text())
        if result["status"] != "completed":
            continue
        for arm in job["arms"]:
            role = arm["role"]
            path = folder / f"{role}-timing_trials.json"
            try:
                if sha(path) != result["outputs"][path.name]:
                    raise ValueError("trial hash changed")
                inputs[str(path)] = sha(path)
                data = json.loads(path.read_text())
                summaries = summarize_trials(data)
                for row in summaries:
                    row.update(task=job["task"], role=role)
                rows.extend(summaries)
                trials.extend(dict(t, task=job["task"], role=role) for t in data)
            except Exception as exc:
                failures.append(dict(worker=job["index"], role=role, error=str(exc)))
    pd.DataFrame(rows).to_parquet(out / "summary.parquet", index=False)
    pd.DataFrame(trials).to_parquet(out / "trials.parquet", index=False)
    target = [r for r in rows if r["encoded"] and r["fused"]]
    complete = len(rows) == 16 and not failures
    decision = dict(
        status="complete_consumed_timing_diagnostic" if complete else "failed",
        failures=failures,
        all_trials_bitwise=complete,
        representation_overhead_gate_passed=complete
        and all(r["within_1_05_transport_overhead"] for r in target),
        common_optimization_for_both_methods=True,
        selector_superiority_established=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        whole_request_peak_certified=False,
        bound="post_selection_shared_prefill_not_request_latency_or_benchmark_quality",
    )
    write_once(out / "GATE.json", decision)
    report = (
        "# Shared lossless transport: randomized repeated timing diagnostic\n\n"
        "Same two consumed inputs, unchanged selected K/V and original greedy reference. "
        "Five warmups and twenty timed blocks per raw/encoded x separate/fused variant. "
        "Variant order is randomized within each block. CPU logit transfers, selection and "
        "prefill are outside the decode timer. Each trial must reproduce all 32 logits/tokens "
        "bitwise. Paired bootstrap ratios describe repeated timing on these inputs, NOT "
        "population quality. Construction is reported separately.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\nEviCastKV and Top-K receive the same codec/transport optimization. "
        "This cannot establish selector superiority, PPL/ROUGE gains or end-to-end speedup. "
        "Decode-owned storage includes raw append-only history and current-layer full K/V. "
        "Construction/whole-request KV/HBM peak remains uncertified. "
        "Prior absolute FullKV-fidelity failures remain unchanged. "
        "No fresh or large campaign follows.\n"
    )
    (out / "report.md").write_text(report)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            inputs=inputs,
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
