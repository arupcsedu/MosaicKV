"""Separate, explicitly exploratory iso-quality memory screening of curves.

Unequal-byte points are NEVER called byte-matched. A positive small-development
screen is not confirmation, and cannot choose the nearest quality-eligible
competitor without a separately frozen independent quality target.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def exploratory_memory_screen(
    frame: Any, *, candidate: str, baselines: tuple[str, ...], replicates: int, seed: int
) -> list[dict[str, Any]]:
    pairs = []
    for task in sorted(frame.task.unique()):
        candidate_rows = frame[(frame.task == task) & (frame.method == candidate)]
        for cap, current in candidate_rows.groupby("cap_mib"):
            for baseline in baselines:
                for base_cap, reference in frame[
                    (frame.task == task) & (frame.method == baseline)
                ].groupby("cap_mib"):
                    if cap < base_cap:
                        pairs.append((task, cap, current, baseline, base_cap, reference))
    alpha = 0.05 / max(1, 2 * len(pairs))
    rng = np.random.default_rng(seed)
    output = []
    for task, cap, current, baseline, base_cap, reference in pairs:
        current = current.sort_values("sample_id").reset_index(drop=True)
        reference = reference.sort_values("sample_id").reset_index(drop=True)
        if (
            current.sample_id.duplicated().any()
            or reference.sample_id.duplicated().any()
            or current.sample_id.to_list() != reference.sample_id.to_list()
        ):
            raise ValueError("equal-quality screen requires unique identical paired sample IDs")
        n = len(current)
        if n < 2 or replicates < 100:
            raise ValueError("exploratory bootstrap needs at least two samples and 100 replicates")
        draw = rng.integers(0, n, size=(replicates, n))

        def ppl(group: Any, draw: Any = draw) -> np.ndarray:
            nll = group.nll_sum.to_numpy(dtype=float)
            tokens = group.tokens.to_numpy(dtype=float)
            if not np.isfinite(nll).all() or not (tokens > 0).all():
                raise ValueError(
                    "PPL screen requires finite NLL and positive complete token counts"
                )
            return np.exp(nll[draw].sum(1) / tokens[draw].sum(1))

        rouge = current.rouge_l_f1.to_numpy()[draw].mean(1) - reference.rouge_l_f1.to_numpy()[
            draw
        ].mean(1)
        ppl_delta = ppl(current) - ppl(reference)
        rouge_low = float(np.quantile(rouge, alpha / 2))
        ppl_high = float(np.quantile(ppl_delta, 1 - alpha / 2))
        record: dict[str, Any] = dict(
            task=task,
            candidate=candidate,
            candidate_cap_mib=float(cap),
            baseline=baseline,
            baseline_cap_mib=float(base_cap),
            sample_count=n,
            classification="exploratory_equal_quality_memory_NOT_byte_matched_NOT_confirmation",
            bootstrap_replicates=replicates,
            familywise_alpha=0.05,
            adjusted_rouge_delta_lower=rouge_low,
            adjusted_ppl_delta_upper=ppl_high,
            quality_screen_pass=rouge_low >= 0 and ppl_high <= 0,
            paper_claim_eligible=False,
            nearest_competitor_established=False,
            independent_quality_target="not_registered_use_only_to_plan_independent_confirmation",
        )
        memory_pass = True
        for boundary in ("initial", "final", "peak"):
            column = f"{boundary}_active_kv_storage_bytes"
            a, b = current[column].to_numpy(), reference[column].to_numpy()
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not (b > 0).all():
                raise ValueError("memory screen requires complete positive storage accounting")
            record[f"{boundary}_memory_reduction"] = float(1 - a.sum() / b.sum())
            memory_pass &= bool((a < b).all())
        latency = current.end_to_end_time.to_numpy()[draw].mean(
            1
        ) / reference.end_to_end_time.to_numpy()[draw].mean(1)
        if not np.isfinite(latency).all() or not math.isfinite(rouge_low + ppl_high):
            raise ValueError("non-finite exploratory metrics or timing")
        record.update(
            strictly_smaller_memory=memory_pass,
            exploratory_memory_quality_pass=memory_pass and record["quality_screen_pass"],
            informational_latency_ratio_upper=float(np.quantile(latency, 1 - alpha / 2)),
        )
        output.append(record)
    return output
