"""Retrospective engineering contrasts against EVERY tested Prefix runtime.

Does not rewrite the registered earlier reuse-only gate. Trial intervals are
not benchmark-quality confidence intervals. No budget matching is implied.
"""

import numpy as np


def analyze(rows, *, seed):
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    arms = sorted({(m, a) for m, a, _ in keys})
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in arms for t in range(-5, 20)
    }:
        raise ValueError("complete unique 5/20 engineering arms required")
    values = {}
    for m, a in arms:
        entries = sorted(
            (r for r in rows if r["method"] == m and r["arm"] == a and r["trial"] >= 0),
            key=lambda x: x["trial"],
        )
        array = np.asarray([x["measurement"]["total_wall_seconds"] for x in entries])
        if not np.isfinite(array).all() or not (array > 0).all():
            raise ValueError("finite positive wall times required")
        values[m, a] = array
    baselines = {a: v for (m, a), v in values.items() if m == "prefixkv_reimpl"}
    if not baselines or ("mosaickv_exact", "original") not in values:
        raise ValueError("original EviCastKV and tested Prefix controls required")
    samples = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    # Four possible Evi alternatives times (five Prefix alternatives + own
    # original) times two inputs = maximum 48 distinct comparisons.
    tail = 0.05 / (2 * 48)
    output = []
    for (method, arm), candidate in values.items():
        if method != "mosaickv_exact" or arm == "original":
            continue

        def contrast(other, candidate=candidate):
            ratio = candidate[samples].mean(1) / other[samples].mean(1)
            return dict(
                mean_ratio=float(candidate.mean() / other.mean()),
                adjusted_ratio_CI=np.quantile(ratio, [tail, 1 - tail]).tolist(),
            )

        own = contrast(values["mosaickv_exact", "original"])
        peers = {a: contrast(other) for a, other in baselines.items()}
        output.append(
            dict(
                arm=arm,
                own_original=own,
                every_tested_prefix_runtime=peers,
                own_improvement_gate=own["adjusted_ratio_CI"][1] <= 0.99,
                similar_to_all_tested_prefix_runtimes=all(
                    x["adjusted_ratio_CI"][1] <= 1.05 for x in peers.values()
                ),
                no_slower_than_all_tested_prefix_runtimes=all(
                    x["adjusted_ratio_CI"][1] <= 1.0 for x in peers.values()
                ),
            )
        )
    return dict(
        retrospective=True,
        engineering_repetitions_not_quality_samples=True,
        maximum_family_comparisons=48,
        contrasts=output,
        fastest_observed_prefix_arm=min(baselines, key=lambda k: baselines[k].mean()),
        paper_superiority_established=False,
        byte_matched=False,
    )
