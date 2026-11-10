#!/usr/bin/env python3
"""Paired common-runtime reference/index experiment on consumed timing inputs.

Five request warmups and three counterbalanced repetitions per method/variant.
Changes only descriptor enumeration; candidate K/V/positions/graph must match.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

from evicastkv_campaign import SOURCE, load_lock
from evicastkv_graph_profile import registered_plan

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

# The common optimization stage namespaces warmups by implementation variant;
# ordinary measurement deliberately has one namespace per method/cap instead.
STAGE = "optimization"
ARMS = [
    [m, 40, v]
    for m in ("prompt_attention_topk", "mosaickv_exact")
    for v in ("pool-reference", "pool-index")
]


def balanced_order(arms, input_index, repeat):
    """Balance implementation order independently for both compared methods."""
    offset = (input_index + repeat) % len(arms)
    ordered = list(arms[offset:] + arms[:offset])
    for method in ("prompt_attention_topk", "mosaickv_exact"):
        reference = ordered.index((method, 40, "pool-reference"))
        optimized = ordered.index((method, 40, "pool-index"))
        if (reference < optimized) != ((input_index + repeat) % 2 == 0):
            ordered[reference], ordered[optimized] = ordered[optimized], ordered[reference]
    return ordered


def prepare(root, parent):
    if root.exists():
        raise FileExistsError("pooling experiment requires new output")
    revision = clean_source(SOURCE)
    lock, bindings, _ = registered_plan(parent)
    # Four consumed timing inputs, one per worker, no new cohort selection.
    ids = lock["cohorts"]["coco"]["timing"]["sample_ids"][:4]
    if len(ids) != 4:
        raise ValueError("four registered timing inputs required")
    lock.update(name="fox_pooling_index_v1", source_root=str(SOURCE), source_sha=revision)
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=4,
        six_a100s_explicitly_authorized=True,
        cpus_per_worker=12,
        wall_minutes=120,
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["historical_bindings"].update(bindings)
    lock["warmups"], lock["repeats"] = 5, 3
    lock["pooling_index_study"] = dict(
        sample_ids=ids,
        only_change="one stable layer index replaces repeated descriptor scans",
        graph_math_unchanged=True,
        candidate_changes_forbidden=True,
        source=str(SOURCE / "mosaickv/src/mosaickv/evaluation/fox_pooling_index.py"),
        source_sha256=sha(SOURCE / "mosaickv/src/mosaickv/evaluation/fox_pooling_index.py"),
        fresh_confirmation_authorized=False,
        strict_latency_upper_ratio=1.0,
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(
        root / "jobs" / f"{STAGE}.json",
        dict(
            protocol_sha256=sha(root / "protocol_lock.json"),
            jobs=[
                dict(workload="coco", subset="timing", indices=[i], stage=STAGE, arms=ARMS)
                for i in range(4)
            ],
        ),
    )
    return dict(workers=4, spec_sha256=sha(root / "jobs" / f"{STAGE}.json"))


def worker(root, index, spec_sha):
    import evicastkv_worker as model_worker

    from mosaickv.evaluation import fox_optimization
    from mosaickv.evaluation.fox_pooling_index import pooling_index

    lock = load_lock(root)
    bound = lock["pooling_index_study"]
    if sha(Path(bound["source"])) != bound["source_sha256"] or index not in range(4):
        raise ValueError("unregistered optimized source/worker")
    original = fox_optimization.budget_implementation

    @contextlib.contextmanager
    def implementation(local_lock, variant, *, required):
        if variant in {"pool-reference", "pool-index"}:
            with pooling_index(variant == "pool-index"):
                yield
        else:
            with original(local_lock, variant, required=required):
                yield

    with (
        patch.object(fox_optimization, "budget_implementation", implementation),
        patch.object(fox_optimization, "factorial_order", balanced_order),
    ):
        model_worker.run_worker(root, STAGE, index, spec_sha)


def analyze(root):
    import statistics

    from evicastkv_analyze import collect, pairing
    from evicastkv_optimization import semantic_check

    from mosaickv.evaluation.fox_statistics import paired_scalar_statistics

    lock = load_lock(root)
    _, _, rows, errors = collect(root, STAGE)
    ids = lock["pooling_index_study"]["sample_ids"]
    expected = {(sid, m, v, r) for sid in ids for m, _, v in ARMS for r in range(3)}
    actual = {(r["sample_id"], r["method"], r["variant"], r["repeat"]) for r in rows}
    if len(rows) != 48 or actual != expected:
        errors.append(dict(error="four inputs x four arms x three repeats required"))
    checks, stats = [], []
    by_key = {(r["sample_id"], r["method"], r["repeat"], r["variant"]): r for r in rows}
    if not errors:
        pairing(rows)
        for sid in ids:
            for method in ("prompt_attention_topk", "mosaickv_exact"):
                for repeat in range(3):
                    try:
                        checks.append(
                            semantic_check(
                                by_key[sid, method, repeat, "pool-reference"],
                                by_key[sid, method, repeat, "pool-index"],
                            )
                        )
                    except Exception as exception:
                        errors.append(
                            dict(sample_id=sid, method=method, repeat=repeat, error=str(exception))
                        )
        if not errors:
            for method in ("prompt_attention_topk", "mosaickv_exact"):
                points = {
                    v: {
                        sid: statistics.mean(
                            by_key[sid, method, r, v]["e2e_wall_seconds"] for r in range(3)
                        )
                        for sid in ids
                    }
                    for v in ("pool-reference", "pool-index")
                }
                groups = {
                    sid: tuple(by_key[sid, method, 0, "pool-reference"]["media_group_ids"])
                    for sid in ids
                }
                comparison = paired_scalar_statistics(
                    points["pool-reference"],
                    points["pool-index"],
                    groups,
                    dict.fromkeys(ids, "COCO"),
                    resamples=10000,
                    family_size=2,
                )
                stats.append(
                    dict(
                        method=method,
                        paired_statistics=comparison,
                        small_consumed_timing_diagnostic_not_paper_evidence=True,
                    )
                )
    write_once(root / "analysis/semantic_checks.json", checks)
    write_once(root / "analysis/paired_timing.json", stats)
    gate = dict(
        status="passed_equivalence" if not errors else "failed",
        errors=errors,
        rows=len(rows),
        checks=len(checks),
        source_sha=lock["source_sha"],
        latency_promotion_not_automatic=True,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "analysis/GATE.json", gate)
    return gate
