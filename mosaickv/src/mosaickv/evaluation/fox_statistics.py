"""Prespecified paired cluster bootstrap; corpus CIDEr is never shard-averaged."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
from mosaickv.evaluation.metrics.types import MetricSample


def media_components(identities, groups_by_id):
    """Transitive closure of input-derived shared-media IDs, independent of scores."""
    identities = tuple(identities)
    if len(set(identities)) != len(identities) or set(identities) != set(groups_by_id):
        raise ValueError("group manifest must cover unique input identities exactly")
    parent = {sid: sid for sid in identities}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    owners = {}
    for sid in identities:
        if not groups_by_id[sid]:
            raise ValueError("missing reusable-media groups")
        for group in groups_by_id[sid]:
            previous = owners.setdefault(group, sid)
            left, right = find(sid), find(previous)
            parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for index, sid in enumerate(identities):
        groups[find(sid)].append(index)
    return tuple(tuple(values) for _, values in sorted(groups.items()))


def cluster_draws(identities, groups_by_id, task_by_id, *, resamples=10000, seed=20271014):
    if resamples < 1 or set(identities) != set(task_by_id):
        raise ValueError("invalid resampling contract")
    components = media_components(identities, groups_by_id)
    strata = defaultdict(list)
    for component in components:
        signature = tuple(sorted({task_by_id[identities[index]] for index in component}))
        strata[signature].append(component)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draw = []
        for _signature, clusters in sorted(strata.items()):
            for _cluster in clusters:
                draw.extend(clusters[rng.randrange(len(clusters))])
        draws.append(tuple(draw))
    return tuple(draws)


def interval(values, *, confidence):
    if not 0 < confidence < 1 or not values or not np.isfinite(values).all():
        raise ValueError("invalid finite bootstrap distribution")
    tail = (1 - confidence) / 2
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]


def difference_report(reference, candidate, differences, *, family_size=6):
    delta = candidate - reference
    return dict(
        reference_value=reference,
        candidate_value=candidate,
        difference=delta,
        relative_difference=delta / abs(reference) if reference else None,
        ordinary_95_ci=interval(differences, confidence=0.95),
        simultaneous_ci=interval(differences, confidence=1 - 0.05 / family_size),
        family_size=family_size,
        simultaneous_coverage=1 - 0.05 / family_size,
        interpretation="development_only_not_equivalence_or_confirmatory_evidence",
    )


def _cider_chunk(payload):
    points, draws = payload
    scorer = TokenizedCider([sample for point in points for sample in point])
    # TokenizedCider creates unique evaluator-local IDs for every occurrence.
    return [[scorer([point[index] for index in draw]) for point in points] for draw in draws]


def corpus_cider_statistics(
    points: dict[str, tuple[MetricSample, ...]],
    comparisons,
    groups_by_id,
    *,
    resamples=10000,
    seed=20271014,
    workers=4,
):
    if not points or workers < 1:
        raise ValueError("missing corpus points or CPU allocation")
    keys = tuple(sorted(points))
    ordered = []
    identities = None
    reference_annotations = None
    for key in keys:
        samples = points[key]
        by_id = {s.sample_id: s for s in samples}
        if len(by_id) != len(samples):
            raise ValueError("duplicate corpus identities")
        if identities is None:
            identities = tuple(sorted(by_id))
            reference_annotations = {sid: by_id[sid].references for sid in identities}
        if (
            tuple(sorted(by_id)) != identities
            or {sid: by_id[sid].references for sid in identities} != reference_annotations
        ):
            raise ValueError("corpus points must have identical inputs AND references")
        ordered.append(tuple(by_id[sid] for sid in identities))
    draws = cluster_draws(
        identities, groups_by_id, dict.fromkeys(identities, "COCO"), resamples=resamples, seed=seed
    )
    width = max(1, math.ceil(len(draws) / (workers * 2)))
    payloads = [(tuple(ordered), draws[i : i + width]) for i in range(0, len(draws), width)]
    if workers == 1:
        chunks = map(_cider_chunk, payloads)
        boot = np.asarray([row for chunk in chunks for row in chunk])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            boot = np.asarray([row for chunk in pool.map(_cider_chunk, payloads) for row in chunk])
    scorer = TokenizedCider([sample for point in ordered for sample in point])
    scores = {key: scorer(ordered[index]) for index, key in enumerate(keys)}
    reports = []
    for reference, candidate in comparisons:
        if reference not in points or candidate not in points:
            raise ValueError("unknown comparison point")
        reports.append(
            dict(
                reference=reference,
                candidate=candidate,
                **difference_report(
                    scores[reference],
                    scores[candidate],
                    list(boot[:, keys.index(candidate)] - boot[:, keys.index(reference)]),
                ),
            )
        )
    return dict(
        metric="cider",
        scope="corpus",
        scores=scores,
        contrasts=reports,
        resamples=resamples,
        seed=seed,
        unique_inputs=len(identities),
        media_clusters=len(media_components(identities, groups_by_id)),
        reducer="pinned lmms_eval PTBTokenizer+Cider x100; DF recomputed each draw",
    )


def paired_scalar_statistics(
    reference,
    candidate,
    groups_by_id,
    task_by_id,
    *,
    resamples=10000,
    seed=20271014,
    hierarchy=False,
):
    identities = tuple(sorted(reference))
    if set(reference) != set(candidate) or not identities:
        raise ValueError("paired inputs differ")
    if any(
        not math.isfinite(value) for values in (reference, candidate) for value in values.values()
    ):
        raise ValueError("nonfinite observations")
    draws = cluster_draws(identities, groups_by_id, task_by_id, resamples=resamples, seed=seed)

    def reduce(values, draw):
        selected = defaultdict(list)
        for index in draw:
            sid = identities[index]
            selected[task_by_id[sid] if hierarchy else "all"].append(values[sid])
        return float(np.mean([np.mean(scores) for scores in selected.values()]))

    original = tuple(range(len(identities)))
    r, c = reduce(reference, original), reduce(candidate, original)
    boot = [(reduce(reference, draw), reduce(candidate, draw)) for draw in draws]
    delta = [b - a for a, b in boot]
    ratios = [a / b for a, b in boot if b > 0]
    return dict(
        **difference_report(r, c, delta),
        speedup_ratio_of_means=r / c if c > 0 else None,
        speedup_95_ci=interval(ratios, confidence=0.95) if len(ratios) == len(draws) else None,
        percentage_reduction=100 * (1 - c / r) if r > 0 else None,
        resamples=resamples,
        seed=seed,
        unique_inputs=len(identities),
        hierarchy="equal-subtask mean" if hierarchy else "input mean",
        cluster_strata="shared-media connected components stratified by subtask signature",
    )
