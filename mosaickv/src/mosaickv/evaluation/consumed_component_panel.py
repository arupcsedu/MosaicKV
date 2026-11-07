"""ID-only bounded component diagnostic, explicitly not a fresh cohort."""

from dataclasses import asdict, replace
from hashlib import sha256

from mosaickv.evaluation.storage_six_overlay import COMPONENT_MODES, OverlayPolicy


def select_inputs(samples, excluded, count=16):
    """Selection depends only on IDs, never answers, historical scores or failures."""
    if type(count) is not int or count != 16:
        raise ValueError("registered sixteen-consumed-input diagnostic only")
    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)) or not set(excluded) <= set(ids):
        raise ValueError("unique consumed IDs and reviewed exclusions required")
    available = [s for s in samples if s["sample_id"] not in excluded]
    if len(available) < count:
        raise ValueError("insufficient already consumed identities")
    chosen = sorted(
        available,
        key=lambda s: (
            sha256(("six-panel-v1:" + s["sample_id"]).encode()).hexdigest(),
            s["sample_id"],
        ),
    )[:count]
    return [{k: s[k] for k in ("sample_id", "prompt", "media")} for s in chosen]


def frozen_policies(seed):
    base = OverlayPolicy(**dict(seed, layers=tuple(seed["layers"])))
    if not (
        base.inline_repair
        and base.graph_positional_search == "indexed"
        and base.graph_cosine_topk == "partition"
        and base.precision_transaction == "prototype"
        and not base.replace_source_codes
        and base.component_mode == "full"
    ):
        raise ValueError("completed frozen inline/indexed/partition policy required")
    return {
        mode: asdict(
            replace(
                base,
                component_mode=mode,
                precision_transaction="none" if mode == "no_forecast" else "prototype",
            )
        )
        for mode in COMPONENT_MODES
    }
