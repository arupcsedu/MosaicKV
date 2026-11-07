"""Correctness-only checks on consumed inputs; no benchmark scores or labels."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from mosaickv.evaluation.source_hybrid_validation import BOUNDARIES


def consumed_inputs(rows):
    """Strip annotations without selecting on answers, scores, or failure identities."""
    if len(rows) != 16 or len({r["sample_id"] for r in rows}) != 16:
        raise ValueError("the complete frozen sixteen-consumed-input panel is required")
    result = []
    for row in rows:
        if not row.get("prompt") or not row.get("media"):
            raise ValueError("complete frozen prompt and media required")
        result.append({k: row[k] for k in ("sample_id", "prompt", "media")})
    return result


def compare(parent, host, device, parent_counterfactual, host_report, device_report):
    """Require exact numeric/decision parity, not a claim about reference answers."""
    if parent.ndim != 2 or parent.shape[0] != 128:
        raise ValueError("all 128 ordinary-parent vocabulary vectors required")
    if not all(torch.isfinite(t).all() for t in (parent, host, device, parent_counterfactual)):
        raise ValueError("nonfinite model or counterfactual logits")
    if not torch.equal(host, device):
        raise ValueError("host/device guard transport changed model logits")
    if not torch.equal(parent[1:], parent_counterfactual):
        raise ValueError("live-parent counterfactual does not match ordinary parent")
    if not torch.equal(parent.argmax(-1), device.argmax(-1)):
        raise ValueError("guard failed to preserve ordinary-parent greedy tokens")
    h, d = (r["live_anchor_guard"] for r in (host_report, device_report))
    for record in (h, d):
        if (
            record["steps"] != 127
            or record["observed_parent_logit_steps"] != 127
            or record["fullkv_fallback"] is not False
            or record["benchmark_labels_accessed"] is not False
        ):
            raise ValueError("complete, no-fallback, label-blind guard required")
    a, b = (torch.tensor(r["observed"], dtype=torch.float64) for r in (h, d))
    if a.shape != (127, 4) or b.shape != a.shape or not torch.equal(a, b):
        raise ValueError("all per-step FP64 guard records must match bitwise")
    tail = d["tail_graph"]
    if (
        tail["records_validated"] is not True
        or tail["record_validation_failed"] is not False
        or tail["recorded_steps"] != 127
    ):
        raise ValueError("device records must validate before publication")
    return dict(
        all_128_host_device_logits_bitwise=True,
        all_127_counterfactual_logits_bitwise_parent=True,
        all_127_guard_records_bitwise=True,
        ordinary_parent_greedy_agreement=1.0,
        actual_vetoes=int(d["prototype_or_exact_changes_vetoed"]),
        answer_quality_measured=False,
    )


def same_storage(host, device):
    for key in BOUNDARIES:
        if type(host.get(key)) is not int or host[key] <= 0 or host[key] != device.get(key):
            raise ValueError(f"guard transport changed measured cache backing: {key}")
    return {k: host[k] for k in BOUNDARIES}


@contextmanager
def parent_export():
    """Test-only copy before the production owner deliberately clears its exports."""
    from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard

    result = {}
    original = LiveAnchorGuard.__exit__

    def close(owner, *args):
        try:
            return original(owner, *args)
        finally:
            if owner.parent_logits:
                result["logits"] = torch.stack(owner.parent_logits).clone()

    with patch.object(LiveAnchorGuard, "__exit__", close):
        yield result


@contextmanager
def finite_fault(enabled):
    """Mechanical veto check only, never reported as repair or quality recovery."""
    from mosaickv.evaluation.guard_tail_graph import GuardTail

    original = GuardTail.run
    injections = []

    def run(owner, parent, active):
        if enabled and owner.calls == 3:
            injections.append(owner.calls)
            active = active + 128.0
        return original(owner, parent, active)

    with patch.object(GuardTail, "run", run):
        yield injections
