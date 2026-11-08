"""Default-off common workspace cap; identical correction/verification math.

Only complete elementwise predicate spans change. All source positions, finite
checks, sparse indices/values, protected-row rejection and commit checks remain.
It is common transport, never an EviCastKV-exclusive compression mechanism.
"""

from contextlib import contextmanager
from unittest.mock import patch

from mosaickv.evaluation.deferred_transport_checks import Validation


@contextmanager
def use(*, scalars=262144):
    if type(scalars) is not int or not 0 < scalars <= 524288:
        raise ValueError("positive predicate workspace cap at most 524288 scalars required")
    original = Validation.predicate_span
    report = dict(
        enabled=True,
        common_to_every_method=True,
        cap_scalars=scalars,
        calls=0,
        maximum_actual_span=0,
        unchanged_elementwise_predicates=True,
        unchanged_sparse_patch_order=True,
        full_coverage_required=True,
    )

    def bounded(state, size, requested, kind):
        if kind not in ("corrections", "verification"):
            return original(state, size, requested, kind)
        span = min(size, scalars, requested if requested is not None else scalars)
        # Record actual spans, not a fictitious full-matrix temporary bound.
        # The transaction's validation flags and commit remain untouched.
        actual = original(state, span, span, kind)
        report["calls"] += 1
        report["maximum_actual_span"] = max(report["maximum_actual_span"], actual)
        return actual

    with patch.object(Validation, "predicate_span", bounded):
        yield report
