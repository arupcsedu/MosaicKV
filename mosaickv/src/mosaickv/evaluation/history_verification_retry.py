"""Strict additive repair for JSON identity-container verification bookkeeping.

The unchanged historical scanner rebuilds ``scopes`` and ``scoped_identities``
as lists of tuples. Saved JSON heads contain lists of lists. Those two forms
encode the same ordered metadata, but V36's Python equality rejected them.
Only that representation normalization is accepted here. No source traversal,
identity value, historical error, path signature or numerical guard is waived.
"""

from __future__ import annotations

import copy
import threading
from unittest.mock import patch

from mosaickv.evaluation import global_history_resume as legacy
from mosaickv.evaluation.fox_campaign import digest

_ORIGINAL_VERIFY = legacy.verify_verification
_VERIFY_LOCK = threading.Lock()
_NESTED_IDENTITIES = ("scopes", "scoped_identities")
_MUTABLE_VERIFICATION_FIELDS = {
    "errors",
    "verification_paths",
    "verification_index",
    "traversal_complete",
}


def _normalized_view(base, result):
    if not isinstance(base, dict) or not isinstance(result, dict):
        raise ValueError("complete immutable verification base and result required")
    state = result.get("state")
    if not isinstance(state, dict) or set(base) != set(state):
        raise ValueError("verification cannot add or remove scanner fields")
    immutable = set(base) - _MUTABLE_VERIFICATION_FIELDS
    if digest({key: base[key] for key in immutable}) != digest(
        {key: state[key] for key in immutable}
    ):
        raise ValueError("verification changed immutable traversal or identity values/types")
    if not isinstance(state.get("errors"), list) or digest(
        state["errors"][: len(base["errors"])]
    ) != digest(base["errors"]):
        raise ValueError("verification rewrote historical error records")
    normalized = dict(result)
    normalized_state = dict(state)
    for name in _NESTED_IDENTITIES:
        for values in (base[name], state[name]):
            if not isinstance(values, list) or any(
                not isinstance(row, (list, tuple)) for row in values
            ):
                raise ValueError("nested identity metadata must remain explicit JSON arrays")
        # The immutable fingerprint above already proves exact ordered values.
        # Keep the base's concrete representation solely for legacy validation.
        normalized_state[name] = copy.deepcopy(base[name])
    normalized["state"] = normalized_state
    if digest(normalized_state) != digest(state):
        raise ValueError("identity bookkeeping normalization changed serialized evidence")
    return normalized


def verify_verification(base, result, *, owner_index):
    """Read-only validation; accept no change except the two container forms."""
    normalized = _normalized_view(base, result)
    return _ORIGINAL_VERIFY(base, normalized, owner_index=owner_index)


def advance_verification(base, roots, scanner, *, owner_index):
    """Resume the saved epoch with unchanged scanner, budgets and stat guards."""
    if not _VERIFY_LOCK.acquire(blocking=False):
        raise RuntimeError("verification retry requires isolated worker processes")
    try:
        validated = []

        def validate_and_capture(actual_base, actual_result, *, owner_index):
            view = _normalized_view(actual_base, actual_result)
            _ORIGINAL_VERIFY(actual_base, view, owner_index=owner_index)
            validated.append((actual_result, view))
            return True

        with patch.object(legacy, "verify_verification", validate_and_capture):
            result = legacy.advance_verification(base, roots, scanner, owner_index=owner_index)
        # Publication remains JSON-equivalent to the actual scanner state;
        # its full canonical state hash and every observation count are intact.
        if len(validated) != 1 or validated[0][0] is not result:
            raise ValueError(
                "legacy verification did not validate the published result exactly once"
            )
        return validated[0][1]
    finally:
        _VERIFY_LOCK.release()
