"""CPU-only replay of opt-in, constructor-only actual-snapshot reuse proofs.

Every ownership observation remains present. Reuse may preserve a previously
resolved physical mapping only while its complete owner/view fingerprint and
weak representatives remain unchanged in the SAME consecutive constructor
phase. This verifier retains scalar provenance, never tensors, storage, owner
objects, or the mapping itself. It does not certify unobserved storage.
"""

import re
from collections.abc import Mapping

_CONSTRUCTOR = re.compile(r"prefill_layer_(0|[1-9][0-9]*)_constructor_workspace\Z")
_SCOPE = "consecutive_constructor_observations_only"
_REUSE_REASON = "identical_continuously_live_constructor_owners"
_FRESH_REASONS = frozenset(
    {
        "mandatory_lifecycle_snapshot",
        "no_cached_snapshot",
        "phase_transition",
        "previous_owner_died",
        "owner_representative_or_storage_changed",
        "owner_or_view_set_changed",
    }
)


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


def _hash(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} requires a canonical SHA-256")
    return value


class SnapshotReuseReplay:
    """Fail-closed checker for a ledger that explicitly declares reuse enabled.

    Legacy ledgers without declared reuse must not instantiate this checker.
    Once instantiated, omitting even one observation's proof is an error.
    """

    def __init__(self):
        self.observations = self.fresh = self.reused = self.snapshot_calls = 0
        self._previous = None
        self._failed = False

    def observe(self, proof, *, index, phase, resolved):
        if self._failed:
            raise RuntimeError("snapshot reuse replay already encountered a failed observation")
        try:
            return self._observe(proof, index=index, phase=phase, resolved=resolved)
        except Exception:
            self._failed = True
            raise

    def _observe(self, proof, *, index, phase, resolved):
        from mosaickv.evaluation.live_owner_snapshot_cache import resolution_mapping_sha256

        _integer(index, "observation index")
        if index != self.observations:
            raise ValueError("every snapshot reuse observation must be contiguous from zero")
        if type(phase) is not str or not phase or len(phase) > 256:
            raise ValueError("bounded explicit snapshot observation phase required")
        if not isinstance(proof, Mapping):
            raise ValueError("every declared-reuse observation requires its explicit proof")
        if (
            proof.get("scope") != _SCOPE
            or proof.get("enabled") is not True
            or proof.get("observation_skipped") is not False
            or proof.get("allocator_bytes_inferred") is not False
            or proof.get("outcome") != "resolved"
            or proof.get("phase") != phase
        ):
            raise ValueError("snapshot reuse proof scope, phase or strict control flags changed")
        if _integer(proof.get("observation_index"), "proof observation index") != index:
            raise ValueError("snapshot reuse proof belongs to a different observation")
        origin = _integer(proof.get("source_observation_index"), "source observation index")
        calls = _integer(proof.get("snapshot_calls"), "actual snapshot calls", minimum=1)
        count = _integer(proof.get("weak_representative_count"), "weak representative count")
        alive = proof.get("prior_weak_representatives_alive")
        if type(alive) is not bool:
            raise ValueError("prior weak representative liveness must be an explicit boolean")
        fingerprint = _hash(proof.get("owner_view_fingerprint_sha256"), "owner/view fingerprint")
        mapping_sha = _hash(proof.get("resolution_mapping_sha256"), "actual resolution mapping")
        if (
            not isinstance(resolved, Mapping)
            or resolved.get("all_required_owners_resolved") is not True
        ):
            raise ValueError("complete actually resolved physical ownership mapping required")
        if resolution_mapping_sha256(resolved) != mapping_sha:
            raise ValueError("snapshot reuse proof does not match its actual physical mapping")
        mode, reason = proof.get("mode"), proof.get("reason")
        constructor = _CONSTRUCTOR.fullmatch(phase) is not None
        previous = self._previous
        if mode == "fresh_native_snapshot":
            if origin != index or calls != self.snapshot_calls + 1:
                raise ValueError("fresh native snapshot must have its own origin and one new call")
            if reason not in _FRESH_REASONS:
                raise ValueError("fresh snapshot has an unrecognized or disabled reason")
            if (not constructor and reason != "mandatory_lifecycle_snapshot") or (
                constructor and reason == "mandatory_lifecycle_snapshot"
            ):
                raise ValueError("lifecycle observations require a fresh mandatory snapshot")
            self.fresh += 1
        elif mode == "reused_actual_resolution":
            if not constructor or previous is None or previous["phase"] != phase:
                raise ValueError(
                    "actual resolution reuse requires the same consecutive constructor"
                )
            if (
                reason != _REUSE_REASON
                or alive is not True
                or calls != self.snapshot_calls
                or origin != previous["origin"]
                or fingerprint != previous["fingerprint"]
                or mapping_sha != previous["mapping_sha"]
                or count != previous["representatives"]
            ):
                raise ValueError(
                    "reused snapshot changed origin, owners, views, liveness or counters"
                )
            self.reused += 1
        else:
            raise ValueError("explicit fresh or reused actual native resolution mode required")
        self.snapshot_calls = calls
        self.observations += 1
        self._previous = dict(
            phase=phase,
            origin=origin,
            fingerprint=fingerprint,
            mapping_sha=mapping_sha,
            representatives=count,
        )
        return dict(index=index, phase=phase, mode=mode, source_observation_index=origin)

    def summary(self):
        if self._failed:
            raise RuntimeError("snapshot reuse replay encountered an invalid or missing proof")
        if not self.observations:
            raise ValueError("snapshot reuse replay requires at least one complete observation")
        return dict(
            schema="independent_constructor_snapshot_reuse_replay_v1",
            observations=self.observations,
            fresh=self.fresh,
            reused=self.reused,
            fresh_native_snapshots=self.fresh,
            reused_actual_resolutions=self.reused,
            snapshot_calls=self.snapshot_calls,
            every_observation_has_proof=True,
            only_consecutive_constructor_reuse=True,
            allocator_bytes_inferred=False,
            whole_request_KV_peak_certified=False,
        )
