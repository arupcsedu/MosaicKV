"""Unused, opt-in constructor-only reuse of an actual CUDA owner resolution.

This is an instrumentation draft, not a memory estimator or a certificate. It
never rounds owner sizes, infers allocator blocks, skips observations, or holds
a tensor/storage alive. Its resolver is the unchanged strict native-snapshot
resolver. No existing ledger uses this class.

Reuse is restricted to consecutive observations of the same constructor phase
whose full backing owners, view descriptors, and continuously live base-tensor
representatives are unchanged. Lifecycle events always take a new snapshot.
The caller must still emit every returned observation/delta and synchronize at
its normal observation boundary. This does not measure opaque allocations not
included in that declared boundary, or remove any retirement-receipt checks.
"""

import copy
import gc
import hashlib
import json
import re
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.kv_allocator_ledger import resolve_snapshot, tensor_owners

_CONSTRUCTOR_PHASE = re.compile(r"prefill_layer_(0|[1-9][0-9]*)_constructor_workspace\Z")


def resolution_mapping_sha256(resolved: Mapping[str, Any]) -> str:
    """Hash complete actual physical mapping, independently of observation metadata."""
    payload = {key: value for key, value in resolved.items() if key != "snapshot_cache"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _storage(tensor: torch.Tensor) -> tuple[int, int, int]:
    storage = tensor.untyped_storage()
    return storage._cdata, storage.data_ptr(), storage.nbytes()


@dataclass(frozen=True)
class OwnerMetadata:
    """Transient metadata; only weak references to ``representative`` are kept.

    Injection is solely for CPU fixtures. Production uses :func:`cuda_metadata`,
    which rejects CPU, cross-device, non-strided and non-tensor inputs. ``view``
    includes storage offset, shape, strides and dtype, not just logical bytes.
    """

    representative: Any
    device: int
    storage_cdata: int
    pointer: int
    nbytes: int
    view: tuple


def storage_representative(tensor: torch.Tensor) -> torch.Tensor:
    """Read-only ancestry helper, also testable with real CPU tensor views."""
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("owner metadata requires real strided tensors")
    identity = _storage(tensor)
    representative, seen = tensor, {id(tensor)}
    while isinstance(representative._base, torch.Tensor):
        base = representative._base
        if id(base) in seen:
            raise ValueError("cyclic tensor-base ancestry")
        seen.add(id(base))
        if base.device != tensor.device or base.layout != torch.strided:
            raise ValueError("tensor-base ancestry changed device or layout")
        if _storage(base) != identity:
            break
        representative = base
    return representative


def cuda_metadata(tensor: torch.Tensor, *, device: int) -> OwnerMetadata:
    """Select the deepest live ``._base`` carrying the same complete backing."""
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("production owner metadata requires real strided CUDA tensors")
    if tensor.device.type != "cuda" or tensor.device.index != device:
        raise ValueError("production owner metadata requires the declared CUDA device")
    identity = _storage(tensor)
    representative = storage_representative(tensor)
    return OwnerMetadata(
        representative=representative,
        device=device,
        storage_cdata=identity[0],
        pointer=identity[1],
        nbytes=identity[2],
        view=(
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            tensor.element_size(),
        ),
    )


def _validated_metadata(metadata: OwnerMetadata, *, device: int) -> tuple:
    if not isinstance(metadata, OwnerMetadata):
        raise ValueError("owner metadata provider returned an invalid record")
    if _int(metadata.device, "owner device") != device:
        raise ValueError("owner metadata changed the declared device")
    cdata = _int(metadata.storage_cdata, "storage cdata", minimum=1)
    pointer = _int(metadata.pointer, "storage pointer")
    nbytes = _int(metadata.nbytes, "storage nbytes")
    if nbytes and pointer == 0:
        raise ValueError("nonempty storage has a null pointer")
    if not isinstance(metadata.view, tuple) or len(metadata.view) != 5:
        raise ValueError("complete storage-offset/shape/stride/dtype metadata is required")
    offset, shape, stride, dtype, element_size = metadata.view
    _int(offset, "storage offset")
    _int(element_size, "element size", minimum=1)
    if not isinstance(shape, tuple) or not isinstance(stride, tuple) or len(shape) != len(stride):
        raise ValueError("shape and strides must be equal-length tuples")
    for value in shape:
        _int(value, "shape extent")
    for value in stride:
        _int(value, "stride")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype must be explicit")
    # Even fixtures must not hide a view extending beyond its supplied backing.
    if all(shape):
        end = offset + sum((n - 1) * s for n, s in zip(shape, stride, strict=True)) + 1
        if end * element_size > nbytes:
            raise ValueError("view exceeds its complete storage backing")
    try:
        weakref.ref(metadata.representative)
    except TypeError as error:
        raise ValueError("owner representative must support weak references") from error
    return (device, cdata, pointer, nbytes)


class LiveOwnerSnapshotCache:
    """Opt-in consecutive constructor-observation reuse; disabled by default.

    ``owner_metadata_fn`` and ``snapshot_fn`` permit synthetic CPU testing,
    without weakening the default production CUDA-only metadata checks. The
    snapshot function is called without arguments. Returned mappings and the
    observation journal are copies: callers cannot mutate cached evidence.
    This class allocates no GPU tensor or workspace.
    """

    def __init__(
        self,
        *,
        device: int,
        enabled: bool = False,
        owner_metadata_fn: Callable[[Any], OwnerMetadata] | None = None,
        snapshot_fn: Callable[[], Any] | None = None,
    ):
        self.device = _int(device, "device")
        if type(enabled) is not bool:
            raise ValueError("enabled must be an explicit boolean")
        self.enabled = enabled
        self._metadata = owner_metadata_fn or (
            lambda tensor: cuda_metadata(tensor, device=self.device)
        )
        self._snapshot = snapshot_fn or torch.cuda.memory_snapshot
        self._previous = None
        self._journal: list[dict[str, Any]] = []
        self._snapshot_calls = 0

    @property
    def observations(self) -> tuple[dict[str, Any], ...]:
        """Every attempted observation, including failed resolutions, in order."""
        return tuple(copy.deepcopy(self._journal))

    def invalidate(self) -> None:
        """Explicit invalidation never removes prior observation records."""
        self._previous = None

    def resolve(self, tensors: Iterable[Any], phase: str) -> dict[str, Any]:
        """Return a complete strict resolution plus ``snapshot_cache`` provenance."""
        index = len(self._journal)
        provenance: dict[str, Any] = dict(
            observation_index=index,
            phase=phase if type(phase) is str else f"invalid_type:{type(phase).__name__}",
            enabled=self.enabled,
            scope="consecutive_constructor_observations_only",
            prior_weak_representatives_alive=False,
            observation_skipped=False,
            allocator_bytes_inferred=False,
        )
        try:
            if type(phase) is not str or not phase or len(phase) > 256:
                raise ValueError("phase must be bounded and nonempty")
            owners, views, representatives = set(), set(), {}
            for tensor in tensors:
                metadata = self._metadata(tensor)
                storage_identity = _validated_metadata(metadata, device=self.device)
                if not metadata.nbytes:
                    continue
                representative = metadata.representative
                representative_id = id(representative)
                representative_record = (storage_identity, weakref.ref(representative))
                old = representatives.get(representative_id)
                if old is not None and old[0] != storage_identity:
                    raise ValueError("one representative reports inconsistent storage identities")
                representatives[representative_id] = representative_record
                owners.add((metadata.pointer, metadata.nbytes))
                views.add((storage_identity, metadata.view))
            # No Tensor or UntypedStorage enters the retained state. Transient
            # loop variables live only for this call, like the caller's inputs.
        except Exception as error:
            self._record_failure(provenance, error)
            raise
        try:
            fingerprint_payload = sorted(views)
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, separators=(",", ":")).encode()
            ).hexdigest()
            previous = self._previous
            alive = previous is not None and all(
                reference() is not None for _, reference in previous["representatives"].values()
            )
            provenance["prior_weak_representatives_alive"] = bool(alive)
            same_representatives = (
                bool(alive) and representatives.keys() == previous["representatives"].keys()
            )
            if same_representatives:
                same_representatives = all(
                    identity == previous["representatives"][key][0]
                    and reference() is previous["representatives"][key][1]()
                    for key, (identity, reference) in representatives.items()
                )
            reusable_phase = _CONSTRUCTOR_PHASE.fullmatch(phase) is not None
            if not self.enabled:
                reason = "disabled"
            elif not reusable_phase:
                reason = "mandatory_lifecycle_snapshot"
            elif previous is None:
                reason = "no_cached_snapshot"
            elif phase != previous["phase"]:
                reason = "phase_transition"
            elif not alive:
                reason = "previous_owner_died"
            elif not same_representatives:
                reason = "owner_representative_or_storage_changed"
            elif views != previous["views"] or owners != previous["owners"]:
                reason = "owner_or_view_set_changed"
            else:
                reason = "identical_continuously_live_constructor_owners"
            reused = reason == "identical_continuously_live_constructor_owners"
            if reused:
                resolved = previous["resolved"]
                source_index = previous["source_index"]
                mapping_sha = previous["mapping_sha"]
            else:
                self._snapshot_calls += 1
                resolved = resolve_snapshot(self._snapshot(), owners, device=self.device)
                source_index = index
                mapping_sha = resolution_mapping_sha256(resolved)
            provenance.update(
                outcome="resolved",
                mode="reused_actual_resolution" if reused else "fresh_native_snapshot",
                reason=reason,
                source_observation_index=source_index,
                snapshot_calls=self._snapshot_calls,
                owner_view_fingerprint_sha256=fingerprint,
                resolution_mapping_sha256=mapping_sha,
                weak_representative_count=len(representatives),
            )
            self._previous = (
                dict(
                    phase=phase,
                    owners=owners,
                    views=views,
                    representatives=representatives,
                    resolved=resolved,
                    source_index=source_index,
                    mapping_sha=mapping_sha,
                )
                if self.enabled and reusable_phase
                else None
            )
            # Internal resolutions are never exposed or mutated. Exactly one
            # deep copy protects the caller boundary; scalar provenance needs
            # only shallow copies. Reused mappings retain their original hash.
            self._journal.append(provenance.copy())
            return {**copy.deepcopy(resolved), "snapshot_cache": provenance.copy()}
        except Exception as error:
            self._record_failure(provenance, error)
            raise

    def _record_failure(self, provenance: Mapping[str, Any], error: Exception) -> None:
        self.invalidate()
        self._journal.append(
            {
                **copy.deepcopy(provenance),
                "outcome": "failed_closed",
                "error_type": type(error).__name__,
                "snapshot_calls": self._snapshot_calls,
            }
        )


def selftest(device: str = "cuda") -> dict[str, Any]:
    """Bounded GPU operator check against a new strict snapshot on *every* call.

    Fixture tensors are permitted here, not in the resolver itself. This is an
    actual-native-allocator check, not a model, peak or timing certificate.
    A bounded failure to observe physical pointer reuse fails the self-test
    rather than claiming that unexercised control passed.
    """
    requested_device = torch.device(device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("snapshot-cache selftest requires an available CUDA device")
    device_index = (
        torch.cuda.current_device() if requested_device.index is None else requested_device.index
    )
    actual_device = torch.device("cuda", device_index)
    cache = LiveOwnerSnapshotCache(device=device_index, enabled=True)
    results: list[dict[str, Any]] = []

    def observe(tensors, phase, label, *, expected_mode=None):
        torch.cuda.synchronize(actual_device)
        actual = cache.resolve(tensors, phase)
        expected = resolve_snapshot(
            torch.cuda.memory_snapshot(),
            tensor_owners(tensors, device=device_index),
            device=device_index,
        )
        physical = {key: value for key, value in actual.items() if key != "snapshot_cache"}
        provenance = actual["snapshot_cache"]
        result = dict(
            label=label,
            all_mappings_equal=physical == expected,
            mapping_hash_matches=(
                provenance["resolution_mapping_sha256"] == resolution_mapping_sha256(expected)
            ),
            mode=provenance["mode"],
            mode_matches=expected_mode is None or provenance["mode"] == expected_mode,
        )
        results.append(result)
        return actual

    phase = "prefill_layer_0_constructor_workspace"
    owner = torch.empty(128, dtype=torch.bfloat16, device=actual_device)
    observe([owner], phase, "first", expected_mode="fresh_native_snapshot")
    observe([owner], phase, "identical", expected_mode="reused_actual_resolution")
    observe([owner[:64], owner[64:]], phase, "new_alias_set", expected_mode="fresh_native_snapshot")
    observe(
        [owner[64:], owner[:64], owner[:64]],
        phase,
        "equivalent_alias_set",
        expected_mode="reused_actual_resolution",
    )
    extra = torch.empty(127, dtype=torch.bfloat16, device=actual_device)
    observe([owner, extra], phase, "owner_added", expected_mode="fresh_native_snapshot")
    observe([owner], phase, "owner_removed", expected_mode="fresh_native_snapshot")
    del extra
    old_pointer = owner.untyped_storage().data_ptr()
    dead_owner = weakref.ref(owner)
    del owner
    gc.collect()
    torch.cuda.synchronize(actual_device)
    # Hold unsuccessful small fixtures so allocation can progress through other
    # inactive blocks instead of repeatedly choosing the same nonmatching one.
    nonmatching, replacement, pointer_reused = [], None, False
    for _ in range(64):
        candidate = torch.empty(128, dtype=torch.bfloat16, device=actual_device)
        if candidate.untyped_storage().data_ptr() == old_pointer:
            replacement, pointer_reused = candidate, True
            break
        nonmatching.append(candidate)
    if replacement is None:
        replacement = torch.empty(128, dtype=torch.bfloat16, device=actual_device)
    del candidate
    observe([replacement], phase, "dead_owner_pointer_reuse", expected_mode="fresh_native_snapshot")
    old_owner_released = dead_owner() is None
    replacement.resize_(256)
    observe([replacement], phase, "in_place_resize", expected_mode="fresh_native_snapshot")
    for lifecycle in ("prefill_layer_0_KV_published", "prefill_layer_0_native_retired"):
        for repeat in range(2):
            observe(
                [replacement],
                lifecycle,
                f"{lifecycle}_{repeat}",
                expected_mode="fresh_native_snapshot",
            )
    observe(
        [replacement], phase, "constructor_after_lifecycle", expected_mode="fresh_native_snapshot"
    )
    reuse_count = sum(row["mode"] == "reused_actual_resolution" for row in results)
    all_equal = all(row["all_mappings_equal"] for row in results)
    return dict(
        all_passed=(
            all_equal
            and pointer_reused
            and old_owner_released
            and reuse_count > 0
            and all(row["mapping_hash_matches"] and row["mode_matches"] for row in results)
        ),
        all_mappings_equal=all_equal,
        cache_reuse_count=reuse_count,
        pointer_reuse_observed=pointer_reused,
        previous_owner_released=old_owner_released,
        observation_count=len(cache.observations),
        retained_every_observation=len(cache.observations) == len(results),
        checks=results,
        scope="bounded_actual_allocator_operator_selftest_not_model_or_peak_certificate",
    )
