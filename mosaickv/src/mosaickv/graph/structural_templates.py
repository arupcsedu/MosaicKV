"""Bounded engine-scoped templates for input-independent structural edges.

Only CPU indices/weights are retained. Query/K/V features, predictions, model
owners and media tensors are never cached. Every lookup binds all structural
metadata, positions, compatibility controls and the eligible-node set.
"""

import hashlib
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, astuple, fields

import numpy as np

from mosaickv.graph.types import BlockEvidenceMetadata, EdgeType

_bank = ContextVar("evicastkv_structural_template_bank", default=None)
_enabled = ContextVar("evicastkv_structural_template_activation", default=False)
_direct_metadata = ContextVar("evicastkv_structural_template_direct_metadata", default=False)
_METADATA_FIELDS = tuple(f.name for f in fields(BlockEvidenceMetadata))
STRUCTURAL = frozenset(EdgeType) - {
    EdgeType.SEMANTIC_SIMILARITY,
    EdgeType.ATTENTION_COACTIVATION,
}


def _metadata_tuple(metadata, direct):
    # Only this frozen, scalar/tuple-only schema is eligible. Unknown subclasses
    # keep the existing generic dataclass semantics. JSON consumes the same
    # values synchronously; no metadata owner is retained by the template bank.
    if direct and type(metadata) is BlockEvidenceMetadata:
        return tuple(getattr(metadata, name) for name in _METADATA_FIELDS)
    return astuple(metadata)


def cache_key(nodes, config, eligible, execution, *, direct_metadata=None):
    direct = _direct_metadata.get() if direct_metadata is None else direct_metadata
    if type(direct) is not bool:
        raise ValueError("explicit metadata-key serialization policy required")
    data = dict(
        config=asdict(config),
        eligible=None if eligible is None else sorted(eligible),
        execution=execution,
        nodes=[
            (
                n.node_id,
                n.block.layer,
                n.block.kv_head,
                n.block.modality.value,
                n.block.original_logical_positions,
                _metadata_tuple(n.evidence, direct),
            )
            for n in nodes
        ],
    )
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class TemplateBank:
    def __init__(self, maximum_bytes=4 * 2**20):
        if type(maximum_bytes) is not int or maximum_bytes < 1024:
            raise ValueError("positive bounded structural-template CPU budget required")
        self.maximum_bytes = maximum_bytes
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.skipped = 0
        self.maximum_resident_bytes = 0

    @property
    def resident_bytes(self):
        return sys.getsizeof(self.entries) + sum(v[1] for v in self.entries.values())

    def get(self, key):
        item = self.entries.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return item[0]

    def put(self, key, finalized, node_count, maximum_neighbors):
        rows, columns, weights, types = finalized
        if not (len(rows) == len(columns) == len(weights) == len(types)):
            raise ValueError("aligned finalized structural edges required")
        if any(t not in STRUCTURAL for t in types):
            raise ValueError("dynamic query/semantic edges cannot enter a structural template")
        segments = []
        for edge_type in sorted(set(types), key=str):
            ids = np.asarray([i for i, t in enumerate(types) if t == edge_type], dtype=np.int64)
            r, c = (np.asarray(v, dtype=np.int64)[ids].copy() for v in (rows, columns))
            w = np.asarray(weights, dtype=np.float64)[ids].copy()
            if (
                not bool(np.isfinite(w).all())
                or bool((w <= 0).any())
                or bool(((r < 0) | (r >= node_count) | (c < 0) | (c >= node_count)).any())
                or bool((r == c).any())
                or len(set(zip(r.tolist(), c.tolist(), strict=True))) != len(r)
                or bool((np.bincount(r, minlength=node_count) > maximum_neighbors).any())
            ):
                raise ValueError("valid sparse bounded-degree structural edges required")
            for v in (r, c, w):
                v.setflags(write=False)
            segments.append((r, c, w, edge_type))
        segments = tuple(segments)
        size = sys.getsizeof(key) + sys.getsizeof(segments)
        size += sum(
            sys.getsizeof(segment) + sum(sys.getsizeof(a) for a in segment[:3])
            for segment in segments
        )
        # Account for the retained (segments,size) tuple as well.
        size += sys.getsizeof((segments, size)) + sys.getsizeof(size)
        if size + sys.getsizeof(OrderedDict()) > self.maximum_bytes:
            self.skipped += 1
            return segments
        self.entries[key] = (segments, size)
        self.entries.move_to_end(key)
        while self.resident_bytes > self.maximum_bytes and self.entries:
            self.entries.popitem(last=False)
            self.evictions += 1
        self.maximum_resident_bytes = max(self.maximum_resident_bytes, self.resident_bytes)
        return segments

    def report(self):
        return dict(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            oversized_templates_skipped=self.skipped,
            resident_CPU_array_and_shallow_container_bytes=self.resident_bytes,
            maximum_resident_CPU_array_and_shallow_container_bytes=self.maximum_resident_bytes,
            maximum_registered_bytes=self.maximum_bytes,
            persistent_GPU_bytes=0,
            retained_KV_or_query_tensors=0,
            retained_predictions_or_references=False,
            whole_CPU_heap_peak_certified=False,
        )

    def close(self):
        self.entries.clear()


def current():
    return _bank.get() if _enabled.get() else None


@contextmanager
def engine_scope(maximum_bytes=4 * 2**20):
    if _bank.get() is not None:
        raise RuntimeError("structural-template engine scopes cannot nest")
    bank = TemplateBank(maximum_bytes)
    token = _bank.set(bank)
    try:
        yield bank
    finally:
        bank.close()
        _bank.reset(token)


@contextmanager
def activate(enabled, *, direct_metadata=False):
    if (
        type(enabled) is not bool
        or type(direct_metadata) is not bool
        or (direct_metadata and not enabled)
        or (enabled and _bank.get() is None)
    ):
        raise ValueError("explicit template activation requires an engine scope")
    token = _enabled.set(enabled)
    direct_token = _direct_metadata.set(direct_metadata)
    try:
        yield
    finally:
        _direct_metadata.reset(direct_token)
        _enabled.reset(token)
