"""Strict CPU reduction of a bounded native allocator-event capture.

Pool membership is transport metadata, NOT a KV/non-KV classification. Unknown
owners remain unknown. ``free_requested`` does not retire backing storage.
"""

from __future__ import annotations

from collections import Counter


def integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} requires an integer >= {minimum}")
    return value


def pool_id(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("two-part native pool identity required")
    return tuple(integer(part, "pool identity") for part in value)


class NativeEventLedger:
    """Requested-byte generations, ordered by the recorder's native sequence."""

    def __init__(self, baseline=()):
        self.active = {}
        self.generations = Counter()
        self.records = []
        self.actions = Counter()
        self.current_requested_bytes = 0
        self.peak_requested_bytes = 0
        for block in baseline:
            self._allocate(block, sequence=-1, baseline=True)

    def _allocate(self, event, *, sequence, baseline=False):
        device = integer(event["device"], "device")
        address = integer(event["addr"], "address", minimum=1)
        size = integer(event["size"], "requested size", minimum=1)
        pool = pool_id(event["pool"])
        key = device, address
        if key in self.active:
            raise ValueError("overlapping live address without free_completed")
        generation = self.generations[key]
        self.generations[key] += 1
        record = dict(
            device=device,
            addr=address,
            size=size,
            pool=list(pool),
            generation=generation,
            start=sequence,
            end=None,
            free_requested=-1
            if baseline and event.get("state") == "active_awaiting_free"
            else None,
            baseline=baseline,
            semantic_role="unknown",
        )
        self.records.append(record)
        self.active[key] = record
        self.current_requested_bytes += size
        self.peak_requested_bytes = max(self.peak_requested_bytes, self.current_requested_bytes)

    def ingest(self, capture):
        events = capture["events"]
        count = integer(capture["count"], "event count")
        capacity = integer(capture["capacity"], "recorder capacity", minimum=1)
        if (
            capture.get("complete") is not True
            or capture["overflow"] is not False
            or count > capacity
            or count != len(events)
        ):
            raise ValueError("incomplete or overflowed native recorder")
        if self.actions:
            raise ValueError("one complete bounded capture per ledger required")
        for sequence, event in enumerate(events):
            if type(event["sequence"]) is not int or event["sequence"] != sequence:
                raise ValueError("native sequence has a gap, duplicate, or reordering")
            action = event["action"]
            self.actions[action] += 1
            if action == "alloc":
                self._allocate(event, sequence=sequence)
            elif action in {"free_requested", "free_completed"}:
                key = (
                    integer(event["device"], "device"),
                    integer(event["addr"], "address", minimum=1),
                )
                record = self.active.get(key)
                if record is None:
                    raise ValueError("free without observed live generation")
                if pool_id(event["pool"]) != tuple(record["pool"]):
                    raise ValueError("free pool differs from allocation generation")
                if integer(event["size"], "free requested size", minimum=1) != record["size"]:
                    raise ValueError("free requested size differs from allocation generation")
                if action == "free_requested":
                    if record["free_requested"] is not None:
                        raise ValueError("duplicate free_requested")
                    record["free_requested"] = sequence
                else:
                    if record["free_requested"] is None:
                        raise ValueError("free_completed without free_requested")
                    record["end"] = sequence
                    self.current_requested_bytes -= record["size"]
                    del self.active[key]
            elif action in {"segment_free", "segment_unmap"}:
                device = integer(event["device"], "device")
                address = integer(event["addr"], "address", minimum=1)
                stop = address + integer(event["size"], "segment size", minimum=1)
                if any(
                    d == device and a < stop and address < a + record["size"]
                    for (d, a), record in self.active.items()
                ):
                    raise ValueError("segment retired with a live generation")
            elif action == "oom":
                raise ValueError("allocator OOM invalidates the diagnostic")
            elif action not in {"snapshot", "segment_alloc", "segment_map"}:
                raise ValueError("unknown native allocator action")
        return self

    def verify_snapshot(self, blocks):
        expected = {}
        for block in blocks:
            key = integer(block["device"], "device"), integer(block["addr"], "address", minimum=1)
            if key in expected:
                raise ValueError("duplicate final snapshot backing")
            expected[key] = (
                integer(block["size"], "requested size", minimum=1),
                pool_id(block["pool"]),
            )
        actual = {
            key: (record["size"], tuple(record["pool"])) for key, record in self.active.items()
        }
        if expected != actual:
            raise ValueError("native generations do not match all final requested backings")
        return dict(
            exact_final_requested_backing_match=True,
            final_requested_bytes=self.current_requested_bytes,
            observed_peak_requested_bytes=self.peak_requested_bytes,
            generations=len(self.records),
            aliases_counted_once=True,
            actions=dict(self.actions),
            semantic_KV_roles_assigned=0,
            exact_source_KV_peak_certified=False,
            whole_device_memory_certified=False,
            independent_quality_or_superiority_established=False,
        )
