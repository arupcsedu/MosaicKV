"""Lossless, bounded metadata DAG for repeated allocator frame/alias records.

Only serialization changes. Event order, every field, frees, pointer generations,
snapshot entries and unknown roles survive exactly. This is not a role mapper.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

KIND = "source_allocator_metadata_DAG_v1"
LIMITS = dict(
    encoded_JSON_bytes=2 * 2**30,
    compressed_bytes=512 * 2**20,
    semantic_JSON_bytes=8 * 2**30,
    containers=4_000_000,
    items=40_000_000,
    depth=96,
    hash_cache_bytes=32 * 2**20,
)


def explicit_limits(limits=None):
    """Validate scoped bounds without mutating historical eight-GiB defaults.

    A separately registered producer may permit up to 32 GiB of *semantic*
    repeated metadata. Physical JSON/compressed, node/item/depth/cache bounds
    cannot be raised. Callers must bind the chosen complete mapping in receipts.
    """
    if limits is None:
        return dict(LIMITS)
    if (
        type(limits) is not dict
        or set(limits) != set(LIMITS)
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or limits["semantic_JSON_bytes"] > 32 * 2**30
        or any(value > LIMITS[key] for key, value in limits.items() if key != "semantic_JSON_bytes")
    ):
        raise ValueError("complete explicitly bounded metadata codec limits required")
    return dict(limits)


@lru_cache(maxsize=65536)
def _string(value):
    return json.dumps(value, ensure_ascii=True).encode()


def _scalar(value):
    if type(value) not in (str, int, float, bool, type(None)) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError("finite JSON scalar or explicit metadata container required")
    if type(value) is str:
        return _string(value)
    if type(value) is int:
        return str(value).encode()
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if value is None:
        return b"null"
    return json.dumps(value, allow_nan=False).encode()


def _frame_key(value, active=None, depth=0, *, limits=None):
    limits = explicit_limits(limits) if depth == 0 else limits
    active = set() if active is None else active
    if depth > limits["depth"] or id(value) in active:
        raise ValueError("cyclic or over-deep frame metadata is forbidden")
    if isinstance(value, dict):
        active.add(id(value))
        if any(type(key) is not str for key in value):
            raise ValueError("string metadata keys required")
        result = (
            "d",
            tuple(
                (key, _frame_key(item, active, depth + 1, limits=limits))
                for key, item in sorted(value.items())
            ),
        )
        active.remove(id(value))
        return result
    if isinstance(value, (list, tuple)):
        active.add(id(value))
        result = ("l", tuple(_frame_key(item, active, depth + 1, limits=limits) for item in value))
        active.remove(id(value))
        return result
    _scalar(value)
    return (type(value).__name__, _scalar(value))


def _reference(value, upper):
    if type(value) is dict:
        if (
            set(value) != {"$ref"}
            or type(value["$ref"]) is not int
            or not 0 <= value["$ref"] < upper
        ):
            raise ValueError("backward in-range metadata reference required; cycles forbidden")
        return value["$ref"]
    _scalar(value)
    return None


def _shape(document, *, limits=None):
    limits = explicit_limits(limits)
    if (
        type(document) is not dict
        or set(document) != {"codec", "nodes", "root", "statistics"}
        or document["codec"] != KIND
    ):
        raise ValueError("exact lossless metadata DAG schema required")
    nodes = document["nodes"]
    if not isinstance(nodes, list) or len(nodes) > limits["containers"]:
        raise ValueError("bounded metadata container count required")
    sizes, depths, uses = [], [], [0] * len(nodes)
    items = 0
    for index, node in enumerate(nodes):
        if (
            not isinstance(node, list)
            or len(node) != 2
            or node[0] not in ("d", "l")
            or not isinstance(node[1], list)
        ):
            raise ValueError("explicit dictionary/list metadata node required")
        kind, children = node
        items += len(children)
        if items > limits["items"]:
            raise ValueError("bounded metadata item count exceeded")
        size, depth, keys = 2, 1, set()
        for child in children:
            if kind == "d":
                if (
                    not isinstance(child, list)
                    or len(child) != 2
                    or type(child[0]) is not str
                    or child[0] in keys
                ):
                    raise ValueError("unique complete string dictionary keys required")
                key, value = child
                keys.add(key)
                size += len(_scalar(key)) + 1
            else:
                value = child
            ref = _reference(value, index)
            if ref is None:
                size += len(_scalar(value))
            else:
                uses[ref] += 1
                size += sizes[ref]
                depth = max(depth, depths[ref] + 1)
            if size > limits["semantic_JSON_bytes"] or depth > limits["depth"]:
                raise ValueError("bounded semantic expansion/depth exceeded")
        size += max(0, len(children) - 1)
        if size > limits["semantic_JSON_bytes"]:
            raise ValueError("bounded semantic expansion exceeded")
        sizes.append(size)
        depths.append(depth)
    ref = _reference(document["root"], len(nodes))
    if nodes and ref != len(nodes) - 1:
        raise ValueError("complete final root required; orphan nodes forbidden")
    if any(count == 0 for count in uses[:-1]):
        raise ValueError("all metadata nodes must be reachable from the final root")
    total = sizes[ref] if ref is not None else len(_scalar(document["root"]))
    if total > limits["semantic_JSON_bytes"]:
        raise ValueError("bounded semantic expansion exceeded")
    expected = dict(containers=len(nodes), items=items, semantic_JSON_bytes=total)
    if document["statistics"] != expected:
        raise ValueError("lossless metadata size/count declaration differs from decoded structure")
    return sizes, uses


def encode(value, *, limits=None):
    """Intern exact frame content and repeated container identities, never fields."""
    limits = explicit_limits(limits)
    nodes, memo, active, frames = [], {}, set(), {}
    items = 0

    def visit(item, depth=0, *, frame=False):
        nonlocal items
        if depth > limits["depth"]:
            raise ValueError("bounded semantic depth exceeded")
        if not isinstance(item, (dict, list, tuple)):
            _scalar(item)
            return item
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic metadata is forbidden")
        if identity in memo:
            return {"$ref": memo[identity]}
        key = _frame_key(item, limits=limits) if frame else None
        if key is not None and key in frames:
            memo[identity] = frames[key]
            return {"$ref": frames[key]}
        active.add(identity)
        if isinstance(item, dict):
            if any(type(k) is not str for k in item):
                raise ValueError("string metadata keys required")
            children = [[k, visit(v, depth + 1, frame=k == "frames")] for k, v in item.items()]
            kind = "d"
        else:
            children = [visit(v, depth + 1) for v in item]
            kind = "l"
        active.remove(identity)
        items += len(children)
        if len(nodes) >= limits["containers"] or items > limits["items"]:
            raise ValueError("bounded metadata nodes/items exceeded")
        index = len(nodes)
        nodes.append([kind, children])
        memo[identity] = index
        if key is not None:
            frames[key] = index
        return {"$ref": index}

    root = visit(value)
    document = dict(codec=KIND, nodes=nodes, root=root, statistics={})
    # Calculate the declared compact legacy-JSON length without expanding aliases.
    sizes = []
    for kind, children in nodes:
        total = 2 + max(0, len(children) - 1)
        for child in children:
            item = child[1] if kind == "d" else child
            total += (len(_scalar(child[0])) + 1) if kind == "d" else 0
            ref = _reference(item, len(sizes))
            total += sizes[ref] if ref is not None else len(_scalar(item))
            if total > limits["semantic_JSON_bytes"]:
                raise ValueError("bounded semantic expansion exceeded")
        sizes.append(total)
    document["statistics"] = dict(
        containers=len(nodes),
        items=items,
        semantic_JSON_bytes=sizes[root["$ref"]] if isinstance(root, dict) else len(_scalar(root)),
    )
    _shape(document, limits=limits)
    return document


def decode(document, *, limits=None):
    _shape(document, limits=limits)  # Reject bombs/cycles/forward refs before reconstructing.
    values = []
    for kind, children in document["nodes"]:

        def resolve(value):
            return values[value["$ref"]] if isinstance(value, dict) else value

        value = (
            {key: resolve(item) for key, item in children}
            if kind == "d"
            else [resolve(item) for item in children]
        )
        values.append(value)
    root = document["root"]
    return values[root["$ref"]] if isinstance(root, dict) else root


def semantic_digest(value, *, limits=None):
    """Exact fox_campaign.digest JSON/SHA, bounded buffering of shared subtrees."""
    document = encode(value, limits=limits)
    return document_digest(document, limits=limits)


def document_digest(document, path=(), *, limits=None):
    """Hash exact compact legacy JSON; reused small frames avoid Python expansion."""
    return document_digests(document, (path,), limits=limits)[path]


def document_digests(document, paths, *, limits=None):
    """Validate once for several receipt hashes, sharing bounded frame buffers."""
    limits = explicit_limits(limits)
    sizes, uses = _shape(document, limits=limits)
    cache, used = {}, 0

    def chunks(value):
        nonlocal used
        if not isinstance(value, dict):
            yield _scalar(value)
            return
        index = value["$ref"]
        if index in cache:
            yield cache[index]
            return
        kind, children = document["nodes"][index]

        def body():
            yield b"{" if kind == "d" else b"["
            ordered = sorted(children, key=lambda pair: pair[0]) if kind == "d" else children
            for offset, child in enumerate(ordered):
                if offset:
                    yield b","
                if kind == "d":
                    yield _scalar(child[0]) + b":"
                    yield from chunks(child[1])
                else:
                    yield from chunks(child)
            yield b"}" if kind == "d" else b"]"

        if (
            uses[index] > 1
            and sizes[index] <= 65536
            and used + sizes[index] <= limits["hash_cache_bytes"]
        ):
            cached = b"".join(body())
            cache[index] = cached
            used += len(cached)
            yield cached
        else:
            yield from body()

    results = {}
    for path in paths:
        root = document["root"]
        for key in path:
            if not isinstance(root, dict):
                raise ValueError("metadata hash path does not reference a dictionary")
            kind, children = document["nodes"][root["$ref"]]
            if kind != "d" or key not in dict(children):
                raise ValueError("metadata hash path is absent")
            root = dict(children)[key]
        result = hashlib.sha256()
        for chunk in chunks(root):
            result.update(chunk)
        results[path] = result.hexdigest()
    return results
