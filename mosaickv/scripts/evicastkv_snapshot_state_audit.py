#!/usr/bin/env python3
"""Bounded streaming audit of two existing compact allocator snapshots.

Never decode the semantic DAG. Keep scalar sizes/depths and sparse BlockInfo
state summaries, then select final_allocator_snapshot.segments. This only
requalifies the pending-state compatibility of these original audited inputs;
it does not certify an exact KV peak or an unaudited execution.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import json
import math
import resource
import signal
import subprocess
import time
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/ownership")
PRODUCER_SHA = "84900a6a402e97c7d58499a722c159cfc86e46c7"
PARENT_GATE_SHA = "59d225ef1ce3e1e9fa3b865eebbd07aa1210b41e698e22114c1b491991c02ade"
PROTOCOL_SHA = "dc98aa6275472f1ecbff5275d3351578f52228662d03015b0776cd1bdaead637"
CASE_SHA = (
    {
        "result.json": "dd6b2066b27ab4cd69f05672fbec3e9c7732176e2f5d1763c2c3d21ab38dd1dd",
        "trace_receipt.json": "35dccd64bc172cef466e51c6108da7100b26741a5271cd3207f4f81988ae5488",
        "trace_raw_receipt.json": (
            "63a66e7c24a9fee9c29bc3ba0167f09beed51880617ed9a53a0e8c467fb9b338"
        ),
    },
    {
        "result.json": "b625aaee1de3fbff1318f27c56d5a9996cdf53d57df92c293807ca3d9c02af2e",
        "trace_receipt.json": "3e9c2414262a3a7a0ecb44933ea6f2f87bffa4df6693e7a309885fabea00c2dd",
        "trace_raw_receipt.json": (
            "fe18dd258ff042d0abf3c161016b9bda10812a3509afee45cf9754b32d1327b7"
        ),
    },
)
CODEC = "source_allocator_metadata_DAG_v1"
GATE = "GATE_snapshot_state_compatibility.json"
STATES = ("active_allocated", "active_pending_free", "active_awaiting_free", "inactive")
ALLOWED_FLAGS = frozenset(
    (
        "ambiguous_deferred_owner_timestamp",
        "unclassified_allocation_generations",
        "opaque_allocator_workspace_not_classified",
    )
)
STRUCTURAL_FLAGS = frozenset(
    (
        "free_without_observed_allocation",
        "overlapping_address_without_free_completed",
        "partial_physical_segment_retirement",
        "final_live_backing_snapshot_mismatch",
        "CUDA_OOM_in_request",
        "empty_allocator_capture",
        "request_incomplete",
        "allocator_history_incomplete_or_overwritten",
    )
)


@dataclass(frozen=True)
class Bounds:
    compressed_bytes: int = 512 * 2**20
    encoded_bytes: int = 2 * 2**30
    semantic_bytes: int = 32 * 2**30
    node_buffer_bytes: int = 64 * 2**20
    containers: int = 4_000_000
    items: int = 40_000_000
    depth: int = 96
    sparse_summaries: int = 100_000


BOUNDS = Bounds()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    if Path(path).stat().st_size > 8 * 2**20:
        raise ValueError("bounded control metadata required")
    return json.loads(Path(path).read_text())


class HashedInput:
    def __init__(self, path, bound):
        self.file = Path(path).open("rb")  # noqa: SIM115 -- owner closes in scan's finally.
        self.hash, self.count, self.bound = hashlib.sha256(), 0, bound

    def read(self, size=-1):
        value = self.file.read(size)
        self.count += len(value)
        if self.count > self.bound:
            raise ValueError("compressed archive limit exceeded")
        self.hash.update(value)
        return value

    def close(self):
        self.file.close()


class StreamingJSON:
    def __init__(self, stream, bounds, chunk_bytes):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 2**20:
            raise ValueError("bounded positive parser chunk required")
        self.stream, self.bounds, self.chunk = stream, bounds, chunk_bytes
        self.buffer, self.offset, self.bytes = "", 0, 0
        self.decoder = json.JSONDecoder()

    def more(self):
        self.buffer = self.buffer[self.offset :]
        self.offset = 0
        if len(self.buffer) > self.bounds.node_buffer_bytes:
            raise ValueError("individual node buffer limit exceeded")
        chunk = self.stream.read(self.chunk)
        self.bytes += len(chunk)
        if self.bytes > self.bounds.encoded_bytes:
            raise ValueError("encoded JSON limit exceeded")
        self.buffer += chunk.decode("ascii")
        return bool(chunk)

    def char(self):
        while True:
            while self.offset < len(self.buffer) and self.buffer[self.offset].isspace():
                self.offset += 1
            if self.offset < len(self.buffer):
                return self.buffer[self.offset]
            if not self.more():
                raise ValueError("truncated JSON archive")

    def take(self, expected):
        if self.char() != expected:
            raise ValueError("unexpected JSON punctuation")
        self.offset += 1

    def value(self):
        self.char()
        while True:
            try:
                start = self.offset
                value, end = self.decoder.raw_decode(self.buffer, self.offset)
                if end - start > self.bounds.node_buffer_bytes:
                    raise ValueError("individual node buffer limit exceeded")
                self.offset = end
                return value
            except json.JSONDecodeError:
                if not self.more():
                    raise ValueError("truncated or malformed complete JSON node") from None

    def finish(self):
        if self.buffer[self.offset :].strip():
            raise ValueError("trailing JSON body")
        while self.more():
            if self.buffer.strip():
                raise ValueError("trailing JSON body")


@lru_cache(maxsize=65536, typed=True)
def scalar_size(value):
    if type(value) not in (str, int, float, bool, type(None)) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError("finite JSON scalar required")
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))


def scan(path, receipt, *, bounds=BOUNDS, chunk_bytes=2**20):
    """One compressed pass; no expanded metadata objects or output artifacts."""
    if any(
        type(getattr(bounds, name)) is not int
        or not 1 <= getattr(bounds, name) <= getattr(BOUNDS, name)
        for name in Bounds.__dataclass_fields__
    ):
        raise ValueError("only equal or smaller registered bounds are permitted")
    path = Path(path)
    before = path.stat()
    if before.st_size > bounds.compressed_bytes:
        raise ValueError("compressed archive limit exceeded")
    started = time.monotonic()
    raw = HashedInput(path, bounds.compressed_bytes)
    summary, special, flaglists = {}, {}, {}
    sizes, depths, uses, empty_lists = array("Q"), array("B"), array("I"), array("B")
    count = items = 0
    last = None
    outer = {}

    def reference(value):
        if isinstance(value, dict):
            if (
                set(value) != {"$ref"}
                or type(value["$ref"]) is not int
                or not 0 <= value["$ref"] < count
            ):
                raise ValueError("backward in-range reference required")
            return value["$ref"]
        if type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("finite JSON scalar or explicit backward reference required")
        scalar_size(value)
        return None

    try:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            parser = StreamingJSON(compressed, bounds, chunk_bytes)
            parser.take("{")
            while True:
                key = parser.value()
                if type(key) is not str or key in outer:
                    raise ValueError("unique outer string keys required")
                parser.take(":")
                if key != "nodes":
                    outer[key] = parser.value()
                else:
                    parser.take("[")
                    while parser.char() != "]":
                        node = parser.value()
                        if (
                            type(node) is not list
                            or len(node) != 2
                            or node[0] not in ("d", "l")
                            or type(node[1]) is not list
                        ):
                            raise ValueError("complete dictionary/list DAG node required")
                        kind, children = node
                        items += len(children)
                        if count >= bounds.containers or items > bounds.items:
                            raise ValueError("container/item limit exceeded")
                        hist, current, depth = [0] * 8, {}, 1
                        semantic = 2 + max(0, len(children) - 1)
                        for child in children:
                            if kind == "d":
                                if (
                                    type(child) is not list
                                    or len(child) != 2
                                    or type(child[0]) is not str
                                    or child[0] in current
                                ):
                                    raise ValueError("unique complete dictionary keys required")
                                name, value = child
                                current[name] = value
                                semantic += scalar_size(name) + 1
                            else:
                                value = child
                            ref = reference(value)
                            if ref is None:
                                semantic += scalar_size(value)
                            else:
                                uses[ref] += 1
                                semantic += sizes[ref]
                                depth = max(depth, depths[ref] + 1)
                                if ref in summary:
                                    for index, value in enumerate(summary[ref]):
                                        hist[index] += value
                            if semantic > bounds.semantic_bytes or depth > bounds.depth:
                                raise ValueError("semantic size/depth limit exceeded")
                        if semantic > bounds.semantic_bytes:
                            raise ValueError("semantic size limit exceeded")
                        if kind == "d":
                            if {"address", "size", "state"} <= current.keys():
                                if not {"requested_size", "frames"} <= current.keys():
                                    raise ValueError("complete allocator BlockInfo fields required")
                                state, requested = current["state"], current["requested_size"]
                                if (
                                    state not in STATES
                                    or type(requested) is not int
                                    or requested < 0
                                ):
                                    raise ValueError("unknown allocator state or requested size")
                                index = STATES.index(state)
                                hist[index] += 1
                                hist[index + 4] += requested
                            if {"segments", "device_traces"} <= current.keys() or {
                                "complete_request",
                                "allocator_history_complete",
                                "blocks_exact_certification",
                            } <= current.keys():
                                special[count] = current
                        elif (
                            children
                            and len(children) <= 32
                            and all(type(v) is str for v in children)
                            and (ALLOWED_FLAGS | STRUCTURAL_FLAGS).intersection(children)
                        ):
                            flaglists[count] = children
                        if any(hist):
                            summary[count] = tuple(hist)
                        if len(summary) > bounds.sparse_summaries or any(
                            value > 2**63 for value in hist
                        ):
                            raise ValueError("sparse state summary limit exceeded")
                        sizes.append(semantic)
                        depths.append(depth)
                        uses.append(0)
                        empty_lists.append(int(kind == "l" and not children))
                        count += 1
                        last = current if kind == "d" else None
                        if parser.char() != ",":
                            break
                        parser.take(",")
                        if parser.char() == "]":
                            raise ValueError("trailing node separator is not complete JSON")
                    parser.take("]")
                    outer[key] = "streamed"
                if parser.char() != ",":
                    break
                parser.take(",")
            parser.take("}")
            parser.finish()
    finally:
        raw.close()
    if (
        set(outer) != {"codec", "nodes", "root", "statistics"}
        or outer["codec"] != CODEC
        or type(outer["root"]) is not dict
        or set(outer["root"]) != {"$ref"}
        or type(outer["root"]["$ref"]) is not int
        or outer["root"]["$ref"] != count - 1
        or not last
    ):
        raise ValueError("exact complete rooted metadata DAG required")
    expected = dict(containers=count, items=items, semantic_JSON_bytes=sizes[-1])
    if (
        type(outer["statistics"]) is not dict
        or any(type(value) is not int for value in outer["statistics"].values())
        or outer["statistics"] != expected
        or expected != receipt["codec_statistics"]
    ):
        raise ValueError("exact DAG statistics/semantic size receipt mismatch")
    if any(value == 0 for value in uses[:-1]):
        raise ValueError("orphan metadata node outside rooted DAG")
    if parser.bytes != receipt["encoded_JSON_bytes"] or raw.hash.hexdigest() != receipt["sha256"]:
        raise ValueError("archive compressed hash or encoded-size receipt mismatch")
    snapshot = special[last["final_allocator_snapshot"]["$ref"]]
    hist = summary.get(snapshot["segments"]["$ref"], (0,) * 8)
    capture = special[last["capture_metadata"]["$ref"]]
    flags_ref = capture["blocks_exact_certification"]["$ref"]
    flags = [] if empty_lists[flags_ref] else flaglists[flags_ref]
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("immutable archive changed while streaming")
    result = dict(
        compressed_sha256=raw.hash.hexdigest(),
        compressed_bytes=raw.count,
        encoded_JSON_bytes=parser.bytes,
        codec_statistics=expected,
        state_counts=dict(zip(STATES, hist[:4], strict=True)),
        requested_bytes_by_state=dict(zip(STATES, hist[4:], strict=True)),
        capture_complete=capture["complete_request"],
        allocator_history_complete=capture["allocator_history_complete"],
        capture_flags=flags,
        whole_semantic_DAG_expanded=False,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        wall_seconds=time.monotonic() - started,
        baseline_state_histogram_observed=False,
        final_snapshot_ref=last["final_allocator_snapshot"]["$ref"],
        baseline_pending_exclusion_inputs=dict(
            complete_request=capture["complete_request"] is True,
            complete_history=capture["allocator_history_complete"] is True,
            no_unseeded_free_flag="free_without_observed_allocation" not in flags,
            no_structural_flag=not bool(STRUCTURAL_FLAGS.intersection(flags)),
            observed_final_pending_zero=not (hist[1] or hist[2]),
        ),
    )
    scalar_size.cache_clear()
    gc.collect()
    return result


def baseline_flush_proof():
    relative = "mosaickv/src/mosaickv/measurements/kv_timeline.py"
    raw = subprocess.check_output(["git", "show", f"{PRODUCER_SHA}:{relative}"], cwd=SOURCE)
    tree, proofs = ast.parse(raw), {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {
            "CudaOwnershipSession",
            "AllocationTimeline",
        }:
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and (cls.name, node.name) in {
                    ("CudaOwnershipSession", "__enter__"),
                    ("CudaOwnershipSession", "_flush_cuda_events"),
                    ("AllocationTimeline", "ingest"),
                }:
                    proofs[f"{cls.name}.{node.name}"] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest()
    text = raw.decode()
    if len(proofs) != 3 or not all(
        value in text
        for value in (
            "self._flush_cuda_events()",
            "torch.cuda.synchronize()",
            "tensor = torch.empty(1,",
            "free_without_observed_allocation",
        )
    ):
        raise ValueError("original baseline flush and missing-seed error source changed")
    return dict(
        producer_source_sha=PRODUCER_SHA,
        git_path=relative,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        reviewed_function_AST_sha256s=proofs,
        baseline_histogram_was_not_archived=True,
        logical_exclusion=(
            "With complete allocator history, an omitted starting pending block must "
            "either remain pending at the final snapshot or retire through an "
            "unseeded free_completed event. Both alternatives are excluded only "
            "when final pending counts are zero and structural flags are absent."
        ),
        source_flush_is_supporting_not_observed_baseline_histogram=True,
    )


def validate_receipts(directory, result, receipt):
    """Bind durable raw receipt and completed semantic receipt without DAG decode."""
    archive = directory / "allocator_trace.json.gz"
    raw_path = directory / "trace_raw_receipt.json"
    raw = read(raw_path)
    limits = dict(
        compressed_bytes=BOUNDS.compressed_bytes,
        encoded_JSON_bytes=BOUNDS.encoded_bytes,
        semantic_JSON_bytes=BOUNDS.semantic_bytes,
        containers=BOUNDS.containers,
        items=BOUNDS.items,
        depth=BOUNDS.depth,
        hash_cache_bytes=32 * 2**20,
    )
    keys = (
        "schema",
        "path",
        "sha256",
        "source_sha",
        "encoded_JSON_bytes",
        "codec_statistics",
        "metadata_codec_limits",
    )
    if (
        receipt.get("schema") != "streaming_lossless_BF16_whole_request_ownership_v15"
        or receipt.get("metadata_codec_limits") != limits
        or receipt.get("source_sha") != PRODUCER_SHA
        or receipt.get("semantic_hashes_complete") is not True
        or receipt.get("complete_final_snapshot_preserved") is not True
        or Path(receipt["path"]).resolve() != archive.resolve()
        or Path(receipt["raw_receipt_path"]).resolve() != raw_path.resolve()
        or receipt.get("raw_receipt_sha256") != sha(raw_path)
        or result["outputs"][raw_path.name] != sha(raw_path)
        or raw.get("semantic_hashes_complete") is not False
        or raw.get("complete_final_snapshot_preserved") is not True
        or raw.get("compressed_bytes") != archive.stat().st_size
        or any(raw.get(key) != receipt[key] for key in keys)
    ):
        raise ValueError("durable original archive receipt binding changed")
    return raw_path


def run(root, *, cpu_gate_root):
    from evicastkv_review_followups import cpu_gate

    from mosaickv.evaluation.fox_campaign import clean_source, write_once

    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    gate_path, protocol_path = (
        PARENT / "GATE_streaming_lossless_ownership.json",
        PARENT / "protocol.json",
    )
    gate, protocol = read(gate_path), read(protocol_path)
    if (
        sha(gate_path) != PARENT_GATE_SHA
        or sha(protocol_path) != PROTOCOL_SHA
        or gate["source_sha"] != PRODUCER_SHA
        or protocol["source_sha"] != PRODUCER_SHA
        or gate["complete_numerical_and_trace_controls"] is not True
        or gate["failures"]
    ):
        raise ValueError("exact completed original V16 source capture gate required")
    bindings = {str(p): sha(p) for p in (gate_path, protocol_path)}
    summaries, errors, archives = [], [], {}
    for index in range(2):
        directory = PARENT / f"workers/{index}"
        result_path, receipt_path = directory / "result.json", directory / "trace_receipt.json"
        result, receipt = read(result_path), read(receipt_path)
        bindings.update({str(p): sha(p) for p in (result_path, receipt_path)})
        try:
            if any(sha(directory / name) != expected for name, expected in CASE_SHA[index].items()):
                raise ValueError("registered original case control hashes changed")
            if (
                result["source_sha"] != PRODUCER_SHA
                or result["failures"]
                or result["complete_trace_with_final_snapshot_archived"] is not True
                or result["all_32_128_guards_pass"] is not True
                or receipt["source_sha"] != PRODUCER_SHA
                or receipt["semantic_hashes_complete"] is not True
                or result["outputs"][receipt_path.name] != sha(receipt_path)
                or result["outputs"]["allocator_trace.json.gz"] != receipt["sha256"]
            ):
                raise ValueError("complete bound original producer/archive receipt required")
            raw_path = validate_receipts(directory, result, receipt)
            bindings[str(raw_path)] = sha(raw_path)
            summary = scan(directory / "allocator_trace.json.gz", receipt)
            archives[str(directory / "allocator_trace.json.gz")] = summary["compressed_sha256"]
            summaries.append(dict(index=index, **summary))
            if (
                summary["capture_complete"] is not True
                or summary["allocator_history_complete"] is not True
                or set(summary["capture_flags"]) - ALLOWED_FLAGS
                or any(
                    summary["state_counts"][key]
                    for key in ("active_pending_free", "active_awaiting_free")
                )
            ):
                raise ValueError("pending-state or structural omission cannot be requalified")
        except Exception as error:
            errors.append(dict(index=index, error=f"{type(error).__name__}: {error}"))
    good = not errors and len(summaries) == 2
    report = dict(
        status="requalified_original_two_capture_pending_state_compatibility"
        if good
        else "blocked_snapshot_state_compatibility",
        source_sha=revision,
        producer_source_sha=PRODUCER_SHA,
        CPU_gate=cpu,
        input_bindings=bindings,
        archive_bindings=archives,
        snapshots=summaries,
        errors=errors,
        baseline_flush_source_proof=baseline_flush_proof(),
        original_two_audited_capture_bounds_requalified=good,
        baseline_state_histogram_observed=False,
        baseline_pending_omission_logically_excluded_for_these_captures=good,
        no_role_or_numerical_override=True,
        no_semantic_DAG_expansion=True,
        no_new_GPU_inference=True,
        CLI_resource_limits=dict(address_space_bytes=2 * 2**30, CPU_seconds=240, wall_seconds=300),
        independent_quality_N=0,
        exact_KV_peak_certified=False,
        unaudited_timing_variant_certified=False,
        fresh_confirmation_authorized=False,
        superiority_established=False,
    )
    if any(sha(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("original control metadata changed during audit")
    write_once(Path(root) / GATE, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    signal.alarm(300)
    value = run(args.root, cpu_gate_root=args.cpu_gate_root)
    print(json.dumps(dict(status=value["status"], gate=str(args.root / GATE))))
    if value["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
