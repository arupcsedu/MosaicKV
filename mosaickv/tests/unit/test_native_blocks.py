from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mosaickv.backends.native_blocks import (
    native_audit_path,
    require_scratch_path,
    select_native_blocks,
    selected_position_count,
    write_native_audit,
)
from mosaickv.backends.sglang_native_patch import _release_active_request
from mosaickv.backends.vllm_native_plugin import _compress_ready_requests


def test_retention_one_selects_every_block_and_exact_tail() -> None:
    selection = select_native_blocks(35, 16, 1.0)

    assert selection.selected_blocks == (0, 1, 2)
    assert selection.mandatory_blocks == (0, 2)
    assert selection.selected_positions == 35


def test_compressed_selection_keeps_boundaries_and_is_deterministic() -> None:
    first = select_native_blocks(160, 16, 0.4)
    second = select_native_blocks(160, 16, 0.4)

    assert first == second
    assert len(first.selected_blocks) == 4
    assert first.selected_blocks[0] == 0
    assert first.selected_blocks[-1] == 9
    assert first.selected_positions == 64


def test_short_prompt_keeps_both_mandatory_blocks_even_below_ratio() -> None:
    selection = select_native_blocks(17, 16, 0.1)

    assert selection.selected_blocks == (0, 1)
    assert selection.selected_positions == 17


def test_position_accounting_rejects_conflicts() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        selected_position_count(64, 16, (0, 0, 3))
    with pytest.raises(ValueError, match="outside"):
        selected_position_count(64, 16, (0, 4))


def test_audit_writer_hashes_request_id_and_writes_atomically(tmp_path: Path) -> None:
    destination = write_native_audit(
        tmp_path,
        "vllm",
        "request/with unsafe characters",
        {"status": "ok"},
    )

    assert destination == native_audit_path(tmp_path, "vllm", "request/with unsafe characters")
    assert "/" not in destination.name
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "ok"


def test_scratch_guard_accepts_canonical_mount_alias_and_rejects_home(tmp_path: Path) -> None:
    assert require_scratch_path(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="under /scratch/djy8hg"):
        require_scratch_path(Path.home() / "forbidden-native-cache")


def test_vllm_scheduler_transaction_frees_only_dropped_owned_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Block:
        def __init__(self, block_id: int, *, is_null: bool = False) -> None:
            self.block_id = block_id
            self.is_null = is_null

    class Pool:
        def __init__(self) -> None:
            self.null_block = Block(0, is_null=True)
            self.freed: list[int] = []

        def free_blocks(self, blocks: list[Block]) -> None:
            self.freed.extend(block.block_id for block in blocks)

    class FullAttentionManager:
        block_size = 16

        def __init__(self, blocks: list[Block]) -> None:
            self.req_to_blocks = {"request": blocks}

    pool = Pool()
    source = [Block(index) for index in range(1, 5)]
    group = FullAttentionManager(source)
    coordinator = SimpleNamespace(single_type_managers=[group], block_pool=pool)
    manager = SimpleNamespace(enable_caching=False, coordinator=coordinator)
    request = SimpleNamespace(
        num_computed_tokens=64,
        num_prompt_tokens=64,
        output_token_ids=[],
    )
    scheduler: Any = SimpleNamespace(kv_cache_manager=manager, requests={"request": request})
    output = SimpleNamespace(num_scheduled_tokens={"request": 64})
    monkeypatch.setenv("MOSAICKV_NATIVE_RETENTION_RATIO", "0.5")
    monkeypatch.setenv("MOSAICKV_NATIVE_BLOCK_SIZE", "16")
    monkeypatch.setenv("MOSAICKV_NATIVE_AUDIT_DIR", str(tmp_path))

    _compress_ready_requests(scheduler, output)

    assert [block.block_id for block in source] == [1, 0, 0, 4]
    assert pool.freed == [2, 3]
    assert request._mosaickv_native_compacted is True
    assert request._mosaickv_force_block_reset is True
    audit = json.loads(native_audit_path(tmp_path, "vllm", "request").read_text())
    assert audit["selection"]["selected_blocks"] == [0, 3]
    assert audit["logical_next_position"] == 64


def test_vllm_scheduler_consumes_hash_bound_common_selector_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Block:
        def __init__(self, block_id: int, *, is_null: bool = False) -> None:
            self.block_id = block_id
            self.is_null = is_null

    class Pool:
        def __init__(self) -> None:
            self.null_block = Block(0, is_null=True)
            self.freed: list[int] = []

        def free_blocks(self, blocks: list[Block]) -> None:
            self.freed.extend(block.block_id for block in blocks)

    class FullAttentionManager:
        block_size = 16

        def __init__(self, blocks: list[Block]) -> None:
            self.req_to_blocks = {"request": blocks}

    pool = Pool()
    source = [Block(index) for index in range(1, 7)]
    group = FullAttentionManager(source)
    coordinator = SimpleNamespace(single_type_managers=[group], block_pool=pool)
    manager = SimpleNamespace(enable_caching=False, coordinator=coordinator)
    request = SimpleNamespace(
        num_computed_tokens=96,
        num_prompt_tokens=96,
        output_token_ids=[],
        sampling_params=SimpleNamespace(
            extra_args={
                "mosaickv_native": {
                    "logical_positions": 96,
                    "block_size": 16,
                    "retention_ratio": 0.5,
                    "selected_blocks": [0, 2, 5],
                    "mandatory_blocks": [0, 5],
                    "selected_block_count": 3,
                    "policy": "common_selector_request_wide_vote_projection_v2",
                    "method": "mosaickv_exact",
                    "catalog_sha256": "a" * 64,
                    "selector_planning_seconds": 0.25,
                    "source_trace_sha256": "b" * 64,
                }
            }
        ),
    )
    scheduler: Any = SimpleNamespace(kv_cache_manager=manager, requests={"request": request})
    output = SimpleNamespace(num_scheduled_tokens={"request": 96})
    # The environment ratio is only the fail-closed fallback. A hash-bound
    # request plan owns its exact per-request ratio.
    monkeypatch.setenv("MOSAICKV_NATIVE_RETENTION_RATIO", "0.75")
    monkeypatch.setenv("MOSAICKV_NATIVE_BLOCK_SIZE", "16")
    monkeypatch.setenv("MOSAICKV_NATIVE_AUDIT_DIR", str(tmp_path))

    _compress_ready_requests(scheduler, output)

    assert [block.block_id for block in source] == [1, 0, 3, 0, 0, 6]
    audit = json.loads(native_audit_path(tmp_path, "vllm", "request").read_text())
    assert audit["selector_source"] == "hash_bound_common_hf_selector_projection"
    assert audit["selection"]["selected_blocks"] == [0, 2, 5]
    assert audit["selector_catalog_sha256"] == "a" * 64


def test_sglang_release_frees_only_compacted_active_ownership() -> None:
    class Allocator:
        freed: torch.Tensor | None = None

        def free(self, locations: torch.Tensor) -> None:
            self.freed = locations.clone()

    class RequestPool:
        def __init__(self) -> None:
            self.req_to_token = torch.zeros((1, 400), dtype=torch.int64)
            self.req_to_token[0, :158] = torch.arange(1, 159)
            self.freed: list[int] = []

        def free(self, index: int) -> None:
            self.freed.append(index)

    allocator = Allocator()
    request_pool = RequestPool()
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=request_pool,
    )
    request = SimpleNamespace(
        req_pool_idx=0,
        output_ids=list(range(16)),
        _mosaickv_active_prompt_positions=143,
    )

    _release_active_request(cache, request)

    assert allocator.freed is not None
    assert allocator.freed.tolist() == list(range(1, 159))
    assert request_pool.freed == [0]
