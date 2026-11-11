"""CPU extension tests; no real allocator attachment or CUDA initialization."""

from pathlib import Path

import pytest
import torch

from mosaickv.measurements import native_allocator_events as native


@pytest.fixture(scope="module")
def synthetic():
    return native._load_recorder(testing=True)


def event(i=0, action=0):
    return {
        "device": i % 2,
        "addr": 2**63 + i,
        "size": 17 + i,
        "stream": 0x1234 + i,
        "pool": [2**63 + 1, 18 + i],
        "approx_time": 100 + i,
        "action_code": action,
    }


def test_compile_load_does_not_initialize_cuda():
    before = torch.cuda.is_initialized()
    module = native.load_recorder()
    assert torch.cuda.is_initialized() == before
    assert not hasattr(module, "_test_only_capture")
    assert not module.metadata()["synthetic_test_build"]
    assert module.build_provenance["RTLD_NODELETE"]
    assert len(module.build_provenance["header_sha256"]) == 64
    assert module.cursor() == 0
    with pytest.raises(RuntimeError, match="running recorder"):
        module.stop()
    # No real start attempt: CUDA state in a complete test suite may differ.


@pytest.mark.parametrize("capacity", [0, 2000001])
def test_invalid_capacity_never_attaches(capacity):
    with pytest.raises(ValueError, match="bounded capacity"):
        native.load_recorder().start(capacity)


def test_scalar_fields_and_actions_preserved(synthetic):
    data = [event(i, i) for i in range(9)]
    result = synthetic._test_only_capture(data, 9)
    assert result["complete"] and result["synthetic_test_only"]
    assert result["count"] == result["attempted_events"] == 9
    assert [x["action"] for x in result["events"]] == [
        "alloc",
        "free_requested",
        "free_completed",
        "segment_alloc",
        "segment_free",
        "segment_map",
        "segment_unmap",
        "snapshot",
        "oom",
    ]
    for i, row in enumerate(result["events"]):
        assert row["sequence"] == i
        for key in ("device", "addr", "size", "stream", "pool", "approx_time", "action_code"):
            assert row[key] == data[i][key]
        assert row["mempool"] == row["pool"]
    assert not result["callback_history_required"]
    assert not result["whole_request_KV_peak_certified"]
    assert not result["automatic_KV_roles_assigned"]


def test_overflow_is_explicit_and_not_complete(synthetic):
    result = synthetic._test_only_capture([event(i) for i in range(32)], 3, threads=4)
    assert result["count"] == 3 and result["attempted_events"] == 32
    assert result["overflow"] and not result["complete"]
    assert [x["sequence"] for x in result["events"]] == [0, 1, 2]


def test_concurrent_tickets_and_stop_are_quiescent(synthetic):
    data = [event(i) for i in range(10000)]
    complete = synthetic._test_only_capture(data, len(data), threads=8)
    assert complete["complete"]
    assert [x["sequence"] for x in complete["events"]] == list(range(len(data)))
    assert {x["addr"] for x in complete["events"]} == {x["addr"] for x in data}
    stopped = synthetic._test_only_capture(data, len(data), threads=8, stop_after=2)
    assert stopped["writes_quiesced"] and not stopped["stop_wait_timeout"]
    assert 0 <= stopped["count"] <= len(data)
    assert stopped["count"] == len(stopped["events"])


def test_unknown_action_rejects_completeness(synthetic):
    result = synthetic._test_only_capture([event(action=99)], 1)
    assert result["unknown_action"] and not result["complete"]


def test_no_callback_heap_lock_python_or_cuda_calls():
    source = Path(native.__file__).with_name("csrc") / "native_allocator_events.cpp"
    callback = source.read_text().split("// CALLBACK BEGIN:")[1].split("// CALLBACK END")[0]
    for forbidden in (
        "py::",
        "new ",
        "malloc(",
        "cuda",
        "mutex",
        "push_back",
        "resize(",
        "getApproximateTime(",
        "context_",
        "compile_context_",
    ):
        assert forbidden not in callback
    assert "is_always_lock_free" in source.read_text()
    assert "allocator->initialized()" in source.read_text()


def test_cache_rejects_outside_scratch(monkeypatch):
    monkeypatch.setenv("MOSAICKV_CACHE_ROOT", "/var/tmp/evicastkv-native-cache-not-permitted")
    with pytest.raises(ValueError, match="scratch-only"):
        native.load_recorder()


def test_unreviewed_headers_rejected(monkeypatch):
    monkeypatch.setattr(native, "_HEADERS", {"c10/cuda/CUDACachingAllocator.h": "0" * 64})
    with pytest.raises(RuntimeError, match="differ from source review"):
        native.load_recorder()


@pytest.mark.parametrize(
    "flag", ["PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NO_HIP_MEMORY_CACHING"]
)
def test_uncached_mode_rejected_without_attachment(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(RuntimeError, match="enabled native caching allocator"):
        native.load_recorder().start(8)
