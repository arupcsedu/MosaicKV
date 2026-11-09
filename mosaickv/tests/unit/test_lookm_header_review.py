from copy import deepcopy

import pytest

from mosaickv.measurements.kv_timeline import AllocationTimeline, valid_whole_request_certificate
from mosaickv.measurements.lookm_header_review import reviewed


def raw(filename, name, line):
    ledger = AllocationTimeline()
    ledger.ingest(
        [
            dict(
                action="alloc",
                addr=128,
                size=64,
                stream=0,
                time_us=100,
                frames=[dict(filename=filename, name=name, line=line)],
            ),
            dict(action="free_requested", addr=128, size=64, stream=0, time_us=150),
            dict(action="free_completed", addr=128, size=64, stream=0, time_us=200),
        ]
    )
    return ledger.report(complete=True, history_complete=True, opaque_workspace_classified=False)


@pytest.mark.parametrize(
    "filename,name,line",
    [
        ("streaming_baselines.py", "lookm_layer", 215),
        ("streaming_baselines.py", "encode_layer", 175),
        ("streaming_baselines.py", "__call__", 344),
        ("sequential_kv_lease.py", "forward", 307),
        ("native_history_feasibility.py", "workspace_buffers", 41),
    ],
)
def test_conservatively_charge_exact_audited_cache_source_only(filename, name, line):
    original = raw("/project/mosaickv/evaluation/" + filename, name, line)
    before = deepcopy(original)
    result = reviewed(original)
    assert original == before
    assert valid_whole_request_certificate(result)
    assert result["exact_KV_peak_bytes"] == 64
    for key in ("start", "end", "address", "generation", "bytes"):
        assert result["records"][0].get(key) == original["records"][0].get(key)


@pytest.mark.parametrize(
    "filename,name,line",
    [
        ("streaming_baselines.py", "__call__", 999),
        ("other.py", "encode_layer", 175),
        ("sequential_kv_lease.py", "forward", 320),
    ],
)
def test_unknown_function_or_unreviewed_line_remains_blocked(filename, name, line):
    result = reviewed(raw("/project/mosaickv/evaluation/" + filename, name, line))
    assert not valid_whole_request_certificate(result)
    assert result["unknown_allocation_generations"] == 1
