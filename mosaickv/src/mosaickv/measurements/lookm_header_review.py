"""Narrow source review of saved common-header LOOK-M precision diagnostics.

These functions exclusively construct cache selection/encoding or shared cache
reader buffers. Charge their previously unknown allocations conservatively as
KV workspace; never erase allocation lifetimes or classify generic wrappers.
Not an official LOOK-M reproduction or a whole-process HBM certificate.
"""

from mosaickv.measurements.ownership_review import RULES, review

PRODUCER_SHA = "03d74148ed6c86c81ee137599ccc48586c327cf6"
SOURCE_FUNCTIONS = (
    ("/mosaickv/evaluation/streaming_baselines.py", "lookm_layer", "kv_workspace"),
    ("/mosaickv/evaluation/streaming_baselines.py", "encode_layer", "kv_workspace"),
    (
        "/mosaickv/evaluation/native_history_feasibility.py",
        "workspace_buffers",
        "kv_shared_workspace",
    ),
)
SOURCE_LINES = tuple(
    ("/mosaickv/evaluation/streaming_baselines.py", "__call__", "kv_workspace", line)
    for line in (344, 345, 350, 351, 352, 355)
) + tuple(
    ("/mosaickv/evaluation/sequential_kv_lease.py", "forward", "kv_shared_workspace", line)
    for line in (307, 309, 310, 315)
)


def reviewed(report):
    return review(report, rules=(*RULES, *SOURCE_FUNCTIONS, *SOURCE_LINES))
