"""Scoped attribution only; never changes cache arithmetic or selection."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def instrument(timer):
    from mosaickv.evaluation import (
        consolidated_history,
        encoded_projection_kernel,
        layer_code_delta,
        lossless_code_delta_gpu,
        sequential_kv_lease,
        sequential_source_history,
        shared_source_kernel,
        source_history,
    )

    targets = (
        (layer_code_delta.Replay, "view"),
        (layer_code_delta, "decode"),
        (lossless_code_delta_gpu, "decode"),
        (sequential_kv_lease, "fill_component"),
        (sequential_source_history, "fill_component"),
        (sequential_source_history.SequentialSourceHistoryLayer, "materialize_component"),
        (shared_source_kernel, "project"),
        (encoded_projection_kernel, "project"),
        (source_history, "seal_source_full_tails"),
        (consolidated_history, "consolidate"),
    )
    with ExitStack() as stack:
        for owner, name in targets:
            label = owner.__name__ + "." + name
            stack.enter_context(patch.object(owner, name, timer.wrap(label, getattr(owner, name))))
        yield


def next_dependency(folder):
    """Follow a finite DAG using its published IDs, never poll scheduler logs."""
    import json

    # Most advanced stage first. Each callback runs only after its predecessor.
    for path, key in (
        (folder / "timing/submission.json", "cpu_analysis"),
        (folder / "submission.json", "cpu_analysis"),
        (folder.with_name(folder.name + "-launch") / "construct-submission.json", "job_id"),
        (folder.with_name(folder.name + "-launch") / "choose-submission.json", "job_id"),
        (folder.with_name(folder.name + "-launch") / "wait_micro-submission.json", "job_id"),
    ):
        if path.exists():
            return str(json.loads(path.read_text())[key])
    raise ValueError("no immutable upstream dependency record")
