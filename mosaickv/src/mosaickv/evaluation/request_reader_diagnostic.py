"""Windowed, request-path reader engagement and paired engineering statistics.

Instrumentation is separate from latency trials. No labels, tensors, or decoded
cache bodies enter the counters. All contexts and monkeypatches are scoped.
"""

from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps

import numpy as np

from mosaickv.evaluation.construction_phase_timing import PhaseTimer, phase_totals
from mosaickv.evaluation.reconstruction_policy import current

METHODS = ("mosaickv_exact", "prefixkv_reimpl")
READERS = ("original", "carry_literals")
_progress = ContextVar("request_reader_progress", default=None)


def window(step):
    if type(step) is not int or not 0 <= step < 127:
        raise ValueError("fixed 127-step decode diagnostic required")
    return "early" if step < 16 else "middle" if step < 96 else "late"


def phase_window():
    p = _progress.get()
    if p is None:
        return "prefill"
    return window(min(p["steps"], 126)) if p["started"] else "decode_setup"


class WindowTimer(PhaseTimer):
    def phase(self, name):
        return super().phase(phase_window() + "/" + name)


@contextmanager
def instrument(adapter, timer):
    from unittest.mock import patch

    from mosaickv.evaluation import (
        consolidated_history,
        layer_code_delta,
        source_history,
        storage_runs,
    )
    from mosaickv.evaluation import (
        encoded_projection_kernel as encoded,
    )
    from mosaickv.evaluation import (
        reconstruction_roles as roles,
    )

    calls, state = Counter(), dict(steps=0, started=False)
    original_run, original_decode, original_project = (
        storage_runs.run,
        adapter.decode_one_token,
        encoded.project,
    )

    @wraps(original_decode)
    def decode(*args, **kwargs):
        if _progress.get() is not state:
            raise RuntimeError("decoder outside owned diagnostic run")
        state["started"] = True
        with timer.phase("decode_step"):
            result = original_decode(*args, **kwargs)
        state["steps"] += 1
        return result

    @wraps(original_run)
    def run(*args, **kwargs):
        if _progress.get() is not None or state["steps"]:
            raise RuntimeError("exactly one nonnested decode per request required")
        token = _progress.set(state)
        try:
            result = original_run(*args, **kwargs)
            if state["steps"] != 127:
                raise RuntimeError("incomplete fixed-horizon diagnostic")
            return result
        finally:
            _progress.reset(token)

    @wraps(original_project)
    def project(*args, **kwargs):
        # inspect_kernel returns metadata from THIS launch; no reference launch.
        result = original_project(*args, **dict(kwargs, inspect_kernel=True))
        expected = (
            "raw"
            if args[0].raw
            else "packed_counts"
            if kwargs.get("packed_counts", False)
            else "carried"
            if current() in ("carry_literals", "reuse_both")
            else "original"
        )
        if result["compiled_reader_branch"] != expected:
            raise RuntimeError("requested/compiled reader branch mismatch")
        role = roles._role.get()
        if role == "prompt_source":
            role = "text_overlay"
        calls[phase_window(), role, expected] += 1
        return result if kwargs.get("inspect_kernel", False) else None

    with ExitStack() as stack:
        # The engagement wrapper runs inside the role timer, while retaining
        # the caller's authoritative packet-ownership role.
        stack.enter_context(patch.object(encoded, "project", project))
        stack.enter_context(roles.instrument(timer))
        stack.enter_context(patch.object(adapter, "decode_one_token", decode))
        stack.enter_context(patch.object(storage_runs, "run", run))
        for owner, name, label in (
            (source_history, "seal_source_full_tails", "history_sealing"),
            (consolidated_history, "consolidate", "history_consolidation"),
            (layer_code_delta.Replay, "view", "code_replay"),
            (layer_code_delta.Replay, "__init__", "code_replay_setup"),
        ):
            stack.enter_context(patch.object(owner, name, timer.wrap(label, getattr(owner, name))))
        yield dict(counts=calls, progress=state)


def trace_summary(timer, evidence):
    return dict(
        steps=evidence["progress"]["steps"],
        phases=phase_totals(timer.resolve()),
        branch_counts=[
            dict(window=w, role=r, branch=b, calls=n)
            for (w, r, b), n in sorted(evidence["counts"].items())
        ],
        instrumentation_not_request_latency=True,
    )


def order(seed, *, readers=READERS):
    import random

    rng = random.Random(seed)
    output = []
    for trial in range(-5, 20):
        methods = list(METHODS)
        rng.shuffle(methods)
        for method in methods:
            policies = list(readers)
            rng.shuffle(policies)
            output.extend(
                dict(trial=trial, warmup=trial < 0, method=method, reader=p) for p in policies
            )
    return output


def summarize(rows, seed, *, readers=READERS):
    expected = {(r["trial"], r["method"], r["reader"]) for r in order(seed, readers=readers)}
    keys = [(r["trial"], r["method"], r["reader"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique paired request grid required")
    for row in rows:
        if row.get("logits_bitwise") is not True or row.get("accounting_identical") is not True:
            raise ValueError("bitwise and storage controls must pass for every request")
        if row.get("instrumented") is not False:
            raise ValueError("instrumentation forbidden in latency trials")
        if row.get("tensor_result_auditing") is not False:
            raise ValueError("tensor-audited storage controls are not latency trials")
        for name in (
            "total_wall_seconds",
            "decode_wall_seconds",
            "prefill_construct_retire_wall_seconds",
        ):
            if not np.isfinite(row[name]) or row[name] <= 0:
                raise ValueError("finite positive phase times required")
    by_key = dict(zip(keys, rows, strict=True))
    arrays = {
        (m, p): np.asarray([by_key[t, m, p]["total_wall_seconds"] for t in range(20)])
        for m in METHODS
        for p in readers
    }
    ids = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    tail = 0.05 / (2 * 8)  # four paired policy/method contrasts on each input

    def contrast(a, b):
        draws = a[ids].mean(1) / b[ids].mean(1)
        return dict(
            mean_ratio=float(a.mean() / b.mean()),
            familywise_ratio_bounds=np.quantile(draws, [tail, 1 - tail]).tolist(),
            paired_difference_CI95=np.quantile((a[ids] - b[ids]).mean(1), [0.025, 0.975]).tolist(),
        )

    return dict(
        trials=20,
        warmups=5,
        distributions=[
            dict(
                method=m,
                reader=p,
                mean=float(a.mean()),
                p50=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)),
                mean_decode_seconds=float(
                    np.mean([by_key[t, m, p]["decode_wall_seconds"] for t in range(20)])
                ),
                mean_prefill_construction_retirement_seconds=float(
                    np.mean(
                        [
                            by_key[t, m, p]["prefill_construct_retire_wall_seconds"]
                            for t in range(20)
                        ]
                    )
                ),
            )
            for (m, p), a in arrays.items()
        ],
        within_method={m: contrast(arrays[m, readers[1]], arrays[m, readers[0]]) for m in METHODS},
        versus_prefix={
            p: contrast(arrays["mosaickv_exact", p], arrays["prefixkv_reimpl", p]) for p in readers
        },
        independent_quality_N=0,
        byte_matched_algorithm_comparison=False,
        paper_superiority_established=False,
    )
