"""Actual-logit shortlisting before full-horizon acceptance; no label inputs."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Sequence
from typing import Any

from mosaickv.backends.hf_allocation_guard import decide_draft_stability, draft_row_causes
from mosaickv.config import DraftAllocationGuardConfig


def draft_rank(rows: list[dict[str, Any]], identity: str) -> tuple[Any, ...]:
    return (
        sum(not row["candidate"]["greedy_token_agreement"] for row in rows),
        sum(row["candidate"]["logit_kl"] for row in rows),
        sum(row["candidate"]["maximum_logit_difference"] for row in rows),
        identity,
    )


def meaningful_improvement(rows: list[dict[str, Any]], policy: DraftAllocationGuardConfig) -> bool:
    """A numerical tie is not a meaningful accepted modification."""
    return any(
        row["anchor"][metric] - row["candidate"][metric]
        > policy.absolute_tolerance + policy.relative_tolerance * abs(row["anchor"][metric])
        for row in rows
        for metric in ("logit_kl", "maximum_logit_difference")
    ) or any(
        not row["anchor"]["greedy_token_agreement"] and row["candidate"]["greedy_token_agreement"]
        for row in rows
    )


def draft_shortlist(
    proposals: Sequence[tuple[str, Any]],
    probe: Callable[[Any], Generator[dict[str, Any], None, None]],
    policy: DraftAllocationGuardConfig,
    *,
    short_horizon: int = 4,
    shortlist_size: int = 2,
    synchronize: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    if (
        not 1 <= len(proposals) <= 16
        or len({key for key, _ in proposals}) != len(proposals)
        or type(short_horizon) is not int
        or not 1 <= short_horizon < policy.horizon
        or type(shortlist_size) is not int
        or not 1 <= shortlist_size <= 2
    ):
        raise ValueError("invalid bounded draft-shortlist policy")
    events: list[dict[str, Any]] = []
    survivors: list[tuple[dict[str, Any], Any]] = []
    started_streams: list[Generator[dict[str, Any], None, None]] = []

    def advance(event, stream, until):
        for step in range(len(event["steps"]) + 1, until + 1):
            row = next(stream)
            if row["step"] != step:
                raise ValueError("draft probe prefix sequence changed")
            event["steps"].append(row)
            if draft_row_causes(row, policy):
                event["status"] = "rejected_at_first_violation"
                stream.close()
                return False
        return True

    synchronize()
    t0 = time.perf_counter()
    try:
        for identity, proposal in proposals:
            stream = probe(proposal)
            started_streams.append(stream)
            event = dict(proposal_id=identity, steps=[], status="short_probe_only_not_accepted")
            events.append(event)
            if advance(event, stream, short_horizon):
                survivors.append((event, stream))
                survivors.sort(
                    key=lambda item: draft_rank(item[0]["steps"], item[0]["proposal_id"])
                )
                if len(survivors) > shortlist_size:
                    loser, old_stream = survivors.pop()
                    loser["status"] = "not_in_actual_logit_shortlist"
                    old_stream.close()
        synchronize()
        screening_seconds = time.perf_counter() - t0
        shortlist = [event["proposal_id"] for event, _ in survivors]
        t0 = time.perf_counter()
        for event, stream in survivors:
            if advance(event, stream, policy.horizon):
                event["status"] = "full_horizon_pass"
            event["decision"] = decide_draft_stability(
                event["steps"], policy, allow_early_rejection=True
            )
            stream.close()
        synchronize()
        extension_seconds = time.perf_counter() - t0
    finally:
        for stream in started_streams:
            stream.close()
    passed = [event for event in events if event["status"] == "full_horizon_pass"]
    winner = (
        min(passed, key=lambda event: draft_rank(event["steps"], event["proposal_id"]))
        if passed
        else None
    )
    return dict(
        events=events,
        shortlist=shortlist,
        winner=None if winner is None else winner["proposal_id"],
        accepted=winner is not None,
        meaningful=winner is not None and meaningful_improvement(winner["steps"], policy),
        short_horizon=short_horizon,
        shortlist_size=shortlist_size,
        horizon=policy.horizon,
        checkpoints=list(policy.checkpoints),
        absolute_tolerance=policy.absolute_tolerance,
        relative_tolerance=policy.relative_tolerance,
        screening_seconds=screening_seconds,
        extension_seconds=extension_seconds,
        candidate_updates=sum(len(event["steps"]) for event in events),
        benchmark_labels_accessed=False,
        fresh_cohort_authorized=False,
    )


def validate_shortlist_trace(trace: dict[str, Any]) -> None:
    """Replay ranking from actual short-prefix logits, never a surrogate score."""
    policy = DraftAllocationGuardConfig(
        trace["horizon"],
        tuple(trace["checkpoints"]),
        trace["absolute_tolerance"],
        trace["relative_tolerance"],
    )
    events = trace["events"]
    short = trace["short_horizon"]
    if (
        not 1 <= len(events) <= 16
        or not 1 <= short < policy.horizon
        or not 1 <= trace["shortlist_size"] <= 2
    ):
        raise ValueError("draft shortlist trace exceeds registered limits")
    eligible = []
    for event in events:
        rows = event["steps"]
        if not rows or [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("incomplete/out-of-order probe sequence")
        failures = [row["step"] for row in rows if draft_row_causes(row, policy)]
        if failures and failures != [len(rows)]:
            raise ValueError("probe did not stop at first violation")
        if len(rows) >= short and not any(step <= short for step in failures):
            eligible.append(event)
        if (
            event["status"] == "full_horizon_pass"
            and not decide_draft_stability(rows, policy)["accepted"]
        ):
            raise ValueError("partial or failed proposal claims acceptance")
    expected = [
        event["proposal_id"]
        for event in sorted(
            eligible, key=lambda event: draft_rank(event["steps"][:short], event["proposal_id"])
        )[: trace["shortlist_size"]]
    ]
    if trace["shortlist"] != expected:
        raise ValueError("shortlist was not constructed using actual draft logits")
    if any(
        len(event["steps"]) > short and event["proposal_id"] not in expected for event in events
    ):
        raise ValueError("extended a non-shortlisted candidate")
    for event in events:
        rows = event["steps"]
        failed = any(draft_row_causes(row, policy) for row in rows)
        expected_status = (
            "rejected_at_first_violation"
            if failed
            else "full_horizon_pass"
            if len(rows) == policy.horizon
            else "not_in_actual_logit_shortlist"
        )
        if (
            event["status"] != expected_status
            or len(rows) > policy.horizon
            or (event["proposal_id"] in expected and not failed and len(rows) != policy.horizon)
        ):
            raise ValueError("missing full-horizon extension or inconsistent probe status")
    passed = [event for event in events if event["status"] == "full_horizon_pass"]
    winner = (
        min(passed, key=lambda event: draft_rank(event["steps"], event["proposal_id"]))
        if passed
        else None
    )
    if (
        trace["winner"] != (None if winner is None else winner["proposal_id"])
        or trace["accepted"] != (winner is not None)
        or trace["meaningful"]
        != (winner is not None and meaningful_improvement(winner["steps"], policy))
        or trace["candidate_updates"] != sum(len(event["steps"]) for event in events)
        or trace["benchmark_labels_accessed"]
        or trace["fresh_cohort_authorized"]
    ):
        raise ValueError("draft shortlist winner/probe/label invariant failed")
