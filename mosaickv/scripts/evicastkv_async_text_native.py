#!/usr/bin/env python3
"""Same-value background text/native sharing, consumed-input numerical-first screen."""

from pathlib import Path

import evicastkv_prefill_catalog as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base, SOURCE, driver, parent = prior.base, prior.SOURCE, prior.driver, prior.parent
PREVIOUS = Path(
    "/scratch/djy8hg/runs/evicastkv/prefill-owner-catalog-consumed-20261106-extra-aware1"
)
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_async_text_native.sbatch"
DOC = SOURCE / "mosaickv/docs/async_text_native.md"
ARMS = (
    ("mosaickv_exact", "joint-same-buffer"),
    ("mosaickv_exact", "joint-async-handoff"),
    ("prefixkv_reimpl", "graph-history-graph"),
    ("prefixkv_reimpl", "graph-original-history-graph"),
)
OLD_OVERRIDES = base.overrides


def overrides(method, arm):
    if method == "mosaickv_exact" and arm == "joint-async-handoff":
        return dict(
            OLD_OVERRIDES(method, "joint-same-buffer"), verified_constructor="async_handoff"
        )
    return OLD_OVERRIDES(method, arm)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    gate = driver.read(PREVIOUS / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete preceding common-accounting screen required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for path in (DOC, PREVIOUS / "plan.json", PREVIOUS / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    for i in (0, 1):
        for method, arm in ARMS:
            if arm == "joint-async-handoff":
                continue
            path = PREVIOUS / f"workers/{i}/audited-{method}-{arm}.json"
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            arms=ARMS,
            stage="consumed_async_shared_text_native_construction",
            resolved_arm_overrides=[
                dict(method=m, arm=a, overrides=overrides(m, a)) for m, a in ARMS
            ],
            frozen_candidate_constructor="async_handoff",
            no_selection_or_precision_retuning=True,
            correctness_only=False,
            warmups=5,
            repetitions=3,
            full_original_parent_packet_and_128_logits_required=True,
            both_original_native_and_outer_stride_audits_required=True,
            maximum_pending_layers=2,
            same_request_publication_and_stream_events_required=True,
            no_full_reference_retention_across_components_or_layers=True,
            active_storage_change_allowed_only_for_nonincreasing_numerical_repairs=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def execute(*args, **kwargs):
    value = prior.execute(*args, **kwargs)
    if args[5] == "mosaickv_exact" and args[6] == "joint-async-handoff":
        info = value["overlapping_native_construction"]["report"]
        if (
            not info["completed"]
            or not info["deferred_text_ready"]
            or not info["deferred_text"]
            or not info["thread_retired"]
            or info["native_reference_callbacks"] != 64
            or info["published_text_layers"] != 32
            or info["cached_text_layers"] != len(value["direct_text_construction"]["cached_layers"])
            or info["maximum_pending_layers"] > 2
            or info["engine_count"] != 1
            or info["outer_stride_word_checks"] != (192 if kwargs.get("audit", False) else 0)
        ):
            raise RuntimeError("bounded actual shared-reference construction did not complete")
    return value


def summary(rows, audited):
    reference = audited["mosaickv_exact", "joint-same-buffer"]
    candidate = audited["mosaickv_exact", "joint-async-handoff"]
    length = reference["prefill_storage"]["logical_lengths"][0]
    if length not in (600, 604):
        raise ValueError("only the two same consumed inputs permitted")
    index = 0 if length == 600 else 1
    for method, arm in ARMS:
        if arm == "joint-async-handoff":
            continue
        old = driver.read(PREVIOUS / f"workers/{index}/audited-{method}-{arm}.json")
        if any(audited[method, arm][key] != old[key] for key in driver.STORAGE):
            raise RuntimeError("unchanged comparison control changed physical storage")
    delta = candidate[driver.STORAGE[0]] - reference[driver.STORAGE[0]]
    if (
        delta > 0
        or any(candidate[k] - reference[k] != delta for k in driver.STORAGE[1:])
        or delta
        != candidate["lt_source_conversion"]["added_patch_bytes"]
        - reference["lt_source_conversion"]["added_patch_bytes"]
    ):
        raise RuntimeError("storage may change only by counted nonincreasing native corrections")
    return dict(
        base.ORIGINAL_SUMMARY(rows, audited),
        numerical_repair_byte_delta=delta,
        physical_storage={
            f"{m}:{a}": {k: v[k] for k in driver.STORAGE} for (m, a), v in audited.items()
        },
        whole_request_GPU_peak_certified=False,
        independent_quality_N=0,
        no_fresh_or_larger_campaign_authorized=True,
    )


def finish_configuration():
    base.control.CANDIDATE_ARM = "joint-async-handoff"
    base.control.VARIANT = "consumed_async_shared_text_native_construction"
    base.control.JOB_NAME = "mk-async-text-native"
    prior.OLD_MAIN()


def main():
    base.prepare, base.execute, base.summary, base.overrides = prepare, execute, summary, overrides
    base.SCRIPT, base.DOC, base.ARMS = SCRIPT, DOC, ARMS
    base.control.main = finish_configuration
    base.main()


if __name__ == "__main__":
    main()
