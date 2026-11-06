#!/usr/bin/env python3
"""Common prefill metadata optimization; never withhold it from baselines."""

from pathlib import Path

import evicastkv_verified_constructor as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver, parent = base.SOURCE, base.driver, base.parent
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/verified-constructor-screen-consumed-20261106-v1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/verified-constructor-host-consumed-20261106-warm1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prefill_catalog.sbatch"
DOC = SOURCE / "mosaickv/docs/prefill_owner_catalog.md"
ARMS = (base.ARMS[0], base.ARMS[1], base.ARMS[3], base.ARMS[4])
OLD_EXECUTE, OLD_MAIN = base.execute, base.control.main


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    for folder in (PREVIOUS, PROFILE):
        gate = driver.read(folder / "analysis/decision.json")
        if not gate["complete"] or gate["failures"]:
            raise ValueError("complete preceding timing and warmed attribution required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for path in (
        DOC,
        PREVIOUS / "plan.json",
        PREVIOUS / "analysis/decision.json",
        PROFILE / "analysis/decision.json",
    ):
        bound[str(path)] = sha(path)
    for i in (0, 1):
        for method, arm in ARMS:
            path = PREVIOUS / f"workers/{i}/audited-{method}-{arm}.json"
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_common_prefill_owner_catalog",
            arms=ARMS,
            correctness_only=False,
            warmups=5,
            repetitions=3,
            resolved_arm_overrides=[
                dict(method=m, arm=a, overrides=base.overrides(m, a)) for m, a in ARMS
            ],
            common_prefill_owner_catalog="immutable_layer_and_extra_segments_v2",
            independent_full_owner_reconciliation_required=True,
            all_original_KV_words_and_storage_boundaries_unchanged=True,
            metadata_fix_offered_to_every_baseline=True,
            frozen_candidate_constructor="same_buffer",
            same_buffer_then_handoff_separate_ablations=False,
            handoff_not_retained_due_to_previous_screen=True,
            no_selection_or_precision_retuning=True,
            timing_boundary="complete_request_plus_constructor_scope_and_common_owner_reconciliation",
            candidate_selection_based_only_on_engineering_screen=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def execute(*args, **kwargs):
    value = OLD_EXECUTE(*args, **kwargs)
    info = value["prefill_storage"]["ownership_catalog"]
    if (
        info["policy"] != "immutable_layer_and_extra_segments_v2"
        or info["tensor_owners_retained"]
        or info["independent_full_enumerations"] != 1
    ):
        raise RuntimeError("common exact prefill owner reconciliation did not engage")
    return value


def summary(rows, audited):
    lengths = audited["mosaickv_exact", "joint-same-buffer"]["prefill_storage"]["logical_lengths"]
    if len(set(lengths)) != 1 or lengths[0] not in (600, 604):
        raise ValueError("only the same two consumed full-MHA inputs supported")
    index = 0 if lengths[0] == 600 else 1
    for method, arm in ARMS:
        prior = driver.read(PREVIOUS / f"workers/{index}/audited-{method}-{arm}.json")
        if any(audited[method, arm][key] != prior[key] for key in driver.STORAGE):
            raise RuntimeError("metadata optimization changed measured cache backing storage")
    return dict(
        base.ORIGINAL_SUMMARY(rows, audited),
        all_frozen_initial_final_decode_owned_peak_bytes_identical=True,
        prefill_owner_catalogs={
            f"{m}:{a}": value["prefill_storage"]["ownership_catalog"]
            for (m, a), value in audited.items()
        },
        whole_request_GPU_peak_certified=False,
        independent_quality_N=0,
        no_fresh_or_larger_campaign_authorized=True,
    )


def finish_configuration():
    base.control.CANDIDATE_ARM = "joint-same-buffer"
    base.control.VARIANT = "consumed_common_prefill_owner_catalog"
    base.control.JOB_NAME = "mk-prefill-catalog"
    OLD_MAIN()


def main():
    base.prepare, base.execute, base.summary = prepare, execute, summary
    base.SCRIPT, base.DOC, base.ARMS = SCRIPT, DOC, ARMS
    base.control.main = finish_configuration
    base.main()


if __name__ == "__main__":
    main()
