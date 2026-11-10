"""Recovery may reuse complete jobs, never filter failures or change execution."""

import copy

import evicastkv_review_recovery as recovery
import pytest


def test_imports_match_unchanged_controls_and_keep_the_original_producer():
    jobs = [dict(workload="coco", indices=[0, 1], arms=[["mosaickv_exact", 40, "complete"]])]
    workers = {0: dict(status="passed", errors=[])}
    before = copy.deepcopy(jobs)
    result = recovery.imported_plan(
        dict(jobs=jobs), jobs, workers, parent="/scratch/parent", producer_sha="a" * 40
    )
    assert result == {"0": dict(parent_root="/scratch/parent", source_sha="a" * 40)}
    assert jobs == before


@pytest.mark.parametrize("status,errors", [("failed", []), ("passed", ["numerical error"])])
def test_failed_worker_is_never_imported(status, errors):
    jobs = [dict(workload="coco", indices=[0, 1])]
    with pytest.raises(ValueError, match="failed correctness"):
        recovery.imported_plan(
            dict(jobs=jobs),
            jobs,
            {0: dict(status=status, errors=errors)},
            parent="/scratch/parent",
            producer_sha="a" * 40,
        )


@pytest.mark.parametrize(
    "changed", [dict(indices=[0]), dict(workload="milebench"), dict(indices=[1, 0])]
)
def test_sample_filtering_and_reordering_cannot_reuse_the_original_job(changed):
    original = dict(workload="coco", indices=[0, 1])
    modified = dict(original, **changed)
    with pytest.raises(ValueError, match="identical parent"):
        recovery.imported_plan(
            dict(jobs=[original]),
            [modified],
            {0: dict(status="passed", errors=[])},
            parent="/scratch/parent",
            producer_sha="a" * 40,
        )


def test_duplicate_or_remapped_parent_indices_fail_closed():
    job = dict(workload="coco", indices=[0])
    with pytest.raises(ValueError, match="duplicate parent"):
        recovery.imported_plan(dict(jobs=[job, job]), [job], {}, parent="p", producer_sha="s")
    with pytest.raises(ValueError, match="indices"):
        recovery.imported_plan(
            dict(jobs=[dict(workload="milebench"), job]),
            [job],
            {1: dict(status="passed", errors=[])},
            parent="p",
            producer_sha="s",
        )
