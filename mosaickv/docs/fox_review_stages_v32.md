# V32 finite automatic review continuation

This controller schedules CPU callbacks only. The existing, independently
guarded COCO launcher is the sole source of two new GPU workers. No fresh
identity selection, materialization, confirmation, or paper campaign is enabled.

From a clean V32 source:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_stages_v32.py \
  launch --root NEW_IMMUTABLE_SCRATCH_ROOT
```

The finite chain is:

1. Full current-source CPU suite plus the acknowledged V31 dispatch `22832447`.
2. Read V31's actual write-once dispatch and analysis receipts; refuse errors.
3. After actual whole analysis, independently replay all original GPU artifact
   hashes, retirement receipts, ownership deltas, and the 128-prediction controls.
4. Only if whole-request certification and exact byte matching pass, launch
   the two consumed-input COCO 128-prediction controls.
5. Only after the acknowledged COCO analyzer passes all actual controls and
   hashes, launch the separately bounded V30 CPU provenance chain, reusing the
   current-source full CPU suite. Stop there.

Every callback is write-once. A failed producer records a blocked decision;
it does not authorize GPU retries or successor research. A single `scontrol`
observation at a stage boundary avoids stale dependencies after Slurm's
`MinJobAge`: verified completed gates need no old `afterany` job ID. A terminal
producer with no gate reaches bounded error analysis, never unconditional
promotion. There is no polling loop.

Callbacks use one CPU node in `interactive`: 4 CPUs/8 GiB/5 minutes, except
the full suite (15 minutes) and whole replay (12 CPUs/24 GiB/20 minutes).
All caches and outputs remain on scratch. Execution and validation SHAs are
kept distinct; original failures remain immutable.
