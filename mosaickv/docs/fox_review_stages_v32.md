# V34 finite automatic review continuation

This controller schedules CPU callbacks only. The existing, independently
guarded COCO launcher is the sole source of two new GPU workers. No fresh
identity selection, materialization, confirmation, or paper campaign is enabled.

The legacy `v32` script and batch filenames are retained for compatibility;
the resolved policy is `V34_packed_raw_whole_COCO_provenance_callbacks_v1`.
All files tracked by actual V33 producer
`422a67994d06ffb0b69943727063a7dfc885b6d2` must remain byte-identical in this
additive tooling source. The historical V31/V32 failures remain immutable.

From a clean V34 source:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_stages_v32.py \
  launch --root NEW_IMMUTABLE_SCRATCH_ROOT
```

The finite chain is:

1. Full current-source CPU suite plus the acknowledged V33 dispatch `22834999`.
2. Read V33's actual write-once dispatch and analysis receipts; refuse errors.
3. After actual whole analysis, independently replay all original GPU artifact
   hashes, retirement receipts, ownership deltas, and the 128-prediction controls.
4. Only if whole-request certification and exact byte matching pass, launch
   the two consumed-input COCO 128-prediction controls.
5. Only after the acknowledged COCO analyzer passes all actual controls and
   hashes, launch the separately bounded V30 CPU provenance chain, reusing the
   current-source full CPU suite. Stop there.

The new GPU scope is exactly two already consumed COCO images, with 128
predictions each, not 128 examples. No benchmark scoring, label access, fresh
sample selection or larger campaign is enabled. Its pinned FP16/GQA boundary
transport does not establish BF16 source reconstruction on that checkpoint.
The final CPU scope remains the two reviewed historical subsets (9 and 512
directories; 210 seconds and 2 GiB per worker) and saved media-hash recovery.
That partial continuation cannot certify global freshness.

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
