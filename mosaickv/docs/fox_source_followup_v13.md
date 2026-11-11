# Source fidelity and ownership follow-up V13

This is a finite engineering diagnostic, not a fresh confirmation campaign.
EviCastKV is the paper name of MosaicKV. The production three-bit storage
variant and the six-component research implementation remain unchanged.

## Observed V12 evidence

The immutable producer is `76d4ced1a10c6746bad73f564e84210522355ef9`, in
`/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1`. Its clean CPU suite
passed 6,220 tests. All numbers below come from its actual saved outputs.

| Control | Observed result | Interpretation |
| --- | --- | --- |
| Eight-bit normalized source | Both consumed inputs fail the unchanged 32-cache-dependent-prediction token guard | More source precision is not a demonstrated fidelity fix |
| Lossless BF16 normalized source | Original prompt K/V are bitwise equal; all 128 fixed-prefix logits are exactly equal on both inputs | This separately labeled representation solves those consumed-input numerical failures |
| Lossless initial prompt storage | 167,270,400 / 168,385,536 bytes versus 314,572,800 / 316,669,952 raw bytes | Actual constructed source packets, not a conditional allocation witness |
| Lossless final storage | 201,081,802 / 202,184,010 bytes | Includes observed append-only lossless history; not a shared-byte comparison |
| Lossless decode-owned observed peak | 240,042,954 / 241,390,922 bytes | Declared tensor boundary, not a whole-request KV-only certificate |
| LOOK-M checkpoint control | Stops before loading weights because bare PyTorch UUID and NVIDIA `GPU-` UUID compare unequal | Correct the identifier representation, not the isolation requirement |
| Allocator ownership | Model executes, but lossless raw trace serialization exceeds the registered 2-GiB JSON cap | No complete trace exists; no ownership certificate may be inferred |
| Historical media | 4,354 new encoded groups verified; 28,086 typed assets total; five ID-only provider gaps remain | Coverage increased, but no globally fresh cohort is certified |

The lossless control retains all prompt positions, uses no benchmark labels,
and does not update weights. Its full fixed-prefix logit parity is stronger
than token agreement on these inputs, but is not an independent task-quality
evaluation. Source reconstruction and exhaustive accounting slow the
instrumented diagnostic; it is not a latency benchmark.

## V13 interventions

1. Normalize full physical GPU UUIDs without accepting malformed, ambiguous,
   MIG, wrong-device, or competing-process cases. Retry only the failed real
   LOOK-M checkpoint control, preserving its patched-official compatibility
   label and its different legacy checkpoint/conversation convention.
2. Construct lossless source packets directly after each original attention
   layer, before the next layer begins. Retire that layer's raw K/V. Avoid
   global source/query capture in the measured request. Reuse the exact BF16
   packet schema that passed V12, with all-word validation and no fallback.
3. Preserve complete allocator events through a bounded, lossless interned
   trace representation instead of duplicating identical stack descriptions.
   Representation limits are resource controls, not numerical tolerances.
   A complete trace is only the input to attribution, not certification itself.
4. Audit available historical image-provider pools without reading answers or
   descriptions. A conservative full-pool media superset can cover images
   without pretending ambiguous row identities or split labels are resolved.
   Missing provider configurations and video execution evidence remain gaps.

CPU regressions and the complete clean-source CPU suite precede GPU work.
Each independent GPU diagnostic receives one A100 and 12 CPUs, at most 20
minutes; aggregate project occupancy remains at most six. CPU array work is
bounded to eight one-core five-minute tasks. Every producer has an `afterany`
analyzer. Submission acknowledgements and every failure are preserved;
neither a lost acknowledgement nor a failed scientific gate triggers a blind
retry. No job continuously polls its producer's log.

A separate CPU-only history driver attaches to V12's actual analyzer job
ledgers through at most four `afterany` callbacks, then continues their exact
finished states for at most 24 additional progress-making segments. Each
segment uses eight one-core, five-minute workers with a 210-second scan budget.
It stops on exhaustion, missing workers, no progress, or the registered limit,
then performs one bounded YAML-event recovery. The original errors and unknown
identities remain in the resulting inventory. Completion triggers the final
V13 report after all numerical analyzers; it does not trigger fresh inference.

## Five original requirements remain distinct

- **Source fidelity:** the lossless variant has a positive consumed-input
  control. The original lossy source remains negative and cannot inherit it.
- **Genuine byte matching:** V11 already constructed exact initial-byte
  matches, but its final/peak boundaries differ. V12 lossless caches are real
  constructed objects, not byte-matched baseline comparisons. Neither solves
  final/peak matching by itself.
- **Fresh confirmation:** every declared historical root and known provider
  must be covered. A finite successful hash job is not proof of global absence.
- **Whole-request memory:** require a complete time-aligned ownership trace,
  classified workspace and an explicit scope. Do not replace this with
  `max_memory_allocated`, reserved HBM, or a decode-only tensor sum.
- **Broader claims:** published operating-point reproduction, independent
  benchmark quality, adjusted strict latency bounds and six-component benefits
  need separate empirical evidence. A storage-only numerical pass is not proof
  that forecasting, graph selection, prototypes or repair help.

Nothing in this protocol invents a superiority claim, removes negative
observations, changes the registered KL/logit/token thresholds, or authorizes
a large fresh cohort automatically.

## Canonical entry point

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_v13.py register \
  --root /scratch/djy8hg/workdir/fox-review-v13/run-NEW --execute
```

Use a clean immutable worktree with initialized pinned submodules. Results are
write-once outside the repository. The final machine-readable report keeps
missing/failed gates and cannot promote a paper campaign.
