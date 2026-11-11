# Fox-review V15: finite repair scope

This is an engineering and provenance diagnostic, not an independent quality
confirmation or a paper-superiority campaign. The original lossy source variant
remains negative. The separately named lossless BF16 source control preserves
all original prompt positions and must not inherit results or accounting from
the lossy variant. The six research mechanisms remain a separate unvalidated
performance claim.

## Registered work

- Run the entire CPU suite from a clean immutable source before any producer.
- Capture a complete, time-aligned allocator event history and final snapshot
  for the streaming lossless control on the same two consumed inputs. Do not
  run another native reference prefill inside its request window. Independently
  recompute the existing 32/128-token guards from saved logits. Named model
  parameter and buffer owners must match actual baseline backing allocations;
  all other unexplained storage remains unknown.
- Refine the existing lossy-source trace on CPU using explicit, source-bound
  classifications. Report persistent-KV and cache-workspace boundaries
  separately. Source-proof coverage alone is not peak certification. An exact
  number requires complete time-aligned allocation lifetimes and equal lower
  and upper bounds; numerical failure is never promoted by accounting success.
- Recover only the two failed InfographicVQA file groups. Two exact encoded
  image hashes and dimensions may use a 64-million-pixel hash-only resource
  bound; ordinary decoding keeps its old limit. No image is truncated, resized,
  excluded, or substituted. Successful parent hashes are reused unchanged.
- Process the exact saved derived-artifact queue in bounded passes of at most
  1,500 files, at most three passes total. Import reviewed proofs and terminal
  failures; continue only unattempted or deadline-pending files. Unsupported
  artifacts remain unresolved, not waived.
- Recover the completed V13 history segment from its eight immutable worker
  files and clean-producer/hash receipts. Register a finite 256 MiB per-file,
  2 GiB total metadata bound only for this derived recovery. Preserve the old
  negative gate and every unresolved error. Do not restart traversal.
- Verify and reuse the existing V13 terminal report, accepting only the exact
  versioned ownership wrapper with matching nested gate, protocol, receipt and
  trace hashes. A missing top-level source field is not a generic exemption.

The canonical entry point is:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_v15.py register \
  --root /scratch/djy8hg/workdir/fox-review-v15/NEW_IMMUTABLE_RUN --execute
```

`register` without `--execute` creates only a disabled registration. Every
producer and analysis writes once. Scope, source, environment lock and parent
evidence are hashed. Failed or unacknowledged submissions are not blindly
retried. CPU jobs use `bii,standard`; the two consumed-input GPU workers request
one A100, 12 CPUs and 20 minutes each through `bi_fox_dgx`. Aggregate project
A100 occupancy must remain at most six. All caches and temporary files stay on
scratch. No package installation, model download or credential persistence is
part of this scope.

Each lane has an `afterany` analyzer and terminal receipt, including failures.
The final report is submitted only after all finite lanes reach a terminal
decision, and waits for acknowledged producers even if analyzer submission
failed. Aged terminal scheduler IDs are reconciled once with accounting;
fresh acknowledgements do not depend on immediate database visibility. There
is no queue/log polling loop, automatic GPU retry or autonomous fresh-cohort
promotion.

## Claims still blocked

Neither conditional byte witnesses nor initial-only cache equality establishes
matching realized initial, final and peak storage. A whole-allocator envelope
is not KV-only peak accounting. The existing two consumed inputs do not
establish benchmark quality or statistical superiority. Missing historical
video/frame journals remain external provenance gaps even after every cached
image pool is hashed. Fresh confirmation, qualified published operating points,
independent quality, strongest-baseline latency superiority and six-component
benefits require their own positive evidence. Negative controls are retained.
