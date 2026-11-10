# CPU closure of the saved request-wide allocator audit

This derived review addresses the six completed consumed-input ownership
captures from `fox-review-v6-parallel-controls-20261110/ownership`. It does
not rerun model inference, change cache construction, or promote a candidate.
The original raw reports and negative certification flags remain unchanged.

## What is certified

The boundary is the time-aligned peak of **PyTorch allocator requested backing
bytes** assigned to `kv_persistent`, `kv_workspace`, or
`kv_shared_workspace`. It charges original and packed cache owners, language
attention scores/context and masks, forecasting, evidence-graph construction,
and cache packing/verification buffers identically for every method.
Storage aliases count once; pointer reuse creates a new generation. Storage
pending free remains live until `free_completed`, not `free_requested`.
Phase maxima are never summed. The report also decomposes the simultaneously
live roles at the earliest peak sequence.

This is **not** a whole-device memory certificate, an allocator-reserved-byte
measurement, or a payload-only active-cache number. It excludes allocator
rounding/reservation, CPU residuals, and vendor allocations outside the
PyTorch allocator. A certificate does not assert a memory advantage: graph
and forecast construction can outweigh reductions in the decode-active cache.
Keep these boundaries separate in paper prose and tables.

## Evidence and fail-closed classification

The producer SHA is fixed to
`7cf31bf285ea7b5f4ae7d7caa615921f2e89f417`. Every local source used by a rule
must match its Git blob at that SHA. Every installed Transformer/PyTorch
source used by a rule must match the explicitly reviewed SHA-256. The
protocol records the source line, reason, file/function/line match, and all
six raw capture hashes. Rules cannot exempt generic `nn.Linear` or broad
`model.forward` calls. A same-named file at an unbound path is rejected.

Unknown allocations remain unknown and produce lower/upper peak bounds plus
concrete unresolved stack groups. Rejected timestamp marks are never assigned
to a guessed or latest pointer generation. Existing `ownership_review.review`
may supersede a rejected mark **only** when all generations have independent
exact stack attribution. Original timestamp blockers remain visible in the
derived report. Missing history, incomplete requests, overlapping generations,
and inconsistent certificates cannot be repaired by classification.

The independent `valid_whole_request_certificate` reducer recomputes the peak
from generation intervals; pass booleans alone are insufficient. Regression
tests cover alias observations, address reuse, pending frees, unknown roles,
source-hash drift, same-suffix source spoofing, forged certificates,
timestamp ambiguity, and simultaneous role accounting.

## Canonical CPU execution

Use a clean immutable worktree and the common environment. Choose a new
output root; neither protocol nor reviewed reports can be overwritten.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_ownership_closure.py prepare \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-ownership-closure-NEW \
  --parent /scratch/djy8hg/runs/evicastkv/fox-review-v6-parallel-controls-20261110
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_ownership_closure.py run \
  --root /scratch/djy8hg/runs/evicastkv/fox-review-ownership-closure-NEW
```

The Python orchestration API is `prepare(root, parent)` followed by
`run(root, parent=None)`. No GPU, dataset access, download, fresh-sample
authorization, or package installation is required. Full reviewed ledgers
are stored as write-once gzip JSON with content hashes. The CPU completion
gate and exact-memory certification verdict are distinct: a complete CPU
review may still leave explicit unresolved ownership and block certification.
