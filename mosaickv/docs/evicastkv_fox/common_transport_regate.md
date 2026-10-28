# Common-policy correctness and bounded missing-media recovery

The two consumed padding failures passed the separately labeled valid-slot
execution diagnostic: sixteen steps per input, zero observed maximum logit
difference, and unchanged 16-token ordinary-generate/reinjection gates. Native
failures remain negative. This is a new transport policy, not a reclassification
of the old arithmetic or evidence of compression/quality superiority.

Before any adoption, run the same two inputs through the exact-method
correctness runner at 16/40 MiB. Both Prompt Top-K and EviCastKV share
`arithmetic.attention_layout_policy=valid_slot_decode`. That complete arithmetic
object is already included in each row's resolved config hash and environment
manifest, so the backend choice is not hidden behind an unchanged runtime-only
config hash. The installed eager function still executes unchanged over the
retained entries; weights, precision and stored K/V are untouched.

The independent original-layout reference can have different selected indices
per KV head. Support that case only with equal valid capacities and consistent
query-head-to-KV-head ownership. Gather each head's valid entries in source
order and restore probability columns on the original physical axis. Unequal
capacities and nonbinary biases fail explicitly. All-valid calls retain the
native object-identity path. The worker's wrapper is scoped and always closed.
This policy is correctness-only: scoring and timing reject it until separately
registered. Its validation is not an exact-source oracle for LOOK-M's transformed
cache and must not be described as LOOK-M fidelity reproduction.

The old raw rows, failed numerical controls and development identities are not
replaced. The eight-request regate stops automatically; it cannot launch
quality evaluation or confirmation. Configuration freezing remains an
engineering lock only. A data-freshness failure cannot be solved by changing
the numerical transport.

## Missing IEdit assets without a multi-gigabyte cache copy

The pinned part-2 archive is 2,018,154,102 bytes and begins with `IEdit` (verified
from a bounded remote header read). Stream only that first task until every
catalog-declared source image and montage is fingerprinted. Enforce a 512 MiB
compressed-prefix read limit, with at most one additional in-memory readahead
block. Do not download/persist/extract the complete archive. Parse only the
leading annotation question-type metadata; do not decode response arrays.

Bind the immutable dataset revision, expected full archive SHA and each
observed asset hash. **Do not claim the full archive SHA was measured:** a
prefix read cannot verify it. Record the actual prefix bytes consumed and
missing/overlapping assets. This resolves asset availability only. Unknown
global video/document aliases and non-exhaustive historical coverage remain
freshness blockers; there is no confirmation successor.
