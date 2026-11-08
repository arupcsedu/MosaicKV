# Exact FP32 scale headers and complete predicate: consumed-input gate

This is a common transport variant, not a renamed six-component algorithm.
EviCastKV and MosaicKV are the same system. The original minima are already
BF16; the new study must not claim that saving again. Default readers remain
unchanged outside the explicit diagnostic context.

## Evidence and frozen correction

`exact-header-predicate-20261108-v1` passed 182 CPU tests and both real CUDA
operator controls. The component-wise model call supplied `chunk_scalars=524288`,
which the wrapper unnecessarily rejected despite an explicit full-span policy.
The failed model rows and partial ownership ledgers remain immutable. The new
wrapper accepts a positive hint **only if** the registered predicate policy
resolves it to the complete tensor span. The two original asymmetric FP32/BF16
predicates, tolerances, protected-token rejection and finite checks are unchanged.
The regression reproduces that component-wise call and rejects a truly chunked
policy. The two successful original control rows/header captures are reusable;
the failed fusion rows are not successful model evidence.

## Feasibility before model execution

Use only the two consumed COCO inputs and actual captured FP32 scales. Encode
their original integer bit patterns with a raw/full-word/split-word dictionary
choice. Charge payload, palette, high words, selector, and the largest FP32
replay workspace **per stream role**. Decline the complete conversion if these
costs erase savings. Raw headers retain their original owner, without pretending
to compress them. CPU encoding, D2H/H2D transfers, and temporary overlaps are
included in request timing and the ownership trace; they are not offline work.

## Request-private replay

Immutable packets retain no original scale owner for a compressed header.
Readers bind the same original FP32 values into request-private scratch, counted
through the existing `Replay.scratch` inventory. No scale rounding, learned
parameters, selection change, prototype or residual is introduced. K and V
reuse a header decoded once in the existing key-reader graph. Header decoding
is inside captured replay, not a one-time warmup substitution. CPU tests cover
ordered eager/prepared/captured replay, owner retirement, role isolation,
request isolation, exact storage and raw/complete-transaction fallback.

## Finite execution and acceptance

`exact_scale_replay_study.py prepare` submits CPU tests and actual-header cost
analysis, then two A100 operator workers only if feasibility passes. Both
operator gates must pass before two model workers execute corrected fusion and
fusion plus exact headers. Every producer has an `afterany` reviewer. One A100,
12 CPUs, 48 GiB and at most one hour per worker; aggregate A100 admission stays
at six. All caches remain under scratch. No new identities or larger campaign.

Require the original 32-token KL/logit/token gate and all 128 vocabulary logits
bitwise identical to the frozen storage-only parent. Independently recheck
ownership ledgers and report initial/final/request-wide KV-plus-workspace peak.
All three boundaries must decrease before considering another storage design.
Requested PyTorch bytes are not rounded allocator usage, vendor workspace or
process HBM. Single instrumented model trials cannot establish clean latency
superiority or independent benchmark quality.

## Common application

The predicate and header contexts are selection-policy agnostic: compatible
baseline transports and the storage-only comparator get the identical context.
Raw exact baselines lacking affine headers receive no fictional header saving.
Any later comparison must measure them again under the same enabled contexts;
historical unoptimized baseline latency must not be used as the comparator.
Micro-prototypes and selective history remain blocked pending this gate.
