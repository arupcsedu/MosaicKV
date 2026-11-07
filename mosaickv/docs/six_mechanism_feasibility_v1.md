# Six-mechanism feasibility gate (separate from the validated storage variant)

This is a bounded consumed-input diagnostic, not a paper confirmation or a
promise of superiority. The immutable storage producer `a6f689f` and its paper
results remain unchanged. New work lives in its own worktree/branch.

## Execution

Use the pinned cached LLaVA-1.5-7B checkpoint, common environment, BF16,
eager attention, batch one, 16 greedy predictions and one-token blocks.
The shared **physical prompt K/V cap is 150 MiB**. Generated cache growth,
preallocation, residual CPU memory and diagnostic source snapshots are reported
separately; this is not a 150-MiB whole-request or lifetime cap.

One route out of 1,024 participates in the bounded graph/utility/lazy-greedy
planner and RoPE-safe prototype/residual construction. Other routes retain
same-capacity exact Prompt Top-K entries. This explicit route-limited
feasibility variant does not prove all-head or six-component superiority.
The original prompt is prefetched only once per request; the hybrid forecast
uses an isolated two-step draft from that prefill. No benchmark references
enter the model-facing artifact or selection.
This correctness-first diagnostic captures native full eager prompt attention;
it does not substitute the optimized query-window attention kernel. Forecast
statistics still use the registered last-16 prompt window. Actual model RoPE
position inputs, not just Python counters, are checked on every observed decode.

Two canary workers each use one A100 and 12 CPUs, with a 30-minute limit.
Aggregate admission uses the existing tested four-A100 controller and an
array limited to two workers. A CPU `afterany` analyzer runs even on failure.
It submits the four-input small component study **only if all canary gates
pass**. No fresh cohort or large campaign is automatically authorized.

The first GPU attempt stopped at the processor boundary before producing any
predictions because raw image paths were passed to the pinned processor.
The correction verifies checksums and decodes RGB PIL images, then tests the
actual pinned cached processor on CPU before any GPU retry. `use_fast=false`
matches the checkpoint's saved slow processor. The failed attempt and its
immutable source remain preserved.
The corrected diagnostic uses the generic tier-payload packer so promotion
reads the actual restored payloads, rather than the fused original-source
gather. Both controls and ablations use that same registered packing policy.

## Required evidence (no flags-only passes)

- Explicit FullKV versus ordinary `generate` and retention-one packed parity
  over at least 16 tokens: token equality, maximum absolute logit error <=0.01.
- Independently executed FullKV/retention-one runtime requests must also have
  identical prefill and decode-logit hashes, rather than sharing one prefill.
- Actual hybrid forecasting, nonzero cross-modality graph edges, nonzero
  value contribution and lazy-greedy calls, real exact/prototype/residual
  payloads, captured model-derived RoPE safety, no exact-only fallback.
- Lossless pinned CPU residuals and live uncertainty-controller calls.
- A separately labeled controlled-entropy repair transaction with nonzero
  asynchronous restored bytes and exactly one current-token recomputation.
  Promoted GPU K/V are compared bitwise to the original source entries.
  The threshold is deliberately zero for that stress control; it is not a
  learned or validated natural trigger and cannot establish quality recovery.
- Independent checks of mandatory positions, source membership, original
  logical/next positions, finite logits, packed physical prompt bytes,
  underlying tensor/backing accounting and draft isolation.
  Registered inactive source/snapshot ownership is additionally reported, not
  silently omitted from the separately labeled registered-owner peak. This is
  still not complete process-memory or whole-request KV-peak certification.
- A-B-A requests through the same adapter, requiring identical A tokens,
  exported logit hashes and storage endpoints after the intervening B request.
- Complete manifests and failure records. A scheduler exit code is not a gate.

## Conditional small component study

Four additional already consumed images use FullKV, Prompt Top-K, the full
route-limited path, prompt-only forecasting, coverage-reward-off,
value-contribution-off, exact-only and no-repair arms. Every compressed arm
uses the same inputs, generation, precision, eager runtime, block/mandatory
policy and physical prompt cap. Realized payload and backing storage must
remain visible; no nominal-ratio comparison is called byte-matched.
Coverage-reward-off does not disable graph descriptors needed by prototypes
and value novelty. The forced repair arm is excluded from quality ablations.

The observers synchronize and inspect tensors, so reported times are
instrumented diagnostic costs, not headline latency. Token agreement is not
an official COCO task score. This engineering gate can establish operational
mechanisms and meaningful state changes, not benchmark superiority.

## Launch

After the scoped source and tests have a clean immutable SHA:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/six_mechanism_feasibility.py \
  prepare --root /scratch/djy8hg/runs/evicastkv/six-mechanism-feasibility-20261106-v1
```

The preparation step runs CPU tests, verifies the parent ledger/media hashes,
removes references from model-facing inputs, freezes all arm configurations,
and submits the finite event-driven chain. No package installation, model
download, historical-artifact overwrite or storage-variant mutation occurs.
