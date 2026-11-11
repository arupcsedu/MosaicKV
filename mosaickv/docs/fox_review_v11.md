# V11: bounded source closure

This study preserves the validated storage variant and all negative results.
It uses only the same two consumed inputs, not a new benchmark cohort.

1. Require complete CPU verification of this clean source.
2. Recover historical generated configs using a separately identified trusted
   reconstruction snapshot, exact canonical digests, byte-identical serialized
   content and immutable ID-ledger bindings. Do not relabel that snapshot as
   the absent original producer or claim global media freshness.
   Then fork the eight saved checkpoint states into new immutable roots for
   one bounded frontier/recovery segment each. Use the reviewed config overlay,
   retain original errors and producer identity, and verify all imported
   hashes. Eight one-CPU workers run at most one segment each; the final
   report depends on their afterany aggregation. Incomplete frontiers remain
   incomplete; no automatic unbounded scanner loop is authorized.
3. Execute a four-arm prompt K/V factorial control, with original versus
   reconstructed K and V. Preserve the same lossless generated-history policy
   and frozen FullKV prefix. Both original prompt components remain stored
   and fully charged in every arm. These higher-memory counterfactuals are
   not compression candidates or byte-matched comparisons.
4. Construct the saved mixed-layer protection witnesses at exactly 45.5 and
   49 MiB initial backing, and the existing uniform BF16 Prompt Top-K at the
   same initial targets. Deduplicate only bitwise-equal cos/sin/union/slots
   headers. Preserve selected packet values, mandatory tokens and logical
   positions. Measure final and decode-owned peak bytes separately; equal
   initial backing is not equal final or peak. This new mixed-protection
   variant is not the historical uniform source policy. Constructor transients
   and the cost of building all three protection families remain charged.
5. Capture the actual nonstreaming source ownership timeline, including
   original K/V, captures, all candidates, retirement, history and workspace.
   Preserve pointer generations, complete allocator stacks, source hashes
   and unknown allocations. Observed checkpoints do not certify unseen
   transients, external CUDA allocations, or the whole device.

GPU stages run sequentially, one A100 and 12 CPUs each. Causal and ownership
workers have 20-minute limits; construction has a 30-minute limit. The
aggregate ceiling remains six A100s. Each producer has an afterany analysis
callback, including when it fails. These accounting studies are independent:
advancing them after a negative quality result is not candidate promotion.

The original 32-token, KL and logit guards remain unchanged. The raw/raw
counterfactual must agree with the reference before interpreting component
effects. Source construction receives no forced prefix, labels or answers.
No FullKV fallback, dummy allocation or silent byte underfill is accepted.

No confirmation or larger campaign is launched by this finite protocol.
Independent quality, source fidelity, real shared-byte curves, exact source
peak certification, latency superiority and useful six-component effects
each require their own evidence. Execution permission is not the missing
scientific gate. New evidence may be negative.

## Reproduction

Initialize the six pinned submodules and use the common environment. All
cache, build and artifact paths must remain on scratch. From a clean checkout:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_v11.py register \
  --root /scratch/djy8hg/workdir/fox-review-v11/run-20261110-1 --execute
```

Omitting `--execute` registers a dry run. Every attempt is write-once;
never alter its active source snapshot or blindly resubmit a lost scheduler
acknowledgement.

## Baseline availability

The isolated LOOK-M patch removes only statically unused missing imports and
an unread map. The pinned upstream checkout is unchanged. CPU import and a
tiny random-weight forward/cache smoke pass under the common environment.
This is an explicitly patched official-source diagnostic, not an unmodified
official run, pretrained parity, benchmark reproduction or permission to
relabel local `lookm_reimpl` results. The legacy generator's missing
`GenerationMixin` remains separately recorded. Existing official PrefixKV
PPL reproduction and its limits are linked in the baseline-readiness audit.
