# Bounded patched LOOK-M/HF generation transport

This is **`patched_official_hf_compat`**, not the unmodified official release
or `lookm_reimpl`. The pinned upstream is
`SUSTechBruce/LOOK-M@ecf0f51a9c416c2d85e47faf2638502f01a6d748`.
Its original import failure and the V11 import-only diagnostic remain separate.
No upstream checkout, installed package, attention operation, merge equation,
parameter, or model weight is modified.

The existing isolated patch removes only three proven dead references: the
missing import, unread dictionary, and unused plotting import. The new dynamic
subclass adds the installed public `GenerationMixin`. It explicitly declares
that the upstream uses legacy tuple K/V caches. This declaration overrides
`_supports_default_dynamic_cache` to return `False`: Transformers 4.55.2's
`_prepare_cache_for_generation` documents that escape for legacy-cache models.
Without it, the new generation driver would insert an incompatible DynamicCache.
The installed generation-source hash and precise package version are recorded.
This is compatibility transport, not a new attention implementation or silent
conversion of cache arithmetic.

## Synthetic CPU controls

The bounded launcher uses randomly initialized CPU weights only. Its tiny model
has one layer and two 128-dimensional heads; the head dimension preserves the
upstream merge implementation's explicit 128-dimensional assumption.

1. On a 16-position synthetic text prompt, compare the upstream generation
   override in the compatibility subclass against explicit greedy forward calls
   on the same model. Require identical four generated tokens, bitwise-equal
   raw logits, identical legacy-cache lengths and identical logical positions.
2. Exercise the actual upstream `TextPivotLlamaAttention_drop` and
   `TextPivotMerge_LayerWise` on 584 synthetic embedding positions containing
   the documented 576-position synthetic image span. The supplied embeddings
   deliberately bypass the absent image encoder, not attention or merging.
   Verify finite logits, real compression, and logical decode positions
   584, 585 and 586 despite a shorter physical cache.
3. Reject incomplete call traces, cache-class substitution, position
   renumbering, changed forward/preparation methods or unreviewed generation
   interfaces. Stop at the first actual interface failure; no dependency
   guessing, aliases or additional arithmetic patches are permitted.

This is availability testing, not a checkpoint reproduction, metric result,
quality comparison, memory superiority or timing result. A real checkpoint,
real media preprocessing, original budget convention, numerical controls,
independent references and a registered published operating point remain
required for a paper-native reproduction claim.

### Observed bounded result

The CPU diagnostic passed on 2026-11-10 using the unchanged common environment
(PyTorch 2.8.0, Transformers 4.55.2). Four FullKV greedy tokens exactly matched
the explicit-forward reference, with maximum raw-logit difference **0.0**.
The native and explicit cache transactions agreed: physical lengths 16–19 and
logical last-input positions 15–18. The actual upstream compressed path reduced
the synthetic 584-position prefill to **116** positions, then grew to 117, 118
and 119 while consuming logical positions 584, 585 and 586. Logits were finite.
The pinned upstream checkout remained unchanged. These random-weight synthetic
numbers must never be used as benchmark quality, memory or latency results.

The generation driver is bound to source SHA-256
`6de1a0fbf0b26352cbd93a2f494fc5fca1856e6636d20947e81f7f83ace7dec5`.
The import patch remains
`ed8d0b2ad29472c6c298937cbb83e771cb5803d484c7f6e40935cdb31389e360`.
Changing either the driver or its version requires a new explicit compatibility
review, not a silent fallback.

## Reproduction

Use the common environment and source `mosaickv/scripts/cache_env.sh`. The
launcher copies the pinned source into an ephemeral scratch directory, applies
the hash-bound dead-import patch there and runs a CPU-only offline subprocess.
It never edits the original checkout or accesses model weights.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/audit_lookm_hf_generation_compat.py \
  --upstream /scratch/djy8hg/workdir/fox-review-v11-code/third_party/LOOK-M
```

The V11 source may be read while its immutable jobs are queued; it must never
be edited. The launcher emits a structured JSON result, including any precise
first interface error. Synthetic success cannot authorize a larger campaign.

## Prepared real-checkpoint control: READY_NOT_RUN

`evicastkv_lookm_hf_checkpoint.py` prepares a separate bounded control. It has
**not run on a GPU**, and supplies no pretrained-quality or published-baseline
reproduction evidence yet. Its API is `prepare(root, cpu_gate_root=...)`,
`worker(root, index=0)`, and `analyze(root)`. Its write-once summary is
`GATE_lookm_hf_checkpoint_compat.json`.

The worker requires one A100, 12 CPUs and a 20-minute hard Slurm limit, with a
1,080-second soft deadline. It admits only the two existing consumed prompt/media
inputs bound by the prior plan's hash. It reads no reference answers. Preparation
requires the complete current-source CPU suite and an executing immutable parent
registration, reruns and seals the bounded synthetic compatibility control on
that source, then hashes every local model shard, tokenizer/processor file,
consumed image, upstream source dependency and compatibility source file.
There is no submission command or automatic larger campaign in this module.

The locally cached original-format checkpoint is
`Zuyan/ElasticCache/llava-v1.5-7b`, repository revision
`833edbdc7512240f2a3aa49feeb9468e2297bdbc`, with weight-index SHA-256
`e843c542d65da614b748791823c2af0aa97168b969fcb3e56235b663a9fb5d19`.
It is **not silently identified with** the HF-converted `llava-hf` checkpoint
used by the earlier source-variant experiments, or with the checkpoint named by
the LOOK-M paper. The cached CLIP tower is bound to
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. Its local path replaces only the
vision-checkpoint location; no vision or attention arithmetic is patched.
The official worker's `llava_llama_2` conversation and image-processing route
are source-bound. FP16 eager language attention, fixed 16-token greedy output,
and disabled EOS are explicit diagnostic controls, not an assertion that every
paper execution control has been reproduced.

The prepared checks are:

1. Load the actual upstream class through the compatibility subclass using
   public, local-files-only checkpoint APIs. Instantiate the public CLIP graph
   before state loading so all 391 checkpoint vision keys are consumed, rather
   than discarded by the original lazy tower. Compare every loaded vision
   parameter word with the registered combined checkpoint in bounded CPU
   transfer chunks. Reject missing, unreviewed unexpected or shape-mismatched
   parameters and any need for embedding resize.
2. Compare native FullKV generation with explicit full-cache greedy forwards
   for 16 tokens on each consumed input. Require identical tokens and maximum
   logit difference at most the preregistered `1e-4` threshold.
3. Construct the actual upstream text-prior-pivot attention modules with
   strictly copied original projection parameters. Execute compressed requests
   A–B–A, resetting scores through the upstream cache-cleaning method. Require
   exact repeated-A logits/tokens, actual compression, finite outputs, original
   logical positions and append-only cache growth.
4. Restore the original attention objects and repeat FullKV A. Require its
   tokens/logits to match the initial FullKV A exactly.
5. Save independent parity tensors and per-request cache traces, including
   partial completed work when a later phase fails. The CPU analyzer recomputes
   token/logit/isolation checks and rejects incomplete cohorts rather than
   intersecting available rows.

Each observed cache boundary records unique underlying backing bytes, not only
view sizes. The reported peak is explicitly the maximum **returned-output cache
backing observed at decode boundaries**, not transient decode-owned or
whole-request KV peak. Construction overlaps, allocator reservations and
unseen temporary storage remain outside that narrow boundary. No timing or
memory-superiority claim is authorized by this compatibility diagnostic.

### Source-native serialized-buffer criterion

The combined cached checkpoint has 718 keys. The original lazy graph exposes
only 295: 391 vision keys and 32 old rotary-frequency buffers were unexpected.
The state-aware compatibility constructor exposes all **686 actual model
state keys**, including every vision parameter, before strict loading. This is
an explicit loader deviation: it uses the combined checkpoint's vision values,
not an assumed-equal external CLIP weight reload. Attention, vision-forward and
merge arithmetic are unchanged. Equality with the external CLIP checkpoint or
the paper's model is not claimed.

The initial stricter serialized-frequency value-equivalence check is preserved
as **negative**. Layer 0 stores BF16 `[64]` frequencies that differ at 14 words
from the configured FP32 frequency cast to BF16, with maximum absolute
difference `0.00048828125`. No numerical tolerance is relaxed and no rotary
buffer is overwritten to hide that discrepancy.

A separately reviewed compatibility criterion follows the pinned source's
actual unused-key semantics. Its per-attention `LlamaRotaryEmbedding` registers
`inv_freq`, `cos_cached` and `sin_cached` as nonpersistent. Its constructor
derives phase caches from configuration; within the prebuilt range, forward
only slices those phase caches. Loading an unexpected serialized `inv_freq`
through the actual module must leave **all buffer bits, pointers and output
phase bits unchanged**. This must pass independently for all 32 actual modules,
with exact class/source/configuration hashes, or GPU execution remains blocked.
The previous failed value-equivalence result remains a separate field.

The two consumed prompts have 646 and 650 expanded positions under the pinned
official conversation route. Their 16-token controls require lengths at most
666, strictly below the actual pinned attention's 2048-position caches. Runtime hooks reject any
larger position, changed cache pointer/layout or phase-cache rebuild. The
FullKV 16-token `1e-4` logit and exact-token requirements, compressed A–B–A
identity and restored-FullKV identity are unchanged. This is
`source_native_ignored_serialized_buffers` loader compatibility, not a claim
that obsolete stored frequencies equal current configured frequencies, an
unmodified official execution, or a published operating-point reproduction.

The actual 32-layer CPU meta/mmap audit passed this source-native criterion:
686 consumed keys, all 391 vision keys accounted for, and all 32 obsolete
serialized-frequency entries independently proven irrelevant to the bounded
forward. The failed serialized-value-equivalence result remains explicit in
that report. A separate tiny random-checkpoint call through the **actual** HF
`from_pretrained` API, with the same explicit-config pattern used by the real
worker, also preserved materialized configured phase buffers despite a
deliberately conflicting nonpersistent frequency entry. Neither control loads
dense pretrained weights into login-node RAM or executes on a GPU. Actual
pretrained 16-token parity, all-391 loaded vision-value checks and request
isolation remain READY_NOT_RUN.
