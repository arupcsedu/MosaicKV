# Head-local protection and asymmetric precision: bounded consumed diagnostic

EviCastKV and MosaicKV name the same system. This is a separately labeled
training-free precision variant, not the validated graph/exact-only paper path.
Original selected positions, original K/V or actual LOOK-M merged K/V, logical
positions, mandatory last prompt token, and append-only generation remain fixed.

## Motivation and attribution

The existing packet protects the union of selected text slot indices across
all heads. A text slot in one head can cause an unrelated visual slot in another
head to be stored losslessly. Test protection per head instead; retain the
original first four/last eight selected entries, text values, boundary layers,
and existing <=0.5% correction exceptions. This is lossy for formerly incidental
visual protection and therefore needs actual model-level checks.

Also test 2-bit quantization separately on keys/values. Quantization axes follow
the existing per-channel keys/per-token values design; that general principle
is established by KIVI (Liu et al., ICML 2024), not claimed novel here:
https://proceedings.mlr.press/v235/liu24bz.html . No official KIVI code is invoked.

## Frozen arms and fairness

Run every method (Prompt Top-K, local PrefixKV/VL-Cache/LOOK-M reimplementations,
and the EviCastKV profiled-prior selector) under all five policies: unchanged
union 3/3-bit reference, head-local 3/3, head-local 3/2, head-local 2/3, head-local
2/2 (key/value bits). Do not choose a weaker baseline or deny it the common
codec. Do not alter the .5 raw source-slot capacity or source profiles. Actual
encoded bytes differ: these are representation/Pareto diagnostics, NOT shared
physical-byte-matched quality comparisons.

Use the same two consumed Description/MM-Vet inputs from the immutable parent
canary; no new identities or annotated answers are loaded. One pinned BF16
LLaVA-1.5-7B HF eager prefill is reused; normalization/query captures and the
unchanged selectors construct all packets before scoring. Two boundary layers
at each end remain entirely lossless. Packet buffers, indices, minima/scales,
exact payloads, history, preallocation and active scratch leases all count.

## Correctness and measurements

CPU tests must pass before GPU admission. Each GPU worker first checks all
lower-bit decode and head-local scatter paths against the CPU oracle, including
all 65,536 BF16 bit patterns and untouched history sentinels. Retention 1 remains
the existing untransformed FullKV bypass; the new format is compressed-only.

For every method/policy:

1. Replay the first 32 saved FullKV-generated tokens to compare 32 affected
   logits at identical prefixes. No annotated benchmark answers are used.
2. Decode its own complete 128-prediction greedy trajectory, with common raw
   BF16 history, no history eviction/quantization, and no FullKV fallback.
3. Require unchanged-union control to reproduce saved lossless-transport
   3-bit logits within .001 and token agreement 1. Preserve all failures.
4. Preserve the registered same-prefix guard relative to the union anchor:
   all 32 greedy tokens identical, FP32 KL <= .02, maximum logit error <=1.0.
   Report FP64 KL as a supplementary diagnostic, never a replacement guard.
5. Record full greedy-token identity, exact prompt bytes and decode-owned
   initial/final/observed peak backing, and decode CUDA/wall timing. The full
   prefill is retained for diagnosis, so no whole-request memory or end-to-end
   latency claim is allowed. Changed answers cannot be called quality-preserved
   solely because a short-prefix gate passes.

A common policy is eligible for further consideration only if all ten
method/input combinations pass transport/isolation, retain complete own-greedy
token identity, pass the unchanged prefix guard, and reduce prompt backing.
That is a representation diagnostic, not a paper result or quality superiority
claim. No fresh cohort or large sweep is automatically authorized here.

Six one-A100 jobs, each 12 CPUs and at most 30 minutes, divide two consumed
inputs into three method groups. Six aggregate A100s is the ceiling; submit only
when no other owned GPU work exists. A finite afterany CPU reducer checks all
artifact hashes. All caches stay under scratch and dependencies stay in the
existing common environment. If a policy fails, do not relax thresholds or
increase cohort size to conceal the failure.

## Decoder-oracle correction (before model execution)

The first array (22216180) stopped before any model load: its oracle constructed
two independent quantized packets on CPU and CUDA. A one-GPU diagnostic showed
the unchanged union packet also differed: CPU/CUDA division at half-bin ties
can select adjacent quantization codes. This is not evidence of a new scatter
failure. Preserve that negative initialization result.

The corrected decoder oracle copies the exact CUDA-encoded codes, minima,
scales, protected values and indices to CPU, then requires bitwise identical
reconstruction from those identical bytes. It does not relax a tolerance or
change the CUDA quantizer. The saved model-logit transport check remains .001,
and all quality guards are unchanged. Test this correction on one GPU before
retrying model workers under a new immutable root.

## Precision-specific correction follow-up

The corrected first study completed all 50 rows, but no policy met every
unchanged same-prefix/greedy-identity gate. Head-local protection saved less
than one percent; low-bit variants saved about 7--14 percent but generally
changed later output. Preserve this as a negative consumed-input diagnostic.

The next bounded study retains only three executed policies: unchanged union3,
head-local 3-bit K / 2-bit V with global correction reranking, and the same
format with deterministic per-head correction quotas. Keep the exception
allowance identical to the original funded patches; do not increase its bytes.
Rank scalar errors using the actual lower-bit reconstruction and the existing
prompt-query/value-demand weights, not errors left over from the 3-bit packet.
No draft outputs, future answers, benchmark labels, or weight updates are used.
Global reranking minimizes weighted squared scalar error for that same count;
this is not a guarantee of logit or benchmark improvement. Per-head quotas are
a separately labeled ablation and have no global-error optimality guarantee.

Apply both variants to every selector, use the same consumed inputs and all
unchanged guards, preserve baseline merges, and bind the previous negative
decision by hash. Old low-bit arms are not rerun. No fresh cohort or larger
campaign follows automatically. Numerical decoder controls now also cover the
new correction patches using identical encoded CPU/GPU bytes.

## Whole-value-row follow-up, frozen before COCO diagnostic results

Keep union3 and global scalar refitting as controls; add one whole-value-row
variant on the SAME two consumed inputs. Remove the V scalar patches and spend
no more than their actual bytes on losslessly encoded complete V rows. Keep
the K scalar corrections unchanged. Rank rows by maximum individual prompt
query's retained attention squared times that row's quantization-error norm
squared. Queries are not pooled before the maximum. This can preserve a
coherent high-demand value vector rather than scattered scalar entries, but
neither the score nor additional exact coverage guarantees task improvement.

The lossless packet and its flat head/token map count. Search actual encoded
prefix costs for the largest affordable ranked prefix; do not increase the
reference exception-byte allowance or cache capacity. The number of exact
scalar values can increase, and that changed coverage is explicitly recorded;
the old 0.005 scalar fraction defines only the reference byte allowance here.
Both packet sizes are monotone for
append-only ranked prefixes, including raw fallback. All original positions,
text/boundary protection, LOOK-M merged values, history, and unchanged 32-token
guard remain. Every baseline receives the option. Run only after the active
COCO array/analysis finishes, so aggregate use never exceeds six A100s. No
COCO answers or failure identities inform this constructor, and no fresh
evaluation or larger campaign is enabled by this diagnostic.
