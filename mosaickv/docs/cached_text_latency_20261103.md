# Exact text-row caching: consumed-input engineering diagnostic

EviCastKV and MosaicKV name the same project. This execution variant disables
the six principal algorithmic mechanisms; it is not a new algorithm quality
claim. Reuse the two already consumed Description/MM-Vet inputs only.

## Evidence and intervention

The repaired real-model traces contain 23 text-overlay projections per K/V
component per decode step. They account for approximately 7.6 ms of summed
instrumented kernel duration; this is not a measured end-to-end speedup.
Whole-layer caching within 6 MiB could cache only one EviCastKV layer and failed
the paired latency gate. Prefetch was slower. Fused attention failed actual-query
word equality. Preserve all three negative results without retries or relaxation.

Cache only the original decoder's selected head-local text K/V rows using the
existing lossless BF16 codec or resident BF16. Remove their redundant normalized
text-source representation. Keep the original non-text source, original K/V
bits, memberships, logical positions, generated-history policy, and eager
attention arithmetic. Never insert an absent text membership into a head.
Retain/count shared base data; count all direct bodies and row indices.

## Frozen finite sequence

1. Complete the common CPU suite from a clean SHA.
2. Profile five operator choices on every eligible layer of every compared
   method: source, and encoded/resident K/V combinations. Five warmups and twenty
   repetitions, layouts with 0/64/127 history positions. Noneligible layers are
   excluded constant objective terms, **not zero-time measurements**.
3. Exact multiple-choice planning allows 0 or 6 MiB additional private storage,
   includes construction plus 127 reads, and reserves cross-layer aliases once.
4. All five methods get audited full requests: MosaicKV, Prompt Top-K, PrefixKV,
   VL-Cache and LOOK-M local representation adaptations. Require uint8 word
   equality of all 128 parent logits, unmodified selected K/V, history sentinels,
   and actual initial/final/decode-owned-peak bytes within the allowance.
5. If meaningful EviCastKV policies pass on both inputs, time the frozen policies
   for EviCastKV and PrefixKV with five warmups and twenty randomized paired
   repetitions. Include conversion and construction. Include descriptor reuse
   and the previous strongest whole-layer PrefixKV control on the same source.
6. Require an adjusted upper ratio <=0.99 against own reuse and <=1.05 against
   **every** measured PrefixKV arm. Report the stricter <=1.0 gate too. Use
   10,000 paired repetition bootstraps and a fixed maximum family of 48 contrasts
   over both consumed inputs; intervals are not independent-example inference.
7. Only a passing common policy authorizes the finite all-five-method timing
   follow-up, using already frozen operator policies, never online retuning.

One A100 and twelve CPUs per worker, maximum two concurrent workers (aggregate
cap six A100s), <=105 minutes per worker, reservation `bi_fox_dgx`. CPU partition
`bii,standard`, two CPUs. `afterany` analyzers check hashes and preserve failures.
No fresh materialization, calibration, paper campaign, native serving claim,
whole-request GPU-peak certification, or independent quality claim is authorized.
Torch allocator peaks are reported separately from decode-owned backing bytes.
