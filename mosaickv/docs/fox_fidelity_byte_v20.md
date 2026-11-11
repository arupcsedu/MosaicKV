# V20: source fidelity and realized-byte repair controls

This is a bounded engineering diagnostic on the same two consumed Description
and MM-Vet inputs. It is not fresh evidence or a paper-scale campaign. Main
checkout, historical source packets, manifests, and negative results are unchanged.

## Actual causes

The V11 low-byte source quantizes normalized projection inputs to three bits.
Its transport patches target native projection of **that quantized input**, not
the original FullKV tensor. All four source rows met registered KL/logit bounds
but changed some greedy predictions. They cannot be called lossless.

The previous history codec has data-dependent correction sizes. Identical
forced tokens do not imply identical hidden states or encoded history sizes.
Its source reader also leases buffers for all 600/604 prompt positions, while
selected Top-K caches contain only 91/98 positions. This explains the roughly
8 MiB peak gap; it was not simply a reporting mistake.

## Distinct controls

1. **Original-bit exact selection:** gather original post-RoPE BF16 K/V,
   keep original logical positions, and compare the transport with an independent
   native `DynamicCache` holding the same selected original values. Retention-1
   additionally compares against actual native FullKV over 128 predictions.
2. **Constructed common-storage comparison:** Prompt Top-K and an explicitly
   labeled 0.5/0.5 prompt-window/shared-prior selection control have the same
   uniform physical capacities and raw append-only history. Their initial,
   final, and decode-owned backing peaks must match exactly. No dummy padding
   or enlarged baseline workspace is allowed. Neither control is the frozen
   graph method or the six-component design.
3. **Same quantized-source reader contrast:** retain the original frozen packet
   and protected exact rows, but read the quantized inputs with native full-width
   K/V projections and original RoPE order. Compare against the fused reader,
   and independently against the original FullKV reference. Unused fused patches
   remain charged; additional native-reader workspace is reported separately.

Passing same-selected native transport is **not** passing FullKV quality after
eviction. Native projection of three-bit input is **not** restoring FullKV input.
The registered 32-prediction guard and later-prefix checks are not relaxed.

## Storage boundaries

45.5 and 49 MiB are initial prompt-storage caps, not lifetime caps. Over 128
predictions, 127 K/V updates occur. Common raw history grows in 16-slot increments
and allocates 128 slots (64 MiB on this model); all allocation backing, including
growth overlap and temporary layer leases, is counted. Valid history payload is
63.5 MiB. Final storage is not represented as staying under the initial cap.

GPU attention-mask and cache-position buffers are enumerated separately from the
existing cache ledger. Original prefill and capture owners held for verification
are explicitly disclosed; they are not online persistent representations. The
existing ledger is decode-owned cache backing, not exact whole-request KV peak.
No whole-request certification or superiority is inferred from these controls.

## Finite verification

`submit_fox_fidelity_byte_v20.sh --root /scratch/djy8hg/workdir/fox-review-v20/RUN`
registers one clean source SHA and runs focused tests, the full CPU suite, two
one-A100 workers, then automatic `afterany` CPU analysis. Each GPU worker uses
12 CPUs and a 30-minute hard limit. CPU or preparation failure blocks GPU
admission and still receives failure analysis. All runs are local-cache-only;
no new samples, benchmark answers, or reference annotations are accessed.

The resulting engineering gates and FullKV predictive gates remain separate.
A negative result authorizes diagnosis, not automatic calibration or a larger
campaign. Whole-request certification still requires complete time-aligned
production owner publications rather than synthetic allocation witnesses.
