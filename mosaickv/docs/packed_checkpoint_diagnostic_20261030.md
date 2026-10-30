# Packed-checkpoint engineering diagnostic

This is a separately versioned, consumed-input, lossless HF diagnostic. It
does not change selection, K/V bits, mandatory tokens, logical positions,
quality thresholds or the normal runtime defaults. The previous completed
engineering evidence and negative quality results remain sealed.

## Hypothesis

Direct packed-source projection currently decodes the preceding 64 exponent
codes to locate literal exponents in the second half of every 128-value group.
Store that half's escape count in the existing sixteen-byte checkpoint instead.
This removes redundant reads and reductions without adding persistent bytes.

For group `g` in a 1,024-value block, the checkpoint stores its preceding
literal count in ten low bits and its first-half count modulo 64 in six high
bits. Group zero always has preceding count zero: its low bits therefore hold
eight overflow flags, one per group, distinguishing first-half counts 0 and 64.
The representation is lossless for every possible half-count vector.

## Frozen execution and acceptance

- Same two previously consumed Description/MM-Vet inputs and frozen Top-K /
  EviCastKV memberships; no references or benchmark labels enter construction.
- Pinned LLaVA-1.5 BF16 full-MHA, HF eager, batch one, serialized requests.
- First verify every BF16 pattern, raw source and all-literal overflow cases.
- Require retention-one parity and original selected K/V/logit bit equality.
- Require identical initial, final and observed decode-owned peak bytes to the
  existing packed-source projection, including metadata and correction bits.
- Run five warmups and twenty randomized paired timing blocks, with the same
  32-token schedule. The existing encoded-KV reference is retained separately.
- A strict no-slower engineering gate requires each paired 95% upper bound of
  the new/old packed-source decode ratio to be at most 1.0. Failure is preserved;
  it does not trigger parameter changes, retries on new inputs or a larger run.
- At most two one-A100 workers concurrently, twelve CPUs each and thirty-minute
  limits. A finite `afterany` CPU analyzer handles every producer outcome.

Only a passing diagnostic could justify applying this optional constructor to
the same six already tested baseline paths. No fresh, calibration or paper-scale
campaign is authorized or implied. Whole-request HBM peak remains uncertified.
