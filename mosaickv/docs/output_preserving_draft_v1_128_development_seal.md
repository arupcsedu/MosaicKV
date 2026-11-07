# Output-Preserving Draft V1: 128-Sample Development Seal

Status: **negative development result; identities consumed; not paper evidence**.

This seal prevents the 128 COCO Caption examples evaluated by the frozen
draft-veto candidate from being used to tune or validate its successor. The
successor may use only the aggregate decision below: replace veto-only ranking
with the preregistered joint normalized prompt/draft objective. It must not use
these examples' answers, references, failure identities, or per-sample metric
differences.

## Immutable inputs and outputs

- Inference plan SHA-256:
  `5e57ddaa67c5bf301af2c71af9d4b06a6e27edcd08ceada95d68fabdb606cbee`
- Analysis report SHA-256:
  `a2b0dd4b2532be95a5fae72a9fe9a43a6c20ca1d76d505d71ab636b8453374bd`
- Paired-sample Parquet SHA-256:
  `75b77c2284cff7be5eeaa3195eca9027c85e1f1b909477bcae9cedcd6a14c81c`
- Quality-comparison Parquet SHA-256:
  `9db18f9232c18118c6429be7c2e877b4c8f9700b772f207b50c5126cd1536fd7`
- Byte-gate Parquet SHA-256:
  `d1f938d28738b1300aed75e75f650ae1e938c752519361e62f3bf763d2aef47e`
- Analyzer source commit: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`

## Aggregate decision

All completeness, provenance, retention-one, exact-cap, and at-least-30%
initial/final/peak memory gates passed. The zero-margin paired CIDEr gate
against 40 MiB Prompt Top-K failed: the candidate-minus-reference estimate was
`-3.0523677586`, with a 95% paired-bootstrap interval of
`[-7.8160453426, 0.9614150922]`. Therefore no paper campaign is authorized.

No sample-level outcome is listed here. Every future preparation step must
exclude all identities bound by the inference plan above and fail closed on
any overlap.
