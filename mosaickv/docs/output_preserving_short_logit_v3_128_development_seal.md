# Output-Preserving Short-Logit V3: 128-Sample Development Seal

Status: **negative development result; identities consumed; not paper evidence**.

This seal prevents the 128 COCO Caption examples evaluated by the bounded
short-horizon logit reranker from being reused to tune or validate a successor.
A successor may use only aggregate, label-blind telemetry: the memory and
correctness gates passed; 78 of 128 requests had no passing non-label probe
plan and used the deterministic Prompt Top-K anchor; and the registered paired
CIDEr gate failed. It must not use these examples' answers, references,
failure identities, or per-sample metric differences.

## Immutable inputs and outputs

- Inference plan SHA-256:
  `1320e35e113a50f4ece8126d0f5464b53926d265eb58c36bded3e015cbf08dff`
- Derived ledger SHA-256:
  `68a9857c145acac2c21ef762352b2414c5a2ab65f883037ecbfea830820e714b`
- Analysis report SHA-256:
  `fbd58568ab42de211f66fa831dc7913795a0d1e4b8c1e41768b1f1cc2cd6a58f`
- Aggregate Parquet SHA-256:
  `13eab0c3f3b62601aeb84ef407500c79f4fcdd336df900064d41e50bedfd7d57`
- Paired-sample Parquet SHA-256:
  `2aac831e959fe7b2770ef4e9374608b5a4212f9427623a78478f54b6cbdb76f4`
- Quality-comparison Parquet SHA-256:
  `59c3c3e452e0d118f1deb0a2bc5664084380843ea5f58ee7fbf21008a8dddc4b`
- Byte-gate Parquet SHA-256:
  `aca04eb53a080afbefedd9d3708216abd579ba7d1854e3ad595870b55fa2f9f8`
- Base source commit: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Bound source-content SHA-256:
  `143d2530adee4878cbfaf8dc08d74389e70d46d60252c69455b884528f0099be`

## Aggregate decision

All execution, provenance, retention-one, exact-cap, trace-completeness, and
at-least-30% final/peak-memory gates passed. The candidate used 25.5 MiB at
prefill and 33.375 MiB at the final/peak boundary, versus 40 MiB and
47.875 MiB for the quality-eligible Prompt Top-K reference, a minimum observed
reduction of `0.3028720627`.

The zero-margin paired CIDEr gate failed. Candidate minus 40 MiB Prompt Top-K
was `-2.1143183121`, with a 95% paired-bootstrap interval of
`[-4.7933686464, 0.6940574616]`. The candidate also did not improve on the
same-budget 25.5 MiB Prompt Top-K anchor: the difference was `-0.1975839884`,
with interval `[-0.6310820905, 0.1061608178]`.

The paper campaign remains blocked. No sample-level outcome or identity is
listed here. Every future preparation step must bind this plan as an exclusion
source and fail closed on overlap.
