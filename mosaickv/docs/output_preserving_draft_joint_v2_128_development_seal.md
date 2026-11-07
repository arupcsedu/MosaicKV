# Output-Preserving Draft-Joint V2: 128-Sample Development Seal

Status: **negative development result; identities consumed; not paper evidence**.

This seal prevents the 128 COCO Caption examples evaluated by the joint
prompt/draft reconstruction candidate from being used to tune or validate a
successor. The successor may use only the aggregate conclusion that improving
the prompt/draft reconstruction surrogate did not satisfy the registered
CIDEr non-inferiority gate. It must not use these examples' answers,
references, failure identities, or per-sample metric differences.

## Immutable inputs and outputs

- Inference plan SHA-256:
  `fab4d6ba6b89197a75c748046e1d45a0cb8ebe78a0892bc960bf62d7189e6456`
- Derived ledger SHA-256:
  `dc9584c102a311311b6a810e2d0140d21418856731fb13a10ee39569e7999316`
- Analysis report SHA-256:
  `dcb251aac276360c1fe0711f9628a589138409179eeea6983d835577712de726`
- Paired-sample Parquet SHA-256:
  `a83fd69c67183b67c43058d6a723f4128ed4405e7d4fc6fb5c5c4d92f4a4ed23`
- Quality-comparison Parquet SHA-256:
  `a14dfde764657a3048c37fc29328038d7002d4781985112ed514d31ac5b66d5f`
- Byte-gate Parquet SHA-256:
  `414682cb72bcba862f13717b0f02f35ed3dcb98a9af1c29d4fa0a73e359a6423`
- Base source commit: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Bound source-content SHA-256:
  `ff17b4d8bab304a85c29730347bedc9ac8e2c0160bd221c4e52190a013325037`

## Aggregate decision

All execution, provenance, retention-one, exact-cap, and at-least-30% memory
gates passed. The zero-margin paired CIDEr gate against 40 MiB Prompt Top-K
failed: the candidate-minus-reference estimate was `-2.3740644113`, with a
95% paired-bootstrap interval of `[-5.8783903557, 0.3382444558]`. Therefore
the paper campaign remains blocked.

No sample-level outcome or identity is listed here. Every future preparation
step must bind this plan as an exclusion source and fail closed on overlap.
