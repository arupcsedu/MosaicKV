# CPU-only key/value causal decomposition

Keep the failed two-step affine source fit negative. Do not change its fit,
numerical gate, codes, protection policy or identities. No GPU run is enabled.
Its improvement in input SSE did not translate into attention-output fidelity.

The projected old/refitted key and value tensors were not saved. Recompute only
this CPU numerical diagnostic, from the same immutable consumed captures and
runtime-BF16 checkpoint weights. Reuse the frozen two-step quantizer. In addition
to old/old and refitted/refitted, evaluate refitted-key/old-value and
old-key/refitted-value counterfactuals. Original-FullKV-key or -value mixtures
are evaluation-only oracles. Retain full causal masks and exact protected rows.

Report key and value SSE, per-query/head attention-output MSE and p95 separately.
Do not combine metrics or describe any mixed tensor pair as a realizable
same-byte representation: some controls require both source encodings or FullKV
information. They cannot support a systems or quality superiority claim.

This isolates whether key-induced attention changes, value changes or their
interaction explain the failed fit. No new fit, task-answer access, benchmark
cohort, model-level inference or GPU experiment is authorized by this protocol.
