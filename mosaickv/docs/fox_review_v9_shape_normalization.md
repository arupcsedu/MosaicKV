# V9: bounded boundary-normalization correction and evidence audit

This is a consumed-input correctness diagnostic, not a new quality cohort,
production-default change, or superiority experiment. EviCastKV is the paper
name of MosaicKV; this experiment does not create a separate algorithm.

## Evidence motivating the correction

V8 reproduced the saved FP16 Mistral RMSNorm discrepancy using the unchanged
installed native forward, original weights, and epsilon. For the first case,
the full-prompt row shape matched saved path A bitwise, while the singleton
shape matched saved path B bitwise. A singleton view into the same allocation
also matched B. This localizes a shape-dependent native reduction difference;
it does not establish full-model parity or repair historical negative results.

### Completed parent jobs and measured outcomes

These are V8 development diagnostics, not independent benchmark evidence.
Slurm accounting reports all 16 jobs completed with exit code `0:0`.

| Group | Jobs | Measured outcome |
| --- | --- | --- |
| Complete CPU suite | `22768710` | 5,770 passed; 3 min 25 s |
| Preparation | `22768711` | Completed in 15 s |
| Native RMSNorm replay and analysis | `22768825`, `22768826` | Both saved shape pairs reproduced; GPU worker 14 s; full-model parity not established |
| Source protection and analysis | `22768860`, `22768861` | Base source packets match across policies; protected rows exact; attention-protection continuation 31/32, so fidelity failed; GPU worker 1 min 32 s |
| Eight bounded inventory workers | Array `22768823`, indices 0–7 | All completed, each 9 min 9–12 s; none finished traversal |
| Inventory aggregation | `22768824` | 129,652 processed files and 946,666 projected records; 18,128 unresolved errors and 29,933 frontier items |
| Final packet | `22768909` | Complete diagnostic artifacts; no freeze or confirmation authorization |

In the first norm case, six of 4,096 FP16 elements differed between native
full-row and singleton execution, with maximum absolute difference `0.00390625`.
This operator-level result motivates V9; it must not be called a full-model
correctness pass. The source-protection token failure and incomplete inventory
remain blockers regardless of V9's numerical outcome.

## Registered numerical experiment

- Reuse exactly the two consumed boundary inputs and the pinned model/config
  from V4; no new examples, benchmark answers, or reference labels are used.
- Keep original model weights, FP16 precision, RMSNorm implementation, epsilon,
  eager attention, logical positions, and absolute logit tolerance `1e-4`.
- Path A performs the ordinary complete prefill and ordinary continuation.
- Path B first uses the existing masked-dummy, full-row prefix transport,
  removes exactly the dummy cache position, and evaluates the withheld final
  prompt token once. Only that boundary forward additionally pads native
  language RMSNorm inputs to the original prompt row shape. No output tensor
  or logit is copied from A to B or numerically patched.
- Restore every temporary Linear, attention, and RMSNorm override immediately
  after that boundary forward. Evaluate the next 15 predictions using ordinary
  decoding on both private caches, forcing the same reference prefix to avoid
  conflating numerical divergence with different inputs.
- Require 16/16 greedy agreement, maximum absolute logit difference at most
  `1e-4`, bitwise-equal initial/final caches, and complete scope counters. Each
  of the `2L+1` language norms must execute exactly once in the boundary phase;
  no padded operator may leak into continuation.
- Exceptions restore original methods. Unsupported model/norm layouts fail
  explicitly. This expensive transport remains evaluation-only and is not
  eligible for online-support, quality, or latency claims.

Any failed execution or numerical comparison remains a negative artifact.
Historical native and shape-only failures remain unchanged even if V9 passes.

## Saved-metadata and oversized-record checks

The CPU audit verifies the eight saved scan checkpoint chains, registration
bindings, cursors, unresolved errors, and traversal frontier. It does not walk
the repository or interpret an incomplete scan as freshness certification.

A separate bounded recovery canary selects the first four lexicographically
ordered oversized-JSONL error locations before examining their contents.
It applies the existing identity-only lexical projection and collector, never
raw JSON decoding of predictions or labels. Canonical source files must remain
inside their registered roots and match the saved signatures/segment hashes.
Limits are 128 MiB per source file, 180 seconds per record, and 600 seconds
overall. Errors, oversized inputs, and timeouts are retained; no replacement
record is chosen after a failure. Neither the original inventory nor any
checkpoint is overwritten. This canary cannot certify the remaining inventory
or authorize catalog access, materialization, or fresh evaluation.

## Finite execution and immutable evidence

The controller first requires the complete CPU test suite from one clean source
SHA. It then performs the bounded saved-evidence checks, submits one A100 worker,
and submits an `afterany` analyzer so negative/failed workers are still reported.
The aggregate ceiling is six A100s; this experiment needs only one, with 12 CPUs
and a two-hour maximum. CPU stages use the `bii` partition. GPU stages use
`bii-gpu`, account `bii_dsc_community`, reservation `bi_fox_dgx`.

All caches, temporary files, artifacts, and logs remain under scratch. The
common environment is used without installation or upgrades. CUBLAS
determinism settings are set before Python. Saved KL comparisons use the
registered AVX2 CPU capability and 12 threads, not a relaxed tolerance.

Registration, scheduler requests, results, hashes, failures, and review status
are write-once. Lost submission acknowledgements do not trigger blind retries.
The final review packet distinguishes implementation completion, measurement
validity, and scientific conclusions. It also records unresolved actual-byte
matching, source guard, freshness, whole-device ownership, and disclosure
limitations. Passing this diagnostic never launches a quality campaign,
freezes a candidate, or establishes superiority.
