# V103 fresh-identity calibration protocol

V103 now authorizes one narrowly scoped local CPU identity/provenance exclusion
scan. It does not authorize catalog access, sample materialization,
disjointness checking, a CPU or GPU experiment, Slurm submission, candidate
freezing, or held-out evaluation. The scan has not run, so no V103 output
directory should exist yet.

## Why V103 is separate

V102 is sealed negative. Its two debug outputs were inspected, and the complete
32-sample ledger for each V102 task was conservatively declared ineligible for
future calibration. V103 therefore has a new experiment identity, selection
salt, planned output root, exclusion manifest, ledgers, and disjointness report.
It does not revise V102 or reuse a V102 result row.

“Fresh” means a canonical sample identity absent from every prior input,
development, debug, calibration, evaluation, and held-out artifact. It does not
mean that an entire public dataset split has never been named before. V103 plans
to use still-unseen identities from the pinned TextVQA training and
InfographicVQA validation pools only after a new identity-only scan proves their
eligibility. Until that scan is hashed and reviewed, the protocol makes no
freshness claim.

## Leakage-resistant ordering

1. Commit and review this protocol without reading dataset rows.
2. Separately authorize a CPU-only identity/provenance exclusion scan. The scan
   must bind and exclude all 64 V102 selected identities and every other prior
   MosaicKV identity it finds.
3. Hash and review the new exclusion manifest.
4. Rehash and review the two existing complete ID-only catalogs. These catalogs
   contain no labels, answers, references, scores, prompts, media, or outcomes.
5. Materialize only two preflight identities per task, then hash and review a
   preflight disjointness report proving that all four are fresh and unique.
6. Only then may a separate authorization permit the four-sample GPU preflight.
7. If and only if that preflight is sealed positive, separately authorize the
   next 32 deterministically ranked calibration identities per task. Hash and
   review a second disjointness report proving zero overlap with prior artifacts
   or preflight identities and 68 unique selected identities in total.

This separation prevents inspection of a debug failure from consuming the
calibration identities. A negative preflight requires the calibration ledger
paths to remain absent. Any failed scan, ambiguous artifact,
catalog mismatch, missing identity, overlap, or count mismatch blocks the next
step.

## Candidate and gates

The algorithm and hyperparameters remain the preregistered V100 exact-only
candidate. V103 does not tune from V102 answers, references, scores, or failure
identities. Commit `3563c66` is required only for corrected TextVQA provenance
and structured hard-budget fallback handling.

The GPU preflight must produce complete FullKV, Prompt Top-K, and MosaicKV rows
on identical inputs. A safe FullKV fallback is recorded as a structured failed
candidate—not packed into the final compressed cache and not treated as a
successful compressed row. Both tasks and every preflight identity must reach
the hard byte target before the 32-sample-per-task calibration can even be
authorized.

All paired arms must use identical weights, precision, processor, chat template,
media preprocessing, prompt, tokenization, generation settings, output length,
active-byte accounting, mandatory-token policy, block size, attention backends,
warmups, and timing protocol. Every completed or failed row must bind the full
software, model, dataset, GPU, seed, configuration, and clean-worktree
provenance required by `AGENTS.md`.

The scientific gates remain unchanged: at least 30% lower active and total KV
memory than Prompt Top-K, no more than five percentage points of paired answer
or token fidelity loss, every task and every robust fold passing, zero residual
storage, and complete provenance. Latency remains informational.

## Protocol-commit and scan-scope seal update

The protocol was committed and reviewed at
`75736bbc115d2ad5d82ae7c7a3a26df41c89d8fe`; its committed file hash is
`c87e76a7555aeea7c7722e5b0cd8604134f5da94fad5fa0f156cea0e4ce26e9f`.
The subsequent scan-scope seal was committed at
`07958152271b8a8d72b51679cbb92cecf20cc005` and reviewed. The present update
authorizes only the exact local CPU scan defined by that seal: its fixed roots,
mandatory V102 inputs, identity/provenance fields, fail-closed behavior, and
single write-once manifest path. It continues to forbid catalog access, sample
selection, dataset-row loading, materialization, disjointness checking, model
loading, and CPU, GPU, or Slurm experiments.

## Current state

The protocol and scan-scope seal have been committed and reviewed. The local
CPU exclusion scan is the sole authorized operational action, but it has not
run. No IDs have been selected, no dataset rows have been loaded, and no job has
been submitted. This authorization update must first receive an immutable
commit and review. Catalog access and every later protocol stage require their
own subsequent review and explicit authorization.
