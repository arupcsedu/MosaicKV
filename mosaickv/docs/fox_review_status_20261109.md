# Prof. Fox review: execution coverage and bounded plan

This is a November 9, 2026 evidence inventory, not a favorable-result certificate.
The companion [JSON matrix](fox_review_status_20261109.json) separates
implementation, measured evidence and scientific conclusion for R01–R10 in the
supplied `change_request.md`. It binds actual source reviews and immutable gate
files by SHA-256. Current source base: `df4251415c6fe96fdf519b534aebd7ec74a93562`.

EviCastKV is the paper name for MosaicKV. Three scopes must stay distinct:

- **Retention:** prompt-conditioned exact selection, original retained K/V and
  logical positions, append-only decoding. The current COCO study is FP16 eager,
  20/40 MiB, 64 predictions and 63 continuation forwards.
- **Source:** quantized shared-source storage and reconstruction, with its own
  historical model, precision and generation settings. The evaluated source
  configuration does not execute the retention configuration's graph selector.
- **Six-component hybrid:** a separate research implementation. Exact-only and
  storage evidence do not establish prototype/residual/repair benefit.

## What actually completed

| Control | Inspectable evidence | Interpretation |
|---|---|---|
| CPU suite | `fox-review-v4-cpu-suite-fullkv-recovery-20261109/GATE_cpu_suite.json`: 5,151 tests, zero failures/skips | Implementation checks passed; no task-quality inference follows. |
| Original V3 correctness | 248/256 completed; eight retained numerical failures | Still negative. Neither the COCO continuation nor changed common transport retroactively passes it. |
| Native COCO subset transport | 128 checks on all 32 validation inputs, hash-bound reuse | Complete passing workload reused; no favorable surviving intersection. |
| COCO scoring | 224/224 requests; 32 unique images | Current development comparison complete; not confirmation. |
| COCO profiling | 56/56 requests; eight inputs | Diagnostic ranges only, excluded from headline latency. |
| COCO unprofiled timing | 336/336 timed requests; 16 inputs, five warmups, three repeats | New controlled FullKV/Top-K/local-port/EviCastKV timings exist. |
| MileBench valid-slot common transport | 8 + 120 checks on 2 + 30 inputs | All 32 consumed inputs pass unchanged numerical bounds under the separately labeled transport. |
| MileBench compressed scoring | 128/128 rows, no failures | Retrospective CPU scoring; Accuracy and ROUGE-L remain separate. |
| MileBench FullKV reference | 32 complete GPU traces recovered on CPU | Valid reference recovery, not new inference or fresh evidence; original postcheck failures are retained. |
| Mechanism/routing study | Array **22742447**, analyzer **22742448**, 32 shards, throttle six | In progress at this assessment. Submission is not a result. |

All run roots in the table are under `/scratch/djy8hg/runs/evicastkv/`.
The JSON matrix supplies full paths and hashes; it is the precise locator.

### Current scientific interpretation

The COCO scoring study does not establish superiority: EviCastKV and Top-K tie at
20 MiB; the 40 MiB EviCastKV-minus-Top-K point estimate is positive but its
adjusted interval has a lower bound of exactly zero. That is not strictly
positive superiority or population equivalence. The local VL-Cache contrast
also remains inconclusive. Official baseline fidelity is not established by
shared-byte execution.

Boundary-B diagnostics completed, but **all 32 FullKV chunked-prefill numerical
controls failed**. Preserve those diagnostics as negative. They are neither a
validated short-answer sensitivity result nor a byte-matched A/B quality table.
The valid boundary-A scoring gate does not answer this different question.

The measurement contract also states that source/plan objects remain resident
through output construction. Published active-cache bytes include their declared
backing storage, but do **not** certify total live KV or a whole-request KV-only
peak. Request allocator peaks and sampled process telemetry are different metrics.

### Freshness: classifications resolved, media certification still incomplete

The three roots listed as unresolved in the older ID inventory are **already
resolved by later immutable evidence**. Repeating that list as a current blocker
would be incorrect. The matrix binds these verified results:

- `coco-cublas-retry-edb6edf-20261014` is a retry of the already-excluded COCO
  pool. Its original launch seal binds its plan and ledger; all eleven bound
  files were reverified without reading answers.
- `milebench-actionprediction-extract-20261018` has a saved original Slurm
  submission line for tar extraction only. The earlier resolution conservatively
  quarantined the complete ActionPrediction pool; that conservative exclusion
  can remain in force rather than inferring fresh IDs from scheduler-log silence.
- `fox-review-20261028-milebench-media-content-v1` is a content-bound input-media
  audit, not model prediction evidence. Its original report and media bindings
  were reverified.

`fox-review-20261028-writing-closure-v2/coco_availability/report.json` already
reports zero unresolved workload roots. It hash-binds 40,504 validation image
identities, conservatively excludes 4,096 and records **36,408 ID-only available
images**. This is real availability evidence, not a media-fresh cohort certificate.
Its `exhaustive_historical_media_inventory_complete` remains false and confirmation
remains unauthorized. The separate root-resolution artifact likewise does not
certify global episode/document/media identity.

The relevant code explains the distinction:

- `evicastkv_evidence.exclusion_inventory` seeds a finite set of top-level
  manifest filenames and follows selected ledger links. It explicitly reports
  `complete_scan=false`, even when its queue is exhausted. The saved two-root
  refresh covers 314 canonical files with no errors; that is not exhaustive
  input/result/media discovery.
- `evicastkv_review_closure_v2.audit` verifies fourteen pinned Arrow shards and
  projects only `question_id`, `file_name` and `id`. It correctly treats the
  auxiliary row index as different from the canonical COCO image identity.
- `identity_exclusion_scan` provides recursive JSON/JSONL/Parquet identity
  projection and strict symlink handling. It excludes media fields, however,
  and its existing launchers are bound to other reviewed dataset contracts.
  They cannot silently become a COCO media-freshness certificate.

A bounded CPU-only follow-up can first freeze the authoritative project roots
and allowed canonical targets; inventory all canonical input/result sources,
including generic-name roots and externally referenced input ledgers; bind the
existing three provenance resolutions; and implement a COCO-specific fail-closed
ID/media projection with synthetic tests. A later separately registered real
verification must establish complete enumeration, stable hashes and required-source
coverage, then reconcile underlying image identity across non-COCO routes.
Canonical public COCO source names can establish source-ID overlap; encoded/pixel
hashes or certified aliases are still needed where a different route renamed or
embedded the image. A hash of a complete Arrow file does not by itself supply
that per-image cross-dataset comparison.

No answers or outcome information is needed for this work. No new scan,
confirmation identity selection, media materialization or submission was performed
for this inventory. With the existing certificate alone, at least 256 genuinely
media-fresh inputs are **not yet proven**, despite ample ID-only availability.

## Review requirement map

| ID | Implemented/measured now | Unresolved claim or execution requirement |
|---|---|---|
| R01: configuration/activity | Recovered coefficients, routing namespaces and frozen exact lock | Actual mechanism activity/counts await analysis; never relabel source as graph execution. |
| R02: numerical correctness | Independent compact/masked references; retention-one and new valid-slot checks | Original failures remain negative; boundary-B first divergent operator needs diagnosis. |
| R03: baselines/scoring | Native scorer fixtures, COCO and separate MileBench metrics, recovered FullKV reference | Caption/EOS/parser audit needs a complete export; LOOK-M/VL-Cache official-path fidelity remains unverified. |
| R04: timing/construction | New full-request COCO timing and bounded profiler evidence | No semantics-preserving speedup measured yet; MileBench transport needs its own repeated timing before latency claims. |
| R05: mechanisms | Registered routing, fixed-candidate/fixed-quota terms, cross-modal removal, seeded controls | Current array not analyzed; window sensitivity and finer independent term attribution are not in its matrix. |
| R06: replay | Source replay implementations exist separately | New common-execution OFF/ON source evidence is not supplied by the exact study. |
| R07: source controls | Shared-source research implementation exists | Direct 3-bit projected K/V, protection controls, audited KIVI-family support and common-model bridge remain separate work. |
| R08: vision/freshness | Consumed manifests, three resolved historical roots and 36,408 ID-only available COCO images | Complete historical media coverage/frozen candidate missing; original MileBench fresh pool unavailable. |
| R09: lifetimes/geometry | Active backing storage and request allocator telemetry | Time-aligned total KV ownership/workspace classification, source-length scaling and validated GQA extension remain incomplete. |
| R10: reproducible paper evidence | Hash-bound development packet, corpus reducers and paired intervals | Final controlled-mechanism exports, qualitative selection, consistency audit and independent confirmation remain open. |

The current mechanism matrix covers routing fractions 0, 1/256, 1/16 and 1/4;
a separately bounded full-routing pilot; fixed-candidate/fixed-quota complete,
prompt-only, coverage-off, value/novelty-off and cross-modal-off variants; and
quota-matched random seeds 13/29/47. Every seed remains separately labeled.
Coverage-off is not graph-off. Jointly disabling value and novelty does not
isolate either term; jointly disabling graph and modality coverage does not
isolate either coverage source. The recovered pooled-attention entropy router
is not the proposed mean per-query entropy comparator. Windows 8/16/32/64 are
not registered arms in this current array and require a separate pre-run lock.

## Bounded authorized sequence

1. **Use saved outputs first.** Complete a CPU-only caption/parser audit: scored
   token/word mean/median/p95, EOS/limit-hit/empty/repetition/reference counts,
   option counts and parser inclusion rules. Recompute native metrics, not
   generic substitutes. Report associations, not a causal grounding explanation.
2. **Finish the registered mechanism array.** Its finite `afterany` analysis
   must retain failures and verify actual routed counts, candidate/quota equality,
   edge activity, selection overlap and costs. Do not select the best random
   seed, pool repeats as extra images, or tune from its answers.
3. **Use the bounded full-routing pilot only if its prerequisites pass.** Do
   not silently expand it to the whole workload or relax resource limits.
4. **Optimize measured bottlenecks only.** Existing profiler ranges identify
   graph construction and exact construction as tractable costs. Limit the next
   implementation pass to at most two dominant operations. Check identical
   selected entries, ties, positions, capacities, storage and numerical outputs;
   then run the registered reference/optimized Top-K/EviCastKV paired timing.
   Changed selections are a new algorithm configuration, not a drop-in speedup.
5. **Keep missing source and sensitivity studies separate.** Recover existing
   evidence before adding GPU work. Any direct-quantization, source/replay,
   window or model/dtype bridge needs its own locked controls and correctness.
6. **Do not submit fresh confirmation yet.** A justified frozen configuration,
   reviewed claim/test family, baseline-fidelity qualification and complete COCO
   ID/media freshness certificate are prerequisites. There is no frozen candidate
   or superiority promise in this inventory. MileBench remains development-only
   until its source-group limitations are resolved.

The user-authorized ceiling is **six A100s in aggregate**, not six per array.
Workers use one GPU, 12 CPUs and a two-hour limit. Finite callbacks analyze
successes and failures without continuous log polling. A scientific negative
result resolves a review question; it is not permission to weaken its gate.

### Implemented follow-up executor (execution is separately recorded)

`mosaickv/scripts/evicastkv_review_followups.py` is dry-run by default. Its
explicit `register --execute` command binds the clean source, four passing
parent gates, original predictions, mechanism submission and diagnostic code.
It submits the complete CPU suite first, then these CPU-only derived audits:

- COCO caption/EOS, scored-length, empty/repetition and complete reference-count
  diagnostics. Pinned corpus CIDEr is recomputed; descriptive per-image
  contributions are not substituted for corpus bootstrap replicates.
- A corrected MileBench quality-statistics export without latency/speedup
  labels. Original estimates, intervals and artifacts remain unchanged.
- An observation-only audit of the actual MileBench parser, including option
  counts and unparseable answers without removing them from denominators.

The finite `afterany` chain then waits for the registered mechanism analysis
and full-routing pilot to finish. The full-routing outcome is preserved, not
inherited as a passing control for the original 1/256 policy. With the passing
common-runtime and mechanism controls, one worker profiles graph CPU functions
on the first two already consumed profiler inputs at 40 MiB. Its selected
outputs, controls and physical bytes must match the original references.
Calling-thread CPU time and wall time are separate; nested times are not added.

Only a passing graph diagnostic admits the separately preregistered window
study: all 64 consumed COCO development inputs, FullKV once per input, Top-K
and EviCastKV at windows 8/16/32/64, 40 MiB and fixed EviCastKV routing 1/16.
The **planned** matrix is 576 requests in sixteen four-input shards, not an
already measured result. Actual forecast-window provenance, original source
length, complete rows and identical realized initial/final/peak bytes are
checked. Seven exploratory corpus-CIDEr contrasts use 10,000 paired cluster
replicates. This study neither freezes a winner nor supplies headline latency.

An independent, bounded FullKV boundary-B diagnostic follows when registered:
two consumed inputs, final-position operator vectors only, original FP16 eager
arithmetic and unchanged `1e-4` logit control. Its capture-completeness verdict
is separate from numerical success. The earlier failures remain negative.
An explicit returned-state handoff fixes continuation portability; the current
in-place HF state convention had masked that omission, so it does not explain
or cure the earlier first-logit failures.

Every model worker receives afterany analysis, including failed producers.
Single-attempt requests and exclusive outputs prevent accidental duplicate
submission. The final packet records completed and blocked stages without
authorizing new cohorts, optimization, configuration freezing or confirmation.
The actual submitted job IDs and source/gate hashes belong in the generated
registration and job ledgers, not in an assertion that implementation implies
execution.

## Manuscript consequences

Use distinct `EviCastKV (retention)` and `EviCastKV (source)` labels wherever
their controls differ. Do not claim a universal lower-memory/equal-quality or
latency win from equal-capacity development points. Include FullKV as an
uncompressed quality/timing reference, not a matched compressed-budget baseline.
Keep Accuracy, ROUGE-L, CIDEr and PPL in distinct columns/families. Preserve
negative ablations, numerical failures and uncertainty in the appendix. Correct
incomplete captions, dangling cohorts, ambiguous names/citations and inaccurate
AI-assistance disclosures without implying unrecorded author approval.
