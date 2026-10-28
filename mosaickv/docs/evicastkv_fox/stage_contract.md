# Stage ledger and non-negotiable distinctions

This file specifies deliverables, not completed experiments. Actual status lives
in the write-once run directory's `analysis/*/GATE_*.json` and raw worker records.

| Prompt | Implementation / execution boundary |
|---|---|
| 1 | Frozen historical configurations, nested development manifests, typed rows, clean-source validator and runnable CLI. Confirmation freshness is separately blocked until exhaustive ID/media exclusion is certified. |
| 2 | Independent float64 original-layout oracle, backing-storage tests, real retention-one comparisons, original-source masked versus packed forced continuation. Numerical/structural failures block dependent runs. |
| 3 | Preserve native COCO/MileBench scoring and invalid-output rules; local baseline fidelity cannot be inferred from shared-byte execution. Boundary A remains the historical protocol; B requires a separate adapter-valid held-back-token control. |
| 4 | Complete-request wall timings and local emission transport, allocator peaks distinct from KV; five warmups/three repeats. Profiling and unprofiled timing are separate. |
| 5 | No optimization may be called measured or semantics-preserving before actual hotspots and identical-selection evidence exist. No preregistered speedup is assumed. |
| 6 | Deterministic routing sweep; same-candidate/same-quota ablations require explicit verification. Coverage-off does not mean graph-off. Cross-modal claims require active cross-modal edges. |
| 7 | Requires a human-reviewed selected configuration and 256 genuinely fresh inputs per workload. No surrogate development result can satisfy this gate. |
| 8 | Current complete runtime is B=1. Optional batching remains NOT_RUN; no new batching scheduler is introduced. |
| 9 | COCO corpus CIDEr must be recomputed inside bootstrap replicates; separate MileBench metric families and equal-subtask hierarchy. Six prespecified primary contrasts and adjusted intervals. |
| 10 | Reproduction and writing packet may report incomplete/negative findings. It must not manufacture missing confirmation tables or rewrite a failed gate into a pass. |

The supplied prompts explicitly request FullKV timing as an **uncompressed
reference**, separately from matched compressed-method latency. This is a new
versioned study scope, not a mutation of historical shared-byte contracts which
excluded FullKV from compressed-method latency tables.

Representation correctness compares a packed subset with the same selected
source keys masked in original layout. It never compares a lossy subset to
unrestricted FullKV as an identity test. Retention-one preserves the historical
1e-4 logit bound. A separate, preregistered FP16 representation bound of eight
machine epsilons (absolute and relative) is recorded independently; it does not
relax the retention-one criterion. Any failure is retained, not waived.

No prototype, residual, repair, draft, training, new backend or new serving
scheduler is implemented under this campaign. Older V3 value-correction jobs
22090651/22090652 are separate diagnostics; their failed coefficient validation
does not constitute evidence for this frozen exact-selection campaign.

## Numerical-contract revision after the first irregular-layout diagnostic

The first real COCO irregular-layout tests failed the provisional **elementwise**
8-epsilon FP16 allclose check at later forced steps (maximum absolute differences
about 0.01--0.023 in the inspected failures). Those attempts remain negative;
they are not relabeled as passes. No benchmark-score-based tuning follows them.

The replacement numerical study adds independent evidence rather than silently
raising that tolerance:

1. An independently gathered compact source cache, with independently built
   additive masks, must match production packing bitwise and its 16-step logits
   within **1e-4 absolute**, including greedy-token identity.
2. A **no-eviction** FullKV/padded-FullKV control tests floating-point layout
   sensitivity on the same checkpoint without changing its retained evidence.
3. The original-layout masked-source execution is still run for all 16 forced
   steps. Its elementwise failures are retained, and a separately registered
   normwise error `||delta logits||inf / max(1, ||reference logits||inf)` must be
   at most eight machine epsilons. The no-eviction control must independently
   satisfy that same bound. This is not the retention-one tolerance.
4. Free-running masked-source agreement is reported separately, excluding the
   untouched first-prefill token. It is not confused with packing equivalence.

The new references do not call the production packer or mask builder. They use
only original K/V, selected positions and original logical decode positions.
Correctness workers do not calculate task scores. All affected reference and
candidate executions are rerun under this versioned diagnostic contract before
dependent quality or performance work. If either independent control fails,
the gate remains blocked. Retention-one's historical **1e-4** criterion remains
unchanged.

Exact-mode routing has two historical namespaces: both
`graph.uncertain_head_fraction` and `prototypes.uncertain_head_fraction` must be
changed coherently in routing ablations, despite prototype construction being
disabled. A route-zero dispatch may use shared Top-K only after checking its
selected IDs against the actual all-stable-head selector; equivalence is not
assumed from the method names.

## Finite execution and evidence export

The current user explicitly authorized automatic progression on October 27.
Finite `afterany` callbacks analyze failures as well as successes; no log-polling
daemon or claim of automatically waking this interactive conversation is used.
The separate writing-packet callback follows recorded successor job IDs once per
completed stage. On failed gates it writes an incomplete/negative packet and stops.

After valid unprofiled measurement, the registered development matrix can run on
the **unchanged reference runtime**. This is not an optimization result: zero
runtime changes and no before/after speedup are claimed. A later optimization
still needs inspection of measured hotspots, semantic equivalence and new timing.
Development includes fixed candidate/quota/descriptor checks, raw utility and
marginal terms in Parquet, explicit removal of all inter-modality edges without
neighbor refill, quota-matched random controls, and an eight-input full-graph
pilot. Every random seed remains separately labeled.

The recovered router computes entropy of **pooled** prompt attention. It is not
silently relabeled as the prompt package's proposed mean per-query entropy.
Graph construction and actual node/edge changes must be measured before making
an inter-modal-specific claim. Secondary mechanism tests form a declared
60-contrast family, separate from the six primary quality contrasts.

SVG and PGFPlots/TikZ sources are generated from measured JSON without installing
plotting packages. Unavailable PDF compilation is not reported as a successful
LaTeX build. Archived attempts keep their own protocol/source hashes. Fresh
confirmation and optional batching remain separately gated and may be NOT_RUN.
