# Choice C preregistered gate

This is a bounded exploratory gate, not a paper-result campaign. It validates
the selected design sequence before any full sweep: prompt anchors, label-free
FullKV fidelity calibration, per-head prompt/draft routing, fine structural
evidence reservations, audited RoPE-neutral prototype merging, and
token-decision-aware repair.

Revision 5 is frozen in
`configs/experiments/choice_c_corrective_gate.yaml`. It uses the pinned
LLaVA-1.5-7B revision and DocVQA revision already recorded by the experiment
catalog. Twenty design-calibration IDs, ten internal-validation IDs, and ten
held-out IDs are mutually disjoint and exclude every preexisting DocVQA ledger.
Generation is greedy and fixed at 16 tokens. No task label or task score enters
calibration.

Three Choice C candidates are selected only by paired FullKV fidelity:
`1-token agreement + log1p(logit KL) + attention-output error + forecast
regret`. Selection adds a 90th-percentile loss term and requires at least 60%
bootstrap selection stability. The chosen profile then calibrates per-head
exact/smaller-cluster routing, query-conditioned prototype weights, and
fail-closed K/V dispersion ceilings. FullKV counterfactual token decisions are
used to calibrate repair thresholds and a 10% trigger-rate budget; task labels
remain unavailable. The profile must beat an exact-only MosaicKV control on a
separate internal-validation ledger before it is evaluated once against Prompt
Top-K, `lookm_reimpl`, and `vl_cache_reimpl`. FullKV is included as the
reference. Published-method reimplementations retain their paper token
granularity; MosaicKV and Prompt Top-K share block size 16. The primary budget
is measured active KV bytes, never nominal token retention.

The full sweep is allowed only when all rows complete, MosaicKV's measured
active bytes are within 2% of the nearest comparator, the paired task-score
bootstrap lower bound is no worse than the registered 0.05 noninferiority
margin, and its held-out FullKV-fidelity error is at least 30% lower than the
nearest comparator. Prototype merging must beat exact-only selection on design
and internal-validation data, repair must satisfy its registered precision
gate, and every MosaicKV row must exercise the genuine three-tier path. With
ten held-out examples, passing is only an engineering gate and cannot support
a statistical paper claim. Failure stops the full sweep and is reported without
post-hoc retuning.

The Slurm entry point is `slurm/choice_c_corrective_gate.sbatch`. It allocates
three GPUs and runs the three independent candidate profiles concurrently with
one model replica per GPU. Only after all three complete does one coordinator
resume their immutable artifacts and perform profile freezing, repair
calibration, internal validation, and held-out evaluation. This barrier keeps
the evidence partitions sequential even though independent design candidates
run in parallel. The revision-5 specification is explicitly classified as a
dirty-tree exploratory gate; a later paper-eligible execution must use the same
frozen specification from a clean committed worktree.

## Revision-5 stability outcome and revision 6

Revision 5 completed all 60 candidate-calibration rows in Slurm job
`17977239`; its coordinator did not start internal validation or held-out
evaluation. All candidates had the identical mean active-byte cost.
`balanced_23` minimized the registered robust fidelity objective, but appeared
as the bootstrap winner in only 50.4% of 1,000 resamples, below the frozen 60%
selection-stability threshold. The correct outcome is therefore a failed
calibration gate, not a frozen profile. The threshold was not relaxed and no
downstream outcome was inspected.

Revision 6 changes only design-calibration sample size from 20 to 40. It keeps
the three candidates, FullKV-fidelity objective, bootstrap seed and resample
count, 60% stability threshold, internal-validation ledger, held-out ledger,
comparators, budgets, and final gates unchanged. The 20 added IDs were selected
from the pinned DocVQA split after excluding every existing ledger; no labels or
scores were used. This is a sequential sample-size correction for a failed
stability criterion, not outcome-driven threshold tuning. The frozen revision-6
specification is `configs/experiments/choice_c_corrective_gate_v6.yaml`.

Revision 6 uses two dependent Slurm allocations. The three-GPU candidate job
(`slurm/choice_c_calibration_candidates.sbatch`) releases its GPUs after the
parallel design stage. A one-GPU coordinator
(`slurm/choice_c_coordinator.sbatch`) runs only after successful candidate
completion and resumes the immutable candidate artifacts before profile
freezing. This avoids reserving idle GPUs during the sequential stages.

Revision 6 also exposed and corrected a calibration implementation bug: byte
eligibility was properly established on the complete calibration ledger, but
the bootstrap loop reapplied the same ceiling to each resampled mean. Since all
three candidates have identical per-sample active bytes, many resamples rejected
every candidate simultaneously and made selection frequencies sum below one.
Bootstrap resampling now measures selector stability only among candidates
already eligible on the full ledger. A regression test covers the single-
candidate, sample-dependent-byte case.

After that correction, no v6 candidate met the unchanged 60% threshold; the
largest selection frequency was 52.2%. `balanced_23` and
`prompt_anchored_15` had essentially identical mean calibration loss (paired
mean difference 0.00075), so simply adding examples was not expected to produce
a meaningful unique winner. Revision 7 therefore freezes one
`consensus_median` profile using the coordinate-wise median of the three v6
parameter sets. The rule uses neither task labels nor observed candidate rank:
utility weights become -0.30/-0.30, prompt/evidence reservations 0.15/0.15,
head-shift threshold 0.15, and prototype geometry 15 blocks/255 positions.
All downstream gates remain unchanged. The frozen specification is
`configs/experiments/choice_c_corrective_gate_v7.yaml`.

The first revision-7 coordinator attempt (`17979771`) stopped before adaptive
routing because all exact-control rows rejected a prompt reservation containing
an optional partial modality fragment. The mask-free exact selector already
required full-size optional blocks, but prompt/evidence reservations were made
before that eligibility filter. The correction applies the identical block-size
eligibility rule while ranking reservations in exact mode. The failed exact
artifact is retained under its original run ID; corrected exact controls use a
new `exact-control-v2` identity. Consensus calibration rows are unchanged and
resumed immutably.

The corrected revision-7 coordinator subsequently exposed a different
capacity mismatch during adaptive routing. Its nine full-source
`exact_head_routes` cannot fit inside a 20% dense HF cache, so every completed
adaptive sample correctly fell back to selection-only execution. Job
`17983293` was stopped before internal validation or held-out access, and its
partial artifacts remain immutable. Revision 8 freezes
`exact_head_fraction: 0` and routes sensitive heads only through smaller,
shorter-span prototype groups. This feasibility correction was made from tier
mode and capacity diagnostics, not from task labels, task scores, internal
validation, or held-out outcomes. Revision 8 also runs its base-prototype and
exact-control calibration rows concurrently on two GPUs; model state, sample
ledger, and deterministic inference settings remain identical.

Revision 8 completed all 40 base, exact-control, adaptive-head, and oracle
repair calibration rows. Adaptive routing was genuine three-tier on all rows
and passed the prototype-versus-exact calibration control. Repair calibration
then found beneficial counterfactual events, but no normalized-risk/decision-
margin rule met both the frozen 10% trigger ceiling and 50% precision floor;
the fail-closed profile disabled online repair. The first coordinator had
already started one internal-validation sample before this missing early-exit
guard was identified. It was cancelled before an internal result row completed
and before any held-out access. The corrected coordinator writes a negative
calibration gate and exits before internal validation whenever prototype or
repair calibration fails. Corrected resume job `17991276` exercised that early
exit successfully in 22 seconds without producing an internal or held-out run.

## Revision-8 calibration outcome

Two-GPU control job `17985856` completed 40 base three-tier rows and 40
exact-selection rows in 45:48 wall time; the individual workers took 45:46 and
24:26. Coordinator `17985857` then completed 40 adaptive-head rows and 40
oracle-repair rows. Every base and adaptive row used genuine `three_tier`
construction. Adaptive routing reduced the registered mean FullKV-fidelity
error from 2.375571 to 2.329087 at identical measured active bytes and was
therefore frozen; the exact-selection control error was 5.198446. Nine
sensitive heads used smaller prototype groups, and no full-source exact-head
route was requested.

The oracle run recorded 102 triggered counterfactual repairs, including 29
token changes and 15 FullKV-token recoveries. Nevertheless, no online rule over
normalized prototype risk and decision margin satisfied both the frozen 10%
trigger-rate ceiling and 50% precision floor. The resulting repair profile is
disabled with zero predicted online triggers. Because decision-aware repair is
a required Choice-C stage, the calibration gate is negative and the full sweep
is forbidden. No held-out sample was accessed. One prematurely started
internal-validation trace was archived and contributes zero rows to any gate.
These dirty-tree runs remain exploratory engineering evidence, not paper
results.

Revision 4 changes no candidate, metric, comparator, or gate threshold.
Revision 3 completed and its numerical gate reported a 40.74% fidelity-error
improvement over byte-matched Prompt Top-K, but the post-run tier audit found
that all five requested `mosaickv_full` rows had the effective method
`mosaickv_full__exact_only_incompatible_assignments`. Prompt reservations had
occupied per-head dense capacity before the prototype anchor cover was added,
so no prototype, residual, or repair path was exercised. Revision 3 is retained
as an exact-selection diagnostic and is not evidence for complete MosaicKV.
Revision 4 makes the deterministic prototype cover an occupied part of the same
budget transaction before bounded prompt and evidence reservations. This is a
representation-feasibility correction, not outcome-driven hyperparameter
tuning. Revision-4 acceptance additionally requires the artifact audit to
observe genuine three-tier `mosaickv_full` rows; the numeric gate alone is not
sufficient.

Because revision 3 exposed the original held-out outcomes before this
implementation correction, revision 4 does not reuse those five examples. Its
immutable evaluation ledger uses seed 20270803 and takes the first five DocVQA
IDs after excluding every prior selector and generalization DocVQA ledger. The
new IDs were frozen without reading their labels or scores. Calibration still
uses the original five calibration-only IDs, so candidate selection remains
disjoint from both held-out revisions.

Revision 3 changes no candidate, data, metric, or gate threshold. Revision 2
stopped during calibration after dense LLaVA capacity checks rejected three
prompt-anchor reservations. Revision 3 adds the missing per-layer/KV-head
capacity constraint during reservation and lets already reserved prompt or
evidence blocks satisfy the prototype anchor cover. Revision-2 partial rows
remain under the versioned scratch output and are not calibration inputs.

Revision 1 was cancelled before any result row completed because its
`group_size=2`/16-position geometry could not represent a 0.2 active cache:
one exact anchor plus one prototype per three source blocks has a 2/3 minimum
active ratio. Its uninspected partial traces are archived under
`/scratch/djy8hg/runs/mosaickv/archives/choice-c-invalid-prototype-geometry-job17971234`.
Revision 2 adds a fail-fast analytical feasibility check and uses group sizes
15 or 31 with spans that can cover the corresponding contiguous 16-token
blocks. This correction was made from representation geometry, before viewing
any sample outcome.

## Revision-4 outcome

Slurm job `17974152` completed 20 calibration rows and 25 held-out rows in 57
minutes. Calibration froze `draft_sensitive` with profile SHA
`a6325aabe62761e4a35e7d0c4ead1b0a446395b3b61c09450a6c8e15cf047d80`.
All five held-out MosaicKV rows used the genuine `mosaickv_full` effective
method and retained nonzero lossless residual storage; no prototype fallback
occurred. All primary measurements were finite and every method evaluated the
same five unseen IDs.

The scientific gate failed. At identical measured active KV bytes, MosaicKV's
registered FullKV-fidelity error was 61.98% worse than Prompt Top-K rather than
30% better. Mean DocVQA score was 0.4000, below LOOK-M's 0.5114 and equal to
Prompt Top-K and VL-Cache on this five-example sample. Decision-aware repair
triggered zero times, so it recovered no held-out error. The full sweep is
therefore not permitted. These are exploratory engineering measurements, not
paper results; the complete artifacts remain at
`/scratch/djy8hg/runs/mosaickv/choice-c-preregistered-gate-v4`.
