# HF baseline completion protocol

This is a prospective execution specification, not a measured results table.
It does not authorize a superiority claim or rename a local adaptation as official.

## 1. Recover the four interrupted VL-Cache shards

The recovery at `/scratch/djy8hg/runs/mosaickv/hf-vlcache-recovery-20261014-v1`
uses the original clean `2c7a266d05a0dd30a4a7440355148549c14baeea` source and
original job/ledger hashes. Only output paths and scheduler IDs change.
Original array tasks 206/207 cover DocVQA shard 03 at 40/80 MiB;
208/209 cover shard 04 at 10/20 MiB. Preserve all earlier partial artifacts.
The log records reservation expiry, not a scoring or model-inference exception.

`analyze_hf_baseline_recovery.py` requires all 128 paired completions and
256 identical IDs per task, method and cap, including recovered shards. It checks
clean-source manifests, shared inputs/configuration, exact physical bytes,
finite measurements and output lengths. It writes separate summary, paired
same-byte and cross-cap quality-memory Parquet artifacts. FullKV has no reported
latency comparison. Cross-cap tests contain no direct latency ratio; a 30% smaller
claim requires a paired quality test on the curves, not isolated nominal ratios.
Pointwise exploratory intervals do not by themselves authorize a selected paper claim.

## 2. Frozen C-only PrefixKV matched validation

Model: `llava-hf/llava-1.5-7b-hf`, revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`; HF eager, FP16; 64 deterministic
tokens; seed 20271001; single-token blocks; shared first/last mandatory policy.
All compressed methods use a hard 40 MiB **initial** K/V cap and append-only
decoding. Final physical bytes must also match for every paired example.
The cap applies to prompt cache, not a constant-memory decode regime: newly
generated tokens add the same cache width to every method.

Compare FullKV (quality reference), Prompt Top-K, MosaicKV exact, the
byte-calibrated PrefixKV selection-only control, and C-only
`prefixkv_question_scored_selection_only_reimpl`. C uses fixed alpha=0.5,
no sink-neutral calibration and no extra question reservation. Its existing
40 MiB profile is copied unchanged from the development run. This is a local
adaptation, **not official or paper-native PrefixKV**. The upstream checkout,
native decode policy and existing result artifacts remain unchanged.

Use 64 examples per OCR task, divided into two 32-ID shards. Exclude all
declared allocation/decode/evidence development and calibration IDs. The
parent pool is historically evaluated: this is a clean retrospective
validation, not a claim of globally untouched data. Select by IDs only.
Freeze all profiles and ledgers before executing any validation.

Every worker checks 16-token FullKV/generate and retention-1 reinjection parity
at absolute logit tolerance 1e-4, identical tokens. Five warmups per method are
stored but excluded from analysis. Quality uses one trial per example, with
paired bootstrap intervals (10,000 resamples, seed 20271014). Per-example
latency is informational; this is not a 20-repetition systems microbenchmark.
Method order rotates deterministically across shards. Missing/failed rows,
non-finite timings, wrong variant labels, eviction, mismatched initial/final
bytes or incomplete token output prevent promotion.

Both tasks must have informative FullKV and Top-K scores, observed C-only
score at least Top-K, and a paired 95% lower bound no lower than -0.05.
This retains the five-point non-inferiority rule; it must not be described as
proof of exact population quality equality. Latency is reported, not used to
waive quality. The validation cannot authorize a 30% memory claim at equal caps.

## 3. Gated larger PrefixKV curves

Prepare a separate 256-ID/task cohort disjoint from validation and calibration.
Use shared physical caps 10/20/40/80 MiB. The control and C-only profiles at
additional caps are generated once from the **same hashed prompt-only captures**,
with fixed algorithms and alpha, before validation; no new labels or tuning.
The 40 MiB frozen profile stays byte-for-byte equivalent as a JSON object.
This makes 64 one-A100 shards (two tasks x eight 32-ID shards x four caps).
Every larger worker independently verifies the successful two-task validation
report and hashes before loading weights. A failed or missing gate blocks it.
Do not launch this stage merely because Slurm reports successful process exits.

## 4. Separate LOOK-M validation

Run `lookm_reimpl` with matched FullKV/Top-K/MosaicKV controls on the validation
IDs and 40 MiB cap. Use the existing ratio placeholders only for internal
partitioning; physical bytes govern the budget. This is a local byte-matched
reimplementation lane, not an official LOOK-M/MileBench reproduction. Larger
LOOK-M execution is not enabled by this protocol.

## Execution boundary

`run_hf_matched_validation.py --prepare` requires a clean immutable source SHA.
It materializes only ID ledgers and offline profiles; it submits nothing.
Use `slurm/hf_matched_validation.sbatch` with source root, run root, plan hash,
lane and stage exported. Register arrays with one shared admission controller:
at most four A100s globally, one GPU and 12 CPUs per task, two hours maximum,
`bii-gpu` / `bi_fox_dgx`. Analyze with `slurm/hf_matched_analysis.sbatch` on `bii`.
All artifacts and caches stay under scratch. Existing unrelated user changes
must not be included in a source commit without authorization.
