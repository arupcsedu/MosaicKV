# V76/V77 disjoint OCR-evidence protocol

## Sealed parent result

V75 is an immutable negative held-out result. It passed the active-memory gate
on DocVQA, MMStar, and ChartQA, but failed the preregistered DocVQA quality
gate. The versioned seal is
`configs/ledgers/choice_c_v75_negative_heldout_seal.yaml`; it binds the v75
protocol, frozen candidate, task ledgers, paired trials, summary, and gate
report by SHA-256.

No v75 answer, reference, task score, sample-level failure, or selected-block
trace may be used to select or tune a v76 candidate. The task-level negative
result is used only to justify opening a separately numbered document-quality
development cycle. V74.6 and its frozen candidate remain unchanged.

## Independently motivated design change

The LLaVA processor exposes an authoritative image-patch grid but does not
expose OCR boxes. V76 therefore does not fabricate OCR text, rows, columns, or
page regions. Its new `spatial_stratified_prompt` reservation divides the
known patch grid into fixed spatial strata independently in every layer, KV
head, and image. Within each stratum it ranks blocks by prompt-window
attention, then local utility, then node ID. A first block from an uncovered
stratum always outranks a second block from an already represented stratum.

This change targets a general document failure mode: a global top-attention
ranking can repeatedly retain nearby patches while dropping low-mass evidence
elsewhere on a page. It remains label-independent and preserves blocks exactly.
The bounded one-pass prototype path remains limited to at most two RoPE-safe
prototypes per routed head. The active byte cap remains exactly 27,500,000
bytes for every candidate; v76 does not reopen byte-cap tuning.

## V76 calibration

V76 materializes 16 new DocVQA and 16 new TextVQA calibration examples with
seed `20270829`. Materialization excludes repository ledgers, all declared
v73/v74 inputs, all sealed v75 held-out inputs, previous observed-result rows,
and matching repository ledgers. The historical main-quality ordering ledger
is not treated as observed because it enumerates the full TextVQA split;
actually executed result rows remain excluded. Candidate selection stays
label-blind and uses only FullKV output agreement, token agreement, active
bytes, and residual bytes.

The unchanged gates require, on every calibration task:

- at least 30% fewer active bytes than Prompt Top-K;
- at least 30% fewer active-plus-residual bytes than Prompt Top-K;
- FullKV answer agreement no more than five percentage points below Prompt
  Top-K; and
- no more than 5% prototype logical bytes.

If no candidate satisfies every gate, the freeze job exits with status 2 and
does not create `frozen_candidate.yaml`.

## V77 untouched validation

V77 remains blocked until a v76 freeze exists. Only then can it materialize 48
new examples each from DocVQA, TextVQA, MMStar, and ChartQA with seed
`20270831`. It excludes all v76 calibration inputs and all v75 held-out inputs,
and every ledger binds the v76 freeze SHA-256. FullKV, Prompt Top-K, and the
single frozen MosaicKV candidate run on identical examples with two measured
repetitions. Latency is informational and cannot reject the quality-memory
result.

The submit scripts intentionally separate the stages:

```bash
bash scripts/submit_choice_c_v76.sh
# Only after v76 creates a passing frozen_candidate.yaml:
bash scripts/submit_choice_c_v77_after_freeze.sh
```

Both scripts require the common environment, scratch-only caches, and a clean
committed source snapshot. Preparing these files does not authorize either
submission.
