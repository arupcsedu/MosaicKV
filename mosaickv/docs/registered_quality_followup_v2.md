# Registered October 29 bounded quality follow-up

User authority: run the discussed PrefixKV and Fox phases autonomously, with a
six-A100 aggregate ceiling. Two finite lanes reserve three A100s each; a successor
array starts only after its predecessor and `afterany` CPU check. One A100,
12 CPUs, 64 GiB, one-hour quality/two-hour timing limits. No other-user jobs change.
After Fox's terminal callback verifies no Fox GPU jobs remain, its slots transfer
to PrefixKV (six total), including future PrefixKV arrays. No intermediate canary
callback can make this transfer. The source-bound terminal marker and scheduler
amendment are retained. All GPU submission waits for clean-source CPU tests.

## PrefixKV: freeze the unchanged candidate, test intermediate memory

- Pinned LLaVA-1.5, FP16, eager, existing processor/media/generation controls.
- Exactly 1,024 output tokens and complete annotation teacher-forcing targets.
- 32 examples each from LLaVA-Description and paper-style MM-Vet, selected by an
  input-ID hash before outcomes. Exclude the previous eight and all explicitly
  enumerated tuning identities. The original benchmark was used historically:
  these are **retrospective development**, not globally untouched confirmation.
- All three methods at every cap: Prompt Top-K, the byte-rescaled append-only
  `prefixkv_selection_only_reimpl`, and frozen MosaicKV exact, at 80/120/160 MiB.
- FullKV once per input/shard, reference hashes held fixed across compressed
  methods. Original baseline reproduction/fidelity remains separate.
- Eight-example shards, 80 workers, 640 quality requests, no quality warmups
  (identical across methods). Quality-run times remain descriptive. Numerical
  FullKV and retention-1 controls run in every worker as before.
- First run references, then first-shard 120-MiB execution checks for all three
  methods. Gate only correctness/storage, not favorable scores. Then complete
  the remaining preselected 32-example matrix.
- Corpus PPL sums NLL/counts within every resample. ROUGE-L stays separate.
  Registered 10,000 paired bootstrap repetitions; family 16 for two tasks,
  two candidate caps versus 160 MiB, two metrics, and two baselines. Zero margins
  unchanged. FullKV is excluded from compressed latency comparisons.
- Freeze the smallest candidate passing both metrics on both tasks versus the
  PrefixKV adaptation. A nearest-competitor claim additionally requires Top-K.
  No cherry-picking one favorable metric/task. Report initial/final/peak memory;
  never substitute nominal caps or initial-only savings for final savings.

## Fox: separately versioned installed-output guard

Keep prompt-only exact selection, 64 tokens, original model/revision, eager
FP16, 20/40 MiB, route fraction 1/256, graph degree, shortlist, coefficients and
mandatory policy. No draft, training, prototypes, repair or new serving path.

The previous variants preserved individual prompt positions but averaged GQA
query heads. The new opt-in `output_guarded` variant preserves both axes and
compares the actual graph-selected subset to its same-capacity Top-K anchor.
Accept the subset only when each query retains at least 95% of anchor mass and
its renormalized attention-output squared error is no worse. Otherwise install
the anchor for that route, not FullKV. Count actual accepted/changed positions.
This is a prompt-level guard, **not** proof of autoregressive quality.

Run an 18-request, two-consumed-input numerical/packing canary. Only after it
passes, run 32 preselected development images disjoint from the previous 64
but drawn from the historical COCO pool; do not call them fresh. FullKV plus
Top-K, VL-Cache reimplementation, frozen MosaicKV, and the guarded variant at
both caps: 288 requests. Corpus CIDEr and 10,000 media-cluster paired bootstrap
replicates; family six. Strict positive adjusted lower bounds required against
all three baselines at both caps to advance. All failures and slower timings
remain in artifacts. No routing-fraction or weight search follows a failure.

## Confirmation and latency

Independent confirmation remains conditional on an actual source/media
disjointness certificate, frozen parameters and power analysis. The existing
historical media inventory is incomplete. Neither full user authority nor a
positive development result makes it complete. The finite chains stop with a
structured blocker rather than manufacture a paper claim. They cannot silently
reuse historically inspected native benchmarks as untouched confirmation.

The fixed latency diagnostic uses the first preregistered calibration input per
task, five warmups and 20 repetitions per method, counterbalanced method order,
and identical GPU controls. Run Top-K/PrefixKV/MosaicKV at 120 MiB and PrefixKV
at 160 MiB. Report technical-repeat intervals (family six), p50/p95 and full
traces, not population-latency evidence from one input. It follows complete
calibration regardless of favorable or unfavorable quality, never failed
execution. A slower result is a reported trade-off, not a suppressed measurement.
No FullKV latency comparator or reused prefill is allowed. October-29 revisions authorize
this bounded successor, superseding the prior writing-only stopping scope; they
do not authorize an unbounded search or repeat-until-significant campaign.
