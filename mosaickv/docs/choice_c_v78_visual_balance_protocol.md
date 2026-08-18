# V78 stable-head visual-balance protocol

## Sealed v76 result

V76 is a negative label-blind calibration result. Every candidate passed the
unchanged memory gate with at least 34.74% lower active and
active-plus-residual bytes than Prompt Top-K. No candidate passed the unchanged
fidelity gate, so freeze job `18572260` correctly exited with status 2 and did
not create `frozen_candidate.yaml`. V77 remains blocked and must not be used to
tune another candidate.

The immutable binding is
`configs/ledgers/choice_c_v76_negative_calibration_seal.yaml`. It records the
protocol, input ledgers, gate report, candidate ranking, and label-blind
selection audit by SHA-256.

## Why 93% prompt mass was misleading

The v76 audit reads FullKV-generated outputs and cache traces, but no benchmark
reference, task score, or label. It found:

- image tokens carried only 10.96% of total prompt-side attention mass;
- MosaicKV retained 99.74% of text prompt mass but only 35.29% of image prompt
  mass on average;
- Prompt Top-K retained 46.60% of image prompt mass;
- only 3.125% of layer/head routes entered the graph planner; and
- spatial stratification changed at most 12.8 image blocks on average out of
  thousands selected per sample.

Thus the aggregate 93% number was dominated by text. The spatial reservation
acted only inside routed heads, while the other 31/32 stable heads continued to
choose text-heavy Prompt Top-K blocks. It could not materially change the
cache-wide image allocation.

## Independently motivated v78 change

V78 adds a deterministic visual-position target to the stable-head exact
selector. The selector still fills the largest attainable physical capacity
and ranks blocks only with prompt attention. Among equally full selections, it
requires a configured share of image/video positions when feasible. It never
uses OCR text, answer labels, benchmark references, or task scores.

The change reallocates already-budgeted exact positions. It does not change:

- the 27,500,000-byte active cap;
- the common four-token block size;
- mandatory-token policy;
- routed-head fraction;
- one-pass RoPE-safe prototype limit;
- residual storage, which stays disabled; or
- the 30% memory and five-percentage-point fidelity gates.

V78 preregisters a no-rebalancing control, a 50% stable-head visual target, and
a 60% target. These two targets bracket the lower-budget MosaicKV allocation
and Prompt Top-K's observed image-block allocation without using benchmark
labels.

## New disjoint calibration

V78 uses 16 new DocVQA and 16 new TextVQA calibration samples with seed
`20270903`. Materialization excludes every declared v73-v76 input directory,
sealed repository ledger, and prior observed result row. Candidate selection
uses only FullKV output agreement, generated-token agreement, active bytes,
residual bytes, and cache telemetry.

The chain is prepared but cannot be submitted from an uncommitted worktree:

```bash
bash scripts/submit_choice_c_v78.sh
```

The script submits only ledger materialization, GPU calibration, and freeze.
It does not submit v77 or any held-out validation. A future separately
versioned held-out set may be materialized only if v78 passes both unchanged
gates and creates `analysis/frozen_candidate.yaml`.
