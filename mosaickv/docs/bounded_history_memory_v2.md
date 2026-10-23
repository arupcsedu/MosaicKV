# Bounded lifetime KV: repairs and a memory-efficiency development candidate

This successor does **not** modify paper-native lanes or historical results.
Its target is smaller initial, final and peak live KV at preserved quality, not
merely execution success or reduction relative to FullKV. No such advantage is
claimed before measurement and independent confirmation.

## Repairs and evidence reuse

V1's four failed VL-Cache arms used adaptive layer lengths with a padded common
packer. The bounded controller correctly rejected padding. The correction
retains VL-Cache's density-derived layer weights, rescales them to the global
prompt byte cap, gathers each layer's selected head positions without padding,
then installs the unchanged history controller. Safety checks remain intact.
This remains a generalized common-ring `vl_cache_reimpl` adaptation, not
paper-native VL-Cache. Default non-ring VL-Cache paths are unchanged.

The selective retry binds the original plan, worker hashes and every referenced
artifact. It reuses 18 successful workers with their **original execution SHAs**
and schedules only indices 4, 9, 15, 20 under the corrective SHA. The new root
records mixed-source execution explicitly; it never relabels old runs as
executing new source. V1 configuration hashes remain unchanged.

PrefixKV's full-campaign repair reuses completed predictions. Its byte check
excludes FullKV, verifies all three compressed methods per sample, and preserves
separate PPL/ROUGE-L. It emits TikZ source without installing matplotlib. The
bootstrap point estimate is the observed corpus difference, not the replicate
mean. New analysis writes to a new directory, preserving the failed output.

## Candidate, not established quality improvement

The new MosaicKV arm retains the forecast, graph, utility and exact anchor,
then allocates physical layer slots using prompt-derived attention and value
evidence. For attention summary a and original value v, importance is based on
a_i ||v_i - sum_j a_j v_j||, with a small anchor-preservation term. No answers,
annotations, scores or oracle future tokens enter this computation. Absolute
value magnitude is retained across layers rather than normalizing every layer
to identical risk. The measured anchor output error is recorded diagnostically.

Complete layer slots move, not individual head capacities. Heads within a
layer have equal lengths; different layers own exact-size tensors. Allocation
stays within 0.5--1.5 times uniform prompt capacity, reserves mandatory boundaries
first, and optionally prioritizes prompt text and four strata per visual span.
These spatial proxies are **not OCR recognition or document boxes**. The exact
global byte total is enforced deterministically without padding. K/V values
and original logical/RoPE positions remain unchanged.

Every compressed method shares V2's history rule: at most 32 generated
positions, bounded by 20% of dense lifetime slots, and at least sink_count + 1.
Four generated sinks are protected; remaining slots form a recent ring. This
frees prompt capacity at small caps without granting MosaicKV a different
generation or mandatory-token policy. Short teacher-forcing targets reserve
only required updates; the same rule applies to every method, without truncation.

Attention/value output reconstruction is a surrogate. It may fail to preserve
autoregressive answers; that outcome must be retained and reported.

## Registered small test

`bounded_history_memory_v2.yaml` freezes revision, FP16 HF eager, seed, 1,024
timed output tokens, five warmups, four examples/task, common physical caps
32/40/80 MiB, and all six arms. Every compressed method runs at every cap on
identical IDs. The cohort excludes V1's four identities but its parent datasets
were historically evaluated: this is **retrospective development**, not an
untouched or paper-confirmatory cohort. PPL and ROUGE-L remain separate.

Use at most six aggregate A100s, one GPU and 12 CPUs/worker, two-hour limits,
the existing reservation, and the admission controller with explicit six-GPU
authorization. Retry only missing/failed compatible workers. FullKV supplies
quality references, never compressed-method latency comparisons.

Only equal realized initial/final/peak bytes permit direct within-cap quality
and latency comparisons. A separate equal-quality memory analysis requires a
frozen independent quality target, strongest/smallest eligible competitor rule,
paired intervals and independent data. Four samples cannot establish the final
claim; this protocol cannot automatically promote to a paper-scale campaign.
