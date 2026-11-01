# Consumed all-method complete-request engineering comparison

Require the complete ten-worker common streamed-prefill identity gate before
preparing or admitting timing. Bind all audited packets, physical backing bytes,
128-token logits and complete teacher-logit results. No new inputs or tuning.

Two one-A100/12-CPU workers, one consumed input per engine. All FIVE frozen local
precision variants receive the verified streamed constructor, lossless generated
history, one layer lease, BF16 eager attention and 128 output predictions. No
official-code labels. EviCastKV is the source-precision VARIANT, not exact-only
graph selection or all six original mechanisms. No weights are updated.

Each worker runs five warmups and twenty timed repetitions PER method. Randomize
all five methods' order inside each repetition with the immutable seed. Every
request does its own processor work, image and language prefill, selection,
encoding, retirement/GC and decode in the SAME engine. No reused prefill between
timed requests. Weights and loaded PIL media are warm; disk/model loading and
output evaluation/hash checks are outside the clock. CUDA events and synchronized
monotonic wall timing; phase records for every completed request, failures retained.

All 250 warmup/trial rows must reproduce their frozen 128-token FULL-vocabulary
logits and initial bytes bitwise. Hash each method's first complete packet. Audit
instrumentation is disabled identically for timing; memory peaks come only from
the separate tensor-audited canary, never from incomplete timing counters.

Report mean/std/p5/p50/p95 and 10,000-replicate paired trial bootstrap intervals
for candidate-minus-baseline time and time ratio. Also preregister conservative
Bonferroni bounds across eight engineering comparisons: four baselines times two
consumed inputs. Similar-time target: adjusted upper ratio <=1.05. A slower
result is retained, not hidden or followed by threshold changes.

These are different ENCODED-memory operating points under the original common
source-slot capacity. They are NOT byte-matched curves. Teacher PPL and FullKV-
reference ROUGE-L remain separately labeled and may disagree. Neither 20 timing
repetitions nor bitwise own-parent output preservation creates independent quality
evidence. No fresh cohort, calibration, paper promotion or larger campaign is
authorized by this engineering gate. Do not infer whole-request memory dominance.
