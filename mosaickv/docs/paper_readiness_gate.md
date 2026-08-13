# MosaicKV paper-readiness gate

## Current status

MosaicKV is not paper-ready yet. The latest A100 run is useful implementation
evidence, but it used a dirty worktree and two retrospectively selected DocVQA
examples. It showed genuine exact, prototype, and residual tiers and about 40%
less active-plus-residual storage than the tested competitors. It did not show
quality noninferiority to Prompt Attention Top-K, and its end-to-end latency was
slower. Those rows must not be used as paper measurements.

The next protocol is versioned in
`configs/experiments/choice_c_paper_candidate_v37.yaml`. It is deliberately
blocked until the implementation is committed, the worktree is clean, and a
new disjoint evaluation ledger is materialized. Previously opened held-out IDs
remain excluded.

## New quality-recovery paths

### Prototype-covered residual storage

A bounded residual tier used to rank all source blocks globally. If several
high-error blocks belonged to one prototype, they could consume most of a small
CPU budget while another prototype received no recoverable member. At decode
time, MosaicKV could correctly identify that second prototype as risky but had
nothing to restore.

The new `prototype_coverage_then_risk` policy first ranks prototype clusters by
their best label-free reconstruction/value priority. It stores one member from
each highest-priority cluster before storing a second member from any cluster.
The original ordering is retained in the materialized residual index, so this
does not change logical positions or determinism. The existing global policy
remains available as an ablation.

### Prefix-safe FullKV-draft disagreement

The isolated draft rollout already evaluates a few future steps using the
untouched FullKV prefill. Previously, MosaicKV kept its query vectors and
attention maps but discarded its next-token distributions. This lost a useful
training-free warning signal: compressed decoding can be confident and still
disagree with FullKV, so entropy alone may not trigger repair.

The runtime now retains the draft probability distribution and computes
`KL(compressed || FullKV draft)` for the corresponding decode step. It exposes
this signal only while every draft input token matches the online prefix. Once
the prefixes differ or the draft horizon ends, the signal is disabled. A
disjoint, label-free calibration stage may choose a KL threshold under the same
trigger-rate and precision limits as other repair signals. If no threshold
passes, repair remains disabled. The online compressor never receives dataset
answers or oracle future queries.

### Budget-neutral quality recovery

Residual capacity must be resolved from a total-storage ceiling, not added on
top of an active-KV budget. The candidate protocol limits active KV plus encoded
residual payloads to at most 70% of the paired competitor's measured storage.
Within that ceiling, additional residual coverage may be used only if it
improves counterfactual repair precision on calibration data. FP8 or INT8 is not
presumed beneficial merely because it is smaller; lossless residuals remain the
default until a separate fidelity gate passes.

## New latency paths

The runtime now reports `host_orchestration`, the nonnegative gap between wall
clock end-to-end latency and the synchronized GPU-event phases. It also splits
the first explicit decode step from the mean of later steady-state steps. This
prevents a paper table from making named GPU phases appear to explain latency
that was actually spent in Python, tensor metadata work, or host/device
coordination.

Candidate optimizations must be evaluated in this order:

1. reduce sparse-graph degree and shortlist size, accepting the change only if
   the frozen FullKV-fidelity gate still passes;
2. compare RoPE-safe prototype groups of 15 and 31 blocks, which can reduce the
   number of prototype payloads and release slots for exact evidence;
3. remove host orchestration identified by the new timing residual, without
   changing selected blocks or prototype tensors;
4. test SDPA decode only as a common backend for every repair-disabled method,
   after eager token/logit parity passes; and
5. keep repair-enabled eager results separate until an attention-observation
   interface with equivalent numerics is verified.

A steady-state decode improvement is useful diagnostic evidence, but it cannot
replace the primary end-to-end result. Compression overhead must remain in the
reported total.

## Claims that can pass

Two questions must remain distinct.

1. **Equal active memory:** at paired measured active-KV bytes within 2%, does
   MosaicKV reduce the preregistered FullKV-fidelity error by at least 30% while
   remaining within the task-score noninferiority margin?
2. **End-to-end systems cost:** at the same model, examples, active-byte budget,
   backend, generation length, and measurement protocol, is MosaicKV at least
   5% faster in the point estimate with a positive paired-bootstrap lower
   bound?

Active-plus-residual storage is reported in addition to active device memory.
A quality-matched Pareto observation at different budgets may be described as a
frontier result, but not as an identical-budget row. Nominal retention ratios
are never the primary comparison.

## Evidence required before a paper table

- Retention 1.0 must pass the documented FullKV identity tolerance.
- The source tree must be clean and committed; every manifest must record the
  source/config SHA and complete hardware/software provenance.
- Candidate selection may use only disjoint design data and FullKV-fidelity
  diagnostics, never evaluation labels.
- A fresh evaluation ledger must exclude every opened design, internal, and
  held-out ID. Its size must be fixed by a power analysis before access.
- Each latency cell needs five warmups and at least twenty blocked, paired
  repetitions on an isolated GPU, with background-process and clock evidence.
- Point estimates, per-example/per-trial rows, and paired confidence intervals
  must all pass. A favorable mean alone is insufficient.
- `mosaickv_full` rows must actually exercise all three tiers. The repair claim
  additionally requires positive held-out repair recall, adequate precision,
  net quality gain, and measured overhead. A disabled or zero-trigger policy
  cannot support a repair-benefit claim.
- The final confirmation must include general VQA, document/OCR, and chart
  tasks and both LLaVA and Qwen-VL model families. Unsupported combinations are
  reported, not silently omitted.

Failure of any primary gate stops the full sweep. Negative results remain part
of the artifact and the method is described according to the evidence it
actually supports.
