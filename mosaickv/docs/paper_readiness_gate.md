# MosaicKV paper-readiness gate

## Current status

The implementation is ready for a larger, fixed **LLaVA-1.5 exact-selection
quality-memory experiment** on HF and vLLM. It is not yet ready to support a
paper claim that complete MosaicKV is generally superior.

The immutable implementation SHA
`0a0d6e546a0798a6540b967802b2f70947ed1391` passed 838 unit tests, 42 CPU
integration tests, Ruff, and strict MyPy. Clean A100 arrays then established:

- HF eager and ordinary vLLM FullKV produced identical scored completion
  tokens, answers, and official task scores for six samples across MMStar,
  TextVQA, and DocVQA;
- fixed-length post-EOS tokens are reported separately and do not alter the
  decoded completion;
- the hash-bound common-selector bridge produced native request-wide
  `mosaickv_exact` plans without fallback;
- every tested MosaicKV row used at least 30% fewer measured active KV bytes
  than its paired Prompt Attention Top-K row;
- mean official task score was equal or better on each of the three tasks; and
- two distinct compressed requests in one persistent vLLM engine passed an
  A-B-A isolation probe.

Clean native transaction job `19022355` additionally matched ordinary vLLM
over the complete 16-token timing stream for two retention-1 requests and
passed physical-block release, accounting, determinism, and isolation checks.
Its fallback selector is transport-only and did not produce a paper method row.

The compressed gate contained only two retrospective examples per task. It is
therefore development evidence, not a statistically adequate paper result.
Its exact answer agreement was 100% on MMStar and 0% on TextVQA and DocVQA,
even though official task scores were noninferior. This distinction is exposed,
not hidden: official benchmark score is primary, while exact string agreement
is a secondary diagnostic.

## Supported paper scope

The next fixed evaluation may use:

- model: `llava-hf/llava-1.5-7b-hf` at revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`;
- tasks: MMStar, TextVQA, and DocVQA, reported separately;
- HF eager FullKV, Prompt Attention Top-K, and the frozen HF MosaicKV candidate;
- ordinary vLLM FullKV, hash-bound vLLM Prompt Attention Top-K, and
  `mosaickv_exact__vllm_requestwide_projection`; and
- deterministic greedy decoding with one pinned common environment and model,
  dataset, prompt, media, tokenization, precision, generation, attention, and
  hardware provenance.

The vLLM result is exact-selection MosaicKV. vLLM prototype injection and
residual repair are unsupported and must not be labeled `mosaickv_proto` or
`mosaickv_full`. SGLang is outside the paper scope.

## Unchanged primary gates

For every task independently:

1. every paired MosaicKV row must use at least 30% fewer measured active KV
   bytes than Prompt Attention Top-K;
2. MosaicKV's paired mean official task score must be no lower than Prompt
   Attention Top-K;
3. paired bootstrap confidence intervals must be reported;
4. exact answer/token agreement, residual storage, compression time, TTFT,
   decode time, and end-to-end latency must be reported as secondary metrics;
5. missing, failed, non-finite, fallback, dirty-source, or provenance-mismatched
   rows fail the run; and
6. incompatible benchmark metrics must never be macro-averaged without an
   explicitly defined normalization.

Latency is not a rejection criterion, but it must include selector planning
and compression overhead. The clean development run was slower than Prompt
Top-K and cannot support a speedup claim.

## Fixed paper-quality execution

`configs/experiments/paper_quality_memory_llava_vllm_v1.yaml` resolves the
two-sample and missing-confidence-interval blockers at the experiment-design
level. It freezes 32 paired samples per task, 10,000 paired bootstrap
resamples, one model/revision, one deterministic 64-token generation policy,
and exactly three methods: FullKV, Prompt Attention Top-K, and
`mosaickv_exact`. The corresponding Slurm array is
`slurm/vllm_llava15_paper_quality_memory_v1.sbatch`; one A100 owns each task,
and every method within that task consumes the same immutable sample ledger.

The serving bridge accepts a hash-bound retention ratio per request. This is
needed because whole-block fitting can produce different attainable physical
ratios for different prompt lengths. The effective ratio is written in each
result row and trace; the run-config ratio is only the fail-closed fallback.
This permits one persistent vLLM engine per method and task rather than one
7B model load per sample, while preserving exact byte accounting and A-B-A
request-isolation evidence.

The fixed cohort is not described as untouched: prior dataset exposure is
disclosed in the protocol, as explicitly allowed for this run. Paper claims
must state that limitation. Per the current paper scope, latency and native
vLLM prototype/repair support are reported or marked unsupported but are not
release gates. No simulated prototype or repair row is permitted.
