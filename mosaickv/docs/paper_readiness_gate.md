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

## Remaining critical evidence blocker

The only blocker for the scoped LLaVA exact-selection quality-memory table is a
larger, fixed paired execution with enough examples to make the per-task score
and confidence intervals meaningful. The existing two-example result cannot
establish generalization.

Broader claims have additional blockers: Qwen/LLaVA family generalization,
HF prototype quality, repair benefit, latency competitiveness, and native
vLLM prototype/repair support each require separate positive evidence. A
disabled or unsupported component must remain an explicit unsupported row,
never a simulated or renamed result.
