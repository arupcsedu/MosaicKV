# Consumed-input storage, quality and latency successor

Successor to `3f577cc`; only LLaVA-Description `000000532947` and MM-Vet
`v1_197`. These are consumed diagnostics, never fresh benchmark evidence.
The main checkout and historical campaigns remain unchanged.

## Four fixed phases

1. Compact replay: fixed two-bit integer BF16-word deltas 0/-1/+1 and literal
   original words for other differences. Preserve signed zero and NaN payloads.
   Require original K/V bit equality in three repeats and unchanged 32-token
   ordinary selected-cache logits. Zero exceptions cost zero bytes; raw fallback
   prevents expansion. Test compact full-union and token-hybrid sources with
   both BF16 reduction flags isolated to projection replay. Always restore the
   flag; original prefill, model decoding and weights remain unchanged.
2. Timing: five warmups and 20 randomized paired repetitions per selector and
   representation (encoded K/V, original-shape token source, minimum-cost
   compact-corrected source). CUDA events and synchronized wall boundaries;
   independent mutable histories. Instrumentation is disabled inside timing;
   phase one owns memory evidence. This is decode-transport timing excluding
   original prefill, selection and construction, not end-to-end latency. Do not
   compare FullKV latency. Retain all trials and bitwise output checks.
3. Source-sharing quality: add 2/8/32 positions per head only from original
   source rows already archived for other heads. No selected evidence is
   removed. Keep uniform per-head growth in each layer. Prioritize the complete
   generation delimiter, then prompt-window attention. No benchmark answers,
   references or failure identities are inputs. Apply additions to both Top-K
   and EviCastKV. EviCastKV diagnostic caps: <=62 MiB initial and <=96 MiB
   declared decode-owned peak. Report unsupported caps, never silently change
   them. Actual storage differs between representations; a common configured
   cap alone is not called exact byte matching.
4. Baseline diagnostics: the same two consumed inputs, FullKV quality reference,
   Prompt Top-K, byte-rescaled append-only `prefixkv_reimpl`, generalized
   LLaVA-1.5 `vl_cache_reimpl`, `lookm_reimpl`, and EviCastKV. Also include the
   separately named two-position source-sharing variant for Top-K and EviCastKV.
   Each method receives available lossless formats and 5+20 paired randomized
   decode trials (FullKV excluded from latency). Original retained source-slot
   capacity is identical before storage coding for the six original methods;
   actual encoded bytes differ and are never called byte-matched. LOOK-M uses
   deterministic balanced per-layer capacities to spend the common source cap
   and retains its merged K/V verbatim. Published-method rows are adaptations,
   not paper-native reproductions. Report fixed-prefix KL/token agreement and
   ROUGE-L against the same frozen 32-token FullKV model output. This is not
   annotation PPL or independent task-score evidence. No new cohort is used.

Source sharing is a separately labeled candidate, not the frozen paper method.
It tests storage-aware evidence additions, not another donor-removal or
prior-mixing sweep. Preserve original K/V, positions and mandatory tokens.
The unchanged guard uses all 32 FullKV-generated prefixes, checkpoints
1/4/16/32, absolute/relative tolerances 1e-7/1e-6, absolute KL <=0.02 and max
logit difference <=1.0. No new wrong token is allowed where the anchor was
correct. Require a meaningful change plus improved KL or recovered token.
Own-greedy transport separately requires bitwise selected-cache identity.
These model-prefix diagnostics are not benchmark scores or paper-native PPL.

Accounting includes payload, corrections, indices, shared RoPE, raw/sealed
history, leases and observed tensor temporaries. It excludes inactive original
diagnostic sources, construction, weights, attention activations, internal
GEMM scratch and allocator reservation. Whole-request peak is uncertified.

The first three phases: two one-A100 workers, 12 CPUs, 30-minute limit. The
baseline phase has sixteen one-method/one-input workers, at most six concurrent
A100s, with the same CPU/time limits. All phases have afterany CPU
analysis. Only completeness/numerical success submits the next fixed phase.
The finite chain cannot submit fresh evaluation, calibration or a paper
campaign. Negative quality stays negative. Aggregate GPU limit six.
All caches and outputs stay under scratch.

Further baseline diagnostics must preserve PrefixKV, VL-Cache and LOOK-M
algorithms. In particular, LOOK-M's merged K/V must not be replaced with
unmodified source K/V: use the common lossless K/V codec when exact source
reconstruction is inapplicable. A LLaVA-1.5 VL-Cache diagnostic is a generalized
reimplementation, not its Mistral paper reproduction. All actual bytes and
unsupported transformations must remain explicit. Larger claims require a
separately frozen, independently validated candidate; none is authorized here.
