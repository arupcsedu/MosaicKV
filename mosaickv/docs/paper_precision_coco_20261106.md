# COCO paper-readiness quality validation (2026-11-06)

This protocol collects missing new-input quality evidence. It does not promote
the two consumed engineering examples into an independent quality result, and
does not authorize a compressed-baseline superiority claim.

## Frozen scope

- **Candidate:** EviCastKV shared-source precision/storage variant, based on
  `caa4f79`. Historical method ID `mosaickv_exact` is retained, but all six
  principal forecasting/graph/selection/prototype/residual/repair mechanisms
  remain inactive in this variant. Do not call this full-system validation.
- **Reference:** ordinary FullKV, for quality and memory characterization only.
  It is excluded from compressed-method latency comparisons.
- **Model:** `llava-hf/llava-1.5-7b-hf`, revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, BF16/HF eager.
- **Dataset:** cached pinned `lmms-lab/COCO-Caption` validation split, revision
  `c3213b54e173ae86517d481de17f31439d6f8c72`.
- Preserve the official one-sentence-caption prompt, processor, RGB input,
  tokenization, greedy generation and 128 predictions. Score only through the
  first EOS; retain all timed tokens and any output-limit flag.
- Freeze the original profile allocator's 0.5 source-cap convention, including
  its maximum-layer scratch reserve; this is not exactly half of all token
  slots and is not an encoded-byte-matched budget. Preserve the training-free
  0.5 shared prior, three-bit source representation, protected words, correction
  thresholds, lossless generated history and per-layer text-reader modes.

The optimized direct constructor previously admitted only 600/604-token
consumed prompts. The unmodified COCO caption template produces **603 tokens**.
A default-off request-local scope adds only that geometry. Do not pad or alter
the prompt. Odd-length allocation preserves the original physical cap and
scratch cost. Bind text accounting to each new request's actual private base
instead of asserting another image's historical byte size. Preserve the 6-MiB
overlay allowance, 32-KiB/layer and 1-MiB/request numerical-repair limits.
Exceeding a bound fails. Legacy geometry and exact historical-size assertions
remain unchanged outside the explicit scope.

## Finite execution

1. Initialize all pinned submodules and pass the complete CPU suite on the
   clean immutable source before admitting GPU work.
2. Select **258 identities** by ID hash/seed 20261106, excluding the historical
   4,096-ID COCO pool and explicitly inventoried top-level and `cohort/`
   `references.json` files. Preserve all 5/6/7-caption annotations and media
   hashes. Reject RGB duplicates within the selection or against the two
   explicit candidate-tuning inputs; never replace samples after outcomes.
   The first two are canaries, not members of the 256-example quality cohort.
3. Run four canary workers (two inputs times candidate/reference). Require
   FullKV-versus-generate/reinjection parity, 128-word-vector retention-1
   identity with all lossy transforms bypassed, and candidate optimized versus
   its own compressed parent bitwise equality across all 128 vocabulary vectors.
   Save tensors, numerical-repair audits, ownership traces, raw outputs and
   manifests. A failure stops progression and remains recorded.
4. Only after those controls pass, run **256 images × two methods**, eight
   examples per immutable shard, five excluded warmups per worker. Parameters
   are frozen before these outputs; no outcome-based tuning or sample removal.
5. Recompute complete-corpus CIDEr and document frequencies in each of 10,000
   paired bootstrap replicates. Validate several full draws against the pinned
   official reducer. Report the candidate-minus-FullKV difference and 95% CI.
   The unchanged strict quality gate requires its lower bound >=0. More samples
   do not guarantee a pass. Keep negative results.
6. Report initial/final active cache backing and candidate decode-owned peak.
   **No whole-request KV-only peak is certified.** Separately measure complete
   request maximum allocated/reserved GPU HBM, including model weights,
   prefill, construction, decoder workspace, graph setup and all live owners.
   Do not relabel allocator peaks as KV-only storage.
7. If quality and the measured initial/final cache and whole-request allocated
   HBM gates pass, collect descriptive candidate timings on six fixed cohort
   indices (2,45,88,131,174,217): five warmups and 20 repetitions. Report p50/p95,
   not a comparative latency win. No FullKV latency ratio enters acceptance.

The complete-request timer includes media loading, processor, image/language
prefill, construction, cold graph lifetime, decoding, CPU output export and
EOS-aware detokenization. Model loading, artifact I/O and metric scoring are
outside. Reset allocator peaks before the request, without compressed-path
inner resets. Actual streaming TTFT is unsupported in this nonstreaming study.

At most **six aggregate A100s**, one GPU/12 CPUs/48 GiB each, two-hour worker
limits. Register arrays with the existing aggregate admission controller.
Chain analysis with `afterany`, so failed producers are analyzed. Use only the
common environment and scratch caches. No packages or manuscript are changed.

## Claim boundary

New IDs are disjoint from the explicitly bound inventories. Global historical
cross-dataset/source-group media freshness is **not certified**; this is not a
globally untouched confirmatory cohort. The paper must disclose that limit.

Every raw row links the model/data/evaluator/environment/source/config and
current scheduler provenance. Raw JSONL, canary tensor captures and derived
Parquet remain separate; failed rows and attempted denominators are retained.

The latest compressed-baseline engineering results had different encoded byte
outcomes, not the equal physical cache boundaries required by
`shared_byte_comparison_v2.yaml`. Therefore **no new PrefixKV/VL-Cache/LOOK-M
comparative GPU rows are submitted by this protocol**. They still require their
separate physically matched lanes and fidelity controls. This run cannot
establish superiority over the strongest compressed baseline, a 30% advantage,
official-paper reproduction, or complete six-component EviCastKV performance.
Successful execution alone must never be called a positive paper result.

## Admission-only retry

The original canary array 22540573 never entered GPU inference: its controller
correctly rejected adoption of unowned user holds, because the launcher omitted
the receipt-bound ownership field. Preserve the canceled unstarted array and
its failed controller logs. The replacement registers `adopt_existing_holds`
only for the exact new array submitted with `--hold`. No other holds are adopted.
It reuses the already sealed ledger byte-for-byte after checking identical
protocol, model/dataset configuration and frozen runtime options. No reference,
identity, threshold, arithmetic or candidate policy is changed by the retry.
