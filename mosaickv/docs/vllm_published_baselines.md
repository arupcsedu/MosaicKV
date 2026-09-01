# Published-baseline projections on vLLM

Status: implemented and fail-closed for vLLM 0.10.2. Development A100 pilot
job `19166158` completed all three task workers with zero failed samples,
hash-bound selector audits, exact byte accounting, and request-isolation
evidence. This two-sample-per-task pilot is not paper-level evidence.

The common HF runtime remains the paper-faithful implementation of
`lookm_reimpl`, `prefixkv_reimpl`, and `vl_cache_reimpl`. vLLM owns a single
request-wide table of fixed-size token blocks, whereas those papers make
layer/head- or token-level decisions. The vLLM backend therefore executes a
version-pinned **request-wide whole-block projection** of each paper selector:

| Method | Source decision preserved | vLLM operation | Deliberately omitted |
|---|---|---|---|
| `lookm_reimpl` | Text-prior cumulative-attention pivot membership and scores | Vote/score projection of selected pivots, followed by atomic request-block release | LOOK-M K/V pivot merging; vLLM has no safe modified-KV injection transaction |
| `prefixkv_reimpl` | Offline-profile layer budgets and selected positions | Cross-layer membership-vote projection, followed by atomic request-block release | Per-layer physical block-table lengths and fixed-distance decode eviction |
| `vl_cache_reimpl` | Layer-adaptive sparsity budgets and per-head selected positions | Cross-head membership-vote projection, followed by atomic request-block release | Independent layer/head physical tables |

These rows are always labeled
`<method>__vllm_requestwide_projection`; they are never labeled official code
or an exact reproduction of the HF cache layout. The source trace records the
paper implementation label, selected membership, projection semantics, and
selector timing. `build_vllm_selector_plan.py` retargets token-sized HF
decisions to the audited vLLM block size, binds every sample to a source-trace
SHA-256, and refuses a missing or mismatched sample.

The runtime supports only eager execution with CUDA graphs and prefix caching
disabled. It preserves logical positions, mandatory boundary blocks, and the
next decode position while releasing only whole physical request blocks.
Prototype injection, residual repair, LOOK-M merged-value injection, and
simulated native rows remain unsupported.

Paper comparisons must use the same model revision, processor, media,
prompts, sample IDs, output length, decoding parameters, precision, vLLM
configuration, and physical active-byte budget. Because coarse projection can
change the realized retained fraction, comparisons use measured active KV
bytes rather than nominal token-retention percentages.

## Development pilot result

The pilot used LLaVA-1.5-7B, FP16, vLLM eager execution, 16-token serving
blocks, deterministic 64-token generation, seed 17, and the same two MMStar,
TextVQA, and DocVQA identities for every method. MosaicKV used 33.3%, 40.0%,
and 36.4% fewer active KV bytes than Prompt Top-K, respectively. It matched
Prompt Top-K task score on MMStar and TextVQA but was 0.25 lower on DocVQA.
Therefore the backend/byte gate passed, while the three-task quality gate did
not. The measured result report is outside the repository at
`/scratch/djy8hg/runs/mosaickv/vllm-published-baseline-pilot-v1/analysis/report.md`.

## Full paired campaign launcher

The full extension uses the exact 1,500 MMStar, 3,000 TextVQA, and 4,500
DocVQA identities already measured for FullKV, Prompt Top-K, and MosaicKV in
the HF/vLLM 9,000-sample composite. Its SHA-256 is
`4aecc162dadabfbf9cd12621dc053ce69304adaf26bb687537bde5adf63b44c4`.
It adds LOOK-M, PrefixKV, and VL-Cache reimplementation rows for both HF and
vLLM, producing 54,000 new rows. The dependent analysis combines those with
54,000 reused rows; it rejects missing samples, failures, duplicate
task/sample/backend/method keys, dirty-source manifests, incorrect effective
labels, and altered provenance.

The launcher creates 142 immutable, resumable shards: 24 MMStar, 47 TextVQA,
and 71 DocVQA. Each shard requests one A100, exactly 12 CPUs, 96 GiB host
memory, and at most two hours. The array admits at most eight concurrent A100
workers and lets Slurm start fewer when the node is partially occupied.

From a clean immutable commit, submit the GPU array and its dependent CPU
analysis with:

```bash
source scripts/cache_env.sh
scripts/assert_clean_worktree.sh
scripts/submit_vllm_published_baseline_full.sh
```

`MOSAICKV_CAMPAIGN_ROOT` may select a new immutable output root. Existing
completion markers and result rows are resumed; conflicting immutable inputs
or outputs fail closed. No model, dataset, pip, or temporary cache is written
under the home directory.
