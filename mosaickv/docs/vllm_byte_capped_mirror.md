# vLLM mirror of the active HF campaigns

Status (2026-10-12, audited after completion): GPU verification array
**21302511**, tasks 0–6, completed with exit code `0:0`, taking 3:21–4:27 per
shard. Revalidation of all seven immutable artifacts passes retention-1 token
parity, A-B-A isolation, exact realized 40 MiB matching, and output integrity.
These remain **development-only**, two-example-per-shard measurements, not
canonical paper evidence or statistical quality validation. No full vLLM
campaign has been submitted.

The audit also found stale retention-ratio telemetry in the native transaction
records and an identity-arm planning-time attribution error. Corrections are
CPU-tested below; a corrected canonical GPU preflight is still required.

## Isolation from HF

### V100 sweep feasibility (2026-10-13)

The requested V100 sweep is **unsupported by the current native path**, not
submitted. It is not sufficient to change `--gres=gpu:a100:1` to V100:

- Installed vLLM 0.10.2 `platforms/cuda.py:CudaPlatform.get_attn_backend_cls`
  (lines 361–373 and 404–411) defaults older V1 devices to FlexAttention and
  explicitly excludes Volta/Turing from FlashAttention-2 in V0 selection.
- MosaicKV `backends/vllm_native_plugin.py:_patch_worker.prepare_inputs`
  accepts only `FlashAttentionMetadata`; other metadata types are rejected.
- The installed FlexAttention backend uses its own physical/logical mapping
  and mask construction in `v1/attention/backends/flex_attention.py`.
  Swapping the backend name would not port or validate this transaction.

`require_native_cuda_support` now rejects SM < 8.0 (including V100 SM 7.0),
missing CUDA devices, and non-FlashAttention native execution before loading
model weights. CPU regressions cover V100, mixed visible devices, alternative
metadata backends, and early engine rejection. The updated targeted suite
passes **85 CPU tests**. This is not a claim that
ordinary vLLM FullKV is universally unsupported on V100, nor that every SM >=
8.0 device is already validated. A V100 port needs a separately audited
metadata/masking path plus GPU identity, isolation, and byte-accounting tests.
No FullKV-only or simulated rows substitute for that compressed sweep.

The scheduler also reports cluster-wide maintenance through **2026-10-15
06:00 America/New_York**, including the V100 nodes. The Fox reservation and
general quota both remain subject to maintenance. Existing HF jobs and source
snapshots were not modified for this feasibility check.

HF array `21292820` continues from its unchanged clean worktree at
`/scratch/djy8hg/worktrees/mosaickv-hf-byte-curve-2c7a266`. No running GPU job
was canceled, and the common Python environment was not installed into,
upgraded, or otherwise modified. Model and dataset caches remain under scratch.

The new development workers use a different detached worktree:
`/scratch/djy8hg/worktrees/mosaickv-vllm-bytecap-development-20261012`.
Their uncommitted changes are explicitly classified as development-only, bound
to a 527-file SHA-256 inventory and immutable job/config/catalog hashes. They
are **not canonical paper measurements**. A clean committed source and a passed
canonical preflight are required before the campaign stage can execute.
Changing the main checkout cannot change these submitted workers.

## What was corrected

1. Added a physical-byte-cap path to the common serving selector catalog. The
   target is `min(source_blocks, floor(cap / physical_bytes_per_block))`, with
   every allocated token slot, including final-block padding, charged. A cap
   smaller than mandatory block storage fails; it is never rounded upward.
2. Bound the cap and KV geometry into each hashed catalog entry. The runtime
   checks them against the loaded checkpoint/configuration before constructing
   the engine, then verifies committed block IDs and active bytes after prefill.
3. Added the installed vLLM LLaVA-v1.6-Mistral path. Its fallback prompt matches
   the HF adapter, and the existing CLIP-only SDPA compatibility hook now also
   covers that checkpoint. Decoder attention remains vLLM FlashAttention.
4. Explicitly configured one request at a time, no chunked prefill, no cascade
   attention, no prefix caching, eager execution, and no CUDA graphs. This bounds
   vision profiling memory and prevents implicit engine defaults from changing
   the post-prefill transaction. These are new execution controls, so older
   latency rows must not be pooled with these runs without checking them.
5. Built an immutable HF-to-vLLM launcher that verifies completed HF source
   predictions and trace hashes; preserves the model/revision, dataset/revision,
   ordered sample IDs, generation parameters, seed, and method-specific source
   selections; and writes all derived inputs/outputs outside HF directories.
   Separate engine processes prevent method switches from leaking plugin state.

Installed-source evidence: vLLM 0.10.2
`vllm/model_executor/models/registry.py` registers
`LlavaNextForConditionalGeneration`; its implementation is
`vllm/model_executor/models/llava_next.py`. The FlashAttention cache layout in
`vllm/v1/attention/backends/flash_attn.py:FlashAttentionBackend.get_kv_cache_shape`
requires block sizes divisible by 16. Explicit engine settings are fields on
`vllm/engine/arg_utils.py:EngineArgs`. MosaicKV's native transaction still requires
one `FullAttentionManager` and rejects unsupported cache/kernel arrangements.
No external package source or model weights were patched.

## Comparison boundary—not identical HF and vLLM algorithms

All compressed methods in one vLLM lane have the same 16-token serving blocks,
first/last mandatory serving blocks, physical cap, model, media, prompts,
precision, deterministic decoding, and engine settings. Exact per-sample
realized-byte equality is checked before a comparison passes.

HF's 1/4-token independent layer/head decisions cannot be represented exactly
by vLLM's request-wide block table. Rows remain named
`<method>__vllm_requestwide_projection`, including PrefixKV and VL-Cache.
They are not official reproduction rows, and LOOK-M merging, independent
PrefixKV layer lengths, prototypes, and residual repair are not implemented
by this projection. The actual serving mandatory policy is recorded separately
from the source HF policy.

For LLaVA-1.5 FP16, a 16-token block occupies 8 MiB across all decoder layers
and KV heads. Caps 16/22/30/40/60 MiB therefore permit 16/16/24/40/56 MiB when
the prompt is long enough. A 10 MiB cap cannot retain both mandatory boundary
blocks of a multi-block prompt: mark it infeasible, not a successful 10 MiB
comparison. LLaVA-v1.6-Mistral's GQA geometry has a 2 MiB serving-block increment.
These are layout calculations, **not measured experiment results**.

The selector is replayed from a hash-bound HF trace. Engine request/decode times
are measured by vLLM; the existing source-selector time is separately traceable.
This does not measure an online native vLLM forecasting/graph implementation,
and does not include an additional online HF prefill. Do not use its reconstructed
selector-plus-engine timing as a native-serving end-to-end speedup claim.
FullKV is a quality/correctness reference and is excluded from latency comparisons.

## Submitted verification

Each task uses two fixed, previously evaluated identities, not a fresh held-out
sample. Each executes ordinary vLLM FullKV, a native retention-1 identity arm,
and the compressed methods at 40 MiB. Every engine runs an A-B-A isolation check.

| Array task | Source lane | Task | Compressed methods |
| --- | --- | --- | --- |
| 0 | Current LLaVA-1.5 HF curves | MMStar | Prompt Top-K, MosaicKV exact |
| 1 | Current LLaVA-1.5 HF curves | TextVQA | Prompt Top-K, MosaicKV exact |
| 2 | Current LLaVA-1.5 HF curves | DocVQA | Prompt Top-K, MosaicKV exact |
| 3 | Verified PrefixKV 40 MiB HF pilot | TextVQA | Prompt Top-K, PrefixKV reimpl, MosaicKV exact |
| 4 | Verified PrefixKV 40 MiB HF pilot | DocVQA | Prompt Top-K, PrefixKV reimpl, MosaicKV exact |
| 5 | Verified LLaVA-v1.6 VL-Cache HF pilot | TextVQA | Prompt Top-K, VL-Cache reimpl, MosaicKV exact |
| 6 | Verified LLaVA-v1.6 VL-Cache HF pilot | DocVQA | Prompt Top-K, VL-Cache reimpl, MosaicKV exact |

Array resource request: one A100, 12 CPUs, 96 GiB RAM, at most two hours per
task, `bii-gpu` / `bi_fox_dgx`. As of October 13, future runs share—not add
to—the aggregate maximum of four A100s at all times. This supersedes the
overnight eight-GPU allowance used at submission. Pending
`JobHeldUser` is the controller's deliberate admission mechanism.

Artifacts:
`/scratch/djy8hg/runs/mosaickv/vllm-bytecap-development-20261012/preflight-v4/`.
Each task records per-arm JSONL, Parquet, manifests, traces, stdout/stderr, and
a terminal validation result, including failures. The gate checks retention-1
tokens, A-B-A isolation, complete identities, finite timing/score fields, fixed
output length, identical prompts/media, exact realized bytes, and maximal
feasible use of the cap. Compressed quality is reported, not assumed equal.

## Expansion inventory and restrictions

### Completion audit and validation correction

The original `status: passed` certified execution only. It did not establish
compressed quality parity. The validator now makes that scope explicit and
reports paired score differences against every compressed baseline separately.
It additionally checks the entire trace hash, A-B-A token evidence (not just
the manifest boolean), executed-versus-bound configuration, source provenance,
reference annotations, trace/raw byte consistency, and HF source token traces.
No predictions, scores, thresholds, budgets, or completed worker artifacts were
changed. The audit tool writes to a new derived directory and runs on CPU only.

Two concrete telemetry corrections are now implemented for subsequent runs:

- Send the catalog's effective retention ratio and block size in per-request
  sampling metadata. The old transaction audit recorded the HF configuration's
  nominal placeholder (often `0.99`), despite committing the correct explicit
  block IDs and physical bytes. The derived audit records each mismatch and
  fails its telemetry gate; campaign construction rejects it.
- Retention-1 catalog construction now assigns zero offline selector-planning
  cost. Keeping every block does not execute the compressed HF source's planner;
  its unrelated recorded cost must not be charged to the identity arm. Native
  transaction timing remains measured separately.

Neither change alters the selected block IDs. They do not solve or conceal
compressed quality loss. Original raw telemetry is retained unchanged.

Measured task-score percentages (two examples per row; task metrics must not
be averaged across tasks):

| Lane / task | FullKV | Prompt Top-K projection | MosaicKV projection | Published-method projection |
| --- | ---: | ---: | ---: | --- |
| Current / MMStar | 0.00 | 0.00 | 0.00 | Not run |
| Current / TextVQA | 0.00 | 15.00 | 15.00 | Not run |
| Current / DocVQA | 0.00 | 0.00 | 0.00 | Not run |
| PrefixKV / TextVQA | 50.00 | 15.00 | 15.00 | `prefixkv_reimpl`: 0.00 |
| PrefixKV / DocVQA | 41.67 | 0.00 | 0.00 | `prefixkv_reimpl`: 0.00 |
| VL-Cache / TextVQA | 50.00 | 15.00 | 15.00 | `vl_cache_reimpl`: 50.00 |
| VL-Cache / DocVQA | 41.67 | 0.00 | 0.00 | `vl_cache_reimpl`: 0.00 |

Every compressed sample realizes exactly 41,943,040 active prefill KV bytes.
The quality reference is ordinary vLLM FullKV, not HF FullKV. All serving
variants retain the `__vllm_requestwide_projection` suffix in raw output.
MosaicKV matches Prompt Top-K's answers in these samples, but its VL-Cache-lane
TextVQA score is 35 percentage points lower than the VL-Cache projection.
Zero-score samples provide no positive quality-preservation evidence; neither
superiority nor statistical non-inferiority follows from this small pilot.
HF and vLLM compressed answers can differ because their cache layouts differ;
the audit reports this instead of claiming cross-backend bitwise parity.

Derived report (includes compressed-only timing breakdown and separate HF
comparisons in JSON):
`/scratch/djy8hg/runs/mosaickv/vllm-bytecap-development-20261012/audit-20261012-v3/`.

Reproduce the CPU audit with a new output directory:

```bash
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/analyze_vllm_byte_capped_mirror.py \
  --index /scratch/djy8hg/runs/mosaickv/vllm-bytecap-development-20261012/jobs.v4.json \
  --index-sha256 5b86bd666b3c01c79245b77c3efe78d9b2f9021d289b9cb3247d96863c63aad6 \
  --output-dir /scratch/djy8hg/runs/mosaickv/vllm-bytecap-development-20261012/audit-new
```

The strengthened validator and existing selector/runtime/schedule tests pass
76 CPU tests; Ruff passes for the changed audit/validation code. No source or
package changes were made in the running HF worktree. The canonical expansion
still requires an explicitly authorized clean source commit and compatible
canonical preflights. This audit does not grant those conditions retroactively.

### Source coverage

| HF lane | Intended mirrored tasks/caps | Remaining prerequisite |
| --- | --- | --- |
| Current 256-example LLaVA-1.5 curves | MMStar/TextVQA/DocVQA; 16/22/30/40/60 MiB | All source shards complete; canonical vLLM preflight on clean source |
| 256-example PrefixKV shared-cap lane | TextVQA/DocVQA; feasible 20/40/80 MiB | Missing submodule/profile repaired October 13; awaiting HF execution after maintenance; 10 MiB serving point infeasible |
| 256-example VL-Cache shared-cap lane | TextVQA/DocVQA; 10/20/40/80 MiB | Completed HF sources and verified LLaVA-v1.6 GPU compatibility |
| COCO CIDEr | Available to the common lmms route | Completed matched-cap HF source/ledger and a task-specific preflight |
| LOOK-M, LLaVA-Description/MM-Vet PPL or frozen-reference ROUGE-L, MileBench | Do not silently substitute OCR metrics or old nominal-ratio results | Matching source artifacts and paper-native scorer/aggregation integration are separate work; not covered by this array |

The builder accepts `--stage campaign` only with a passed canonical preflight
report for the same source/task. Development results cannot authorize that
stage. Do not reuse a nominal-retention row as a same-budget result. No new
commit was created by this implementation task.
