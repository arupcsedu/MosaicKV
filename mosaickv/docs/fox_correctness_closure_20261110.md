# Prof. Fox review: V7 correctness and evidence closure

This batch follows the completed V6 controls at source
`7cf31bf285ea7b5f4ae7d7caa615921f2e89f417`. It does not replace their raw
records, negative numerical results, or statistical conclusions. EviCastKV is
the paper name of MosaicKV; source-storage and exact-selection variants remain
separately identified.

## Registered sequence

1. Run the complete CPU suite from a clean source, with initialized pinned
   submodules and GCC 11.4. Any failure, error, or skipped test blocks GPU work.
2. Review all six saved allocator histories on CPU. Rules bind exact source
   paths, hashes, and stack frames. Independently recompute live allocation
   intervals and the KV-plus-workspace peak; unknown ownership blocks
   certification. A complete certificate need not show a memory advantage.
3. Inventory historical identities using eight partitioned CPU workers, four
   concurrent. Each makes at most three 480-second checkpointed invocations.
   Write output outside the scanned run roots. Missing files, malformed
   metadata, unprojectable nested fields, source mutation and incomplete
   traversal remain explicit blockers. Metadata enumeration does not certify
   cross-dataset media/source-group freshness or authorize a new cohort.
4. Test a separately labeled shape-matched FullKV transport on the two
   consumed Mistral FP16 inputs. A masked placeholder preserves native prefill
   shapes; it is physically removed before the real held token is processed
   exactly once. Only evaluation-time projection/query shapes change. Weights,
   dtype and the registered absolute logit tolerance remain unchanged. Record
   16 identical-prefix predictions and operation engagement. The historical
   native transport remains failed even if this new variant succeeds. This
   expensive control is not a latency result or a production default.
5. Run two consumed LLaVA-1.5 BF16 source workers. Capture one native FullKV
   trajectory, then force the same 128 prediction prefixes through all source
   candidates. Retain the unchanged 32-token guard and later-prefix diagnostics.
   Reference tokens enter only the evaluation decode boundary, never selection.
   Count codes, scales, maps, corrections, protected values, history and workspace.
   Choose the largest attainable registered representation below the ceiling;
   report utilization gaps. An infeasible direct quantizer is labeled an
   uncapped numerical diagnostic, never a byte-matched baseline.
6. Only after complete source execution and passed FullKV controls, bridge the
   exact runtime and source reader on the same BF16 LLaVA model, inputs, eager
   attention, mandatory positions and 128-prediction conditioning. The common
   ceiling is 64 MiB. Different realized sizes and history representations are
   disclosed; this is not automatically an equal-byte comparison.
7. Analyze every producer using finite `afterany` callbacks, including failures.
   Write one final hash-bound packet. Do not automatically freeze a candidate,
   materialize fresh samples, launch confirmation, or widen algorithm scope.

The Mistral numerical test and BF16 LLaVA source test are independent scopes.
A failure in one cannot be relabeled as a pass using the other. Neither a
two-input diagnostic nor complete software execution establishes superiority.

## Resources and reproducibility

GPU scopes execute sequentially: one worker, then two workers, then two workers.
Each uses one A100, 12 CPUs and at most two hours in `bii-gpu`, reservation
`bi_fox_dgx`, account `bii_dsc_community`. The aggregate authorization is six
A100s; this bounded sequence needs at most two concurrently and checks total
owned GPU occupancy before model work. Existing owned GPU work is respected.
All CPU work uses `bii`, four CPUs, 16 GiB and a 30-minute limit. No log polling
or unbounded auto-retry is used.

All paths, caches and temporary/build files remain under `/scratch/djy8hg`.
Use the existing common environment without installing packages. Dataset/model
access is offline; no new evaluation identities are selected.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_closure_sequence.py register \
  --root /scratch/djy8hg/audits/fox-review-v7-closure-20261110 --execute
```

Omit `--execute` for a write-once dry registration without scheduler submission.
Use a different output root for any new attempt; never overwrite a prior result.

## Baseline and claim boundaries

Reuse the pinned official PrefixKV PPL reproduction only in its original
checkpoint/environment scope. Its measured ROUGE-L has no preregistered
published numeric target and is not a separate numeric reproduction pass.
The missing upstream LOOK-M class still prevents claiming official execution;
local rows remain `lookm_reimpl`. A locally implemented direct 3-bit KV format
is not KIVI, and no new official quantization-baseline claim is authorized.

The V6 independent-term intervals did not establish positive individual
mechanism effects. Its pooling timing improvement was measured on consumed
inputs, not independent paper confirmation. The six-component prototype/repair
performance claim remains outside these exact/source controls. Preserve these
limitations regardless of V7 execution outcomes.
