# Bounded PrefixKV lower-memory development lane

This finite follow-up runs concurrently with the Prof. Fox COCO study; it does
not retune that study or overwrite the completed PrefixKV campaign. Both use a
static three-A100 allocation ceiling, totaling at most six, with one A100 and
12 CPUs per worker and two-hour GPU limits. Actual allocation depends on Slurm.
Arrays are recorded with the tested global scheduler in dry-run mode: applying
its generic six-slot throttle would defeat the stricter three-slot partition.

The unchanged exact-selection candidate and PrefixKV allocation implementation
are compared on eight LLaVA-Description and eight paper-style MM-Vet inputs.
ID ordering uses only a seeded hash and excludes every identity in explicitly
bound bounded-history/proposer development plans. These datasets have already
been evaluated historically: the new cohort is **retrospective development**,
not untouched confirmation. No claim of globally fresh media is made.

The model is the pinned HF LLaVA-1.5-7B FP16 checkpoint, eager attention, greedy
fixed-length 1,024-token generation, block size one, first/last mandatory tokens,
and append-only generated history. Every compressed method (Prompt Top-K,
`prefixkv_reimpl`, `mosaickv_exact`) runs at **both** 80 and 160 MiB initial
physical active-KV caps. Baselines and candidate retain exact source K/V; no
prototypes, quantization, residual storage, or repair are enabled.

The PrefixKV row is the explicitly labeled, byte-rescaled offline-profile
**selection-only append-only reimplementation adaptation**, not official
paper-native decoding. Its unchanged source is hash-compared with the historical
four-input official-path parity implementation; that result is preserved,
not reinterpreted as validation of a different transport or new operating point.
Each worker additionally runs current-source FullKV generation/reinjection
parity, and compressed workers run full runtime retention-one parity (16 tokens,
exact token agreement, maximum logit difference <= 0.001, preregistered here).

FullKV is regenerated once per task in matching FP16 and is only a quality
reference. Its generated output artifact is frozen and hashed for ROUGE-L.
PPL uses complete human annotation targets, not generated references, and is
reduced as exp(total NLL / total scored tokens). PPL and ROUGE-L remain separate.
Both metric phases must have complete physical-storage traces; equal initial,
final, and peak storage is required across compressed methods at each common
cap. Append-only final/peak storage must not be confused with the initial cap.

The prospective lower-memory hypothesis is MosaicKV **80 MiB versus PrefixKV
160 MiB**. It is expressly **not byte-matched**. The two-budget curves include
Top-K, so a named PrefixKV result cannot silently become a claim against the
nearest quality-eligible competitor. Fifty-percent smaller *initial* storage
would not imply fifty-percent smaller final storage: 1,023 cache-appending decode
steps grow LLaVA's cache substantially. All reductions use measured boundaries.

Five full-request warmups precede each quality arm. After complete metrics and
both zero-margin paired quality gates pass (PPL upper CI <= 0; ROUGE-L lower CI
>= 0, Bonferroni family four across two tasks/two metrics, 10,000 paired bootstrap
replicates), a finite timing continuation is automatically submitted. It uses
the first two preregistered IDs per task, five warmups per method, twenty paired
repetitions, seeded randomized method order, and unchanged generation/settings.
It tests the latency ratio upper bound <= 1.0 with paired input/repetition
hierarchical bootstrap and family-two adjustment. This is fixed-input diagnostic
timing, not a population-wide serving-latency claim. Negative or incomplete
quality gates stop the chain without a larger cohort or threshold relaxation.

Execution is fail-closed: fresh FullKV references finish before compressed
workers are submitted. `afterany` CPU callbacks analyze missing/failed producers
as well as successes; no permanently orphaned `afterok` array or log-polling
daemon is created. Failed sample IDs remain in the declared denominator. No
large paper campaign or independent confirmation is automatically authorized.

Commands (from the committed clean worktree, with all caches on scratch):

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prefixkv_isoquality_development.py prepare --root /scratch/djy8hg/runs/mosaickv/prefix-isoquality-development-v1
# Use the printed immutable plan SHA:
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prefixkv_isoquality_development.py start --root /scratch/djy8hg/runs/mosaickv/prefix-isoquality-development-v1 --plan-sha PLAN_SHA
```
