# V28: bounded 128-prediction COCO boundary control

This is a correctness diagnostic on the **same two consumed inputs**, not a
new cohort, a quality comparison, an online algorithm change or a timing study.
It may execute only after the whole-request published-KV and byte-match gate
passes. A prerequisite is either a same-clean-source GPU execution, or the
explicitly pinned GPU producer `422a67994d06ffb0b69943727063a7dfc885b6d2`
revalidated by the current clean analyzer. The latter remains labeled
`derived_validation_not_new_GPU_execution`: its execution SHA is **not**
rewritten to the analyzer SHA. The actual run root, immutable plan, original
gate, all input/output bindings, and independent replay provenance must verify.
No other producer or a bare successful boolean is accepted as a bridge.
The pinned actual producer is the bounded whole-reader V28 run at
`/scratch/djy8hg/workdir/fox-review-v33/run-20261111-1`, with immutable plan
`3bbd8e053073bf6c878e1618923c45f02ea2f5325b04a582d1ee648e2aa28745`
and original `analysis/GATE_bounded_reader_v28.json`. Earlier timed-out
instrumentation runs are not reused as successful prerequisites. V33 retains
the equivalent grouped CPU rankings and uses one exact-cover raw allocation
per layer instead of per-head allocations. Selected words, positions and byte
targets are unchanged; every executed owner observation and all scientific
gates remain. The V33 gate must actually pass before this continuation runs;
these bindings alone do not assert a successful result. Negative V31/V32
artifacts are preserved and cannot act as positive prerequisites.
Before admission, Git must prove that every file tracked by that producer is
byte-identical in the new source: only additive CPU/tooling files are allowed.
The numerical transport also retains its exact passed V9 implementation hash.

The native packed-subset COCO gate already passed 128 **checks across 32
inputs**. The separate corrected V9 boundary gate passed 16 predictions on
each of two inputs (32 total). Neither was a 128-prediction boundary run.
The V28 prerequisite recovers those exact sealed artifacts and preserves the
historical negative native chunking result.

## Frozen intervention

- Model: `llava-hf/llava-v1.6-mistral-7b-hf` at
  `2424fdd47412fccc66d91719126b420e9fbd7065`, FP16, eager, original weights,
  RMSNorm epsilon and native arithmetic flags.
- Inputs: `COCO_val2014_000000260440.jpg` and
  `COCO_val2014_000000512467.jpg`; exact original prompt/media hashes.
- A: ordinary complete FullKV prompt prefill.
- B: V9's masked-dummy full-row prefix, remove precisely the dummy cache slot,
  then process the held original final prompt token exactly once.
- Only that held-token forward uses the **unchanged V9** shape transport:
  224 layer Linear calls, 32 attention calls and 65 language RMSNorm calls.
- Restore every method before 127 ordinary native continuation forwards per
  path. Both paths receive the same FullKV-generated prefix.
- Compare all 128 predictions, all logits at the unchanged `1e-4` absolute
  tolerance, original logical positions, and bitwise initial/final caches.
  Per-layer K/V word hashes are saved; CPU analysis independently recomputes
  trajectory comparisons and validates cache-hash consistency.

The parent benchmark's 64-token setting is disclosed; V28 explicitly extends
the **correctness diagnostic** to 128 predictions. It does not rescore the
benchmark or silently replace historical generation controls. It does not
validate BF16 source construction on the FP16/GQA checkpoint.

## Execution

After a clean scoped commit and a positive whole-request gate:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_coco_boundary_v28.py \
  launch --root NEW_IMMUTABLE_SCRATCH_ROOT \
  --whole-gate PATH_TO_PASSED_WHOLE_REQUEST_GATE
```

The launcher admits two one-A100/12-CPU/15-minute workers, with aggregate own
GPU allocation bounded by six, followed by a finite `afterany` CPU analyzer.
Submission acknowledgements and artifacts are write-once; unknown submissions
are never blindly retried. No continuous log polling or automatic fresh cohort
is enabled. Negative/incomplete workers remain in the gate.

Outputs: immutable plan/bindings, per-worker manifests and raw FP32 trajectories,
initial/final per-layer cache hashes, execution-scope counts, structured error
records, and `analysis/GATE_coco_boundary_v28.json`.
