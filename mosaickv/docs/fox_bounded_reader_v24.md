# V24 native batch-dispatch repair

V23's head-group CUDA operator test is negative and remains immutable. N600,
N604 and N731 show a few different BF16 words, despite unchanged complete
token reductions. No pretrained V23 workers were admitted. CPU tests alone
did not establish GPU arithmetic parity.

The subsequent synthetic diagnostic at source
`29805aa4e4f610368d6e1f4165a45e6920a3dc85` measured54 cases (two fixed seeds,
27 sequence lengths). Native groups2/4/8/16 failed; both zero-stride native
batch32 broadcasts passed every score/output word. These are measured tensor
operator results, not benchmark quality or a model-level128-prediction pass.

V24 therefore reconstructs one head and broadcasts its K/V views into native
batch32 BMM without allocating32 copies. It retains the full token axis and
native softmax/rounding. Only one head's result is used. This deliberately
does redundant arithmetic, and no latency win is presumed. The fixed common
KV workspace is now two1024-by128 BF16 component buffers:524,288 bytes.
Attention scores/probabilities/output and opaque vendor workspace remain
separately classified; no source reconstruction buffer is hidden there.

The original-source format, per-head original-KV bit patches, functional raw
head caching, actual-byte intersection, raw-history growth, original logical
positions, three-arm verification and all tolerances are unchanged from
`fox_bounded_reader_v23.md`. The old49MiB lossless-source claim remains
unsupported. The new attainable diagnostic budget is explicit and cannot be
represented as a repair at that old cap. Whole-request peak and paper-quality
certification remain open even if constructed decode-owned boundaries match.

Run only the two consumed Description/MM-Vet inputs after complete CPU and
fresh CUDA operator checks pass. Compare128 original-source predictions to
original FullKV, raw selected predictions to the SAME-selected native cache,
and retention1 raw to original FullKV; require128/128 greedy equality and
maximum logit difference<=0.001. Preserve the registered32-token guards.
Automatic finite afterany analysis must recompute gates from saved logits and
exact constructed initial/final/observed decode-owned peak bytes. No fresh
cohort, calibration, six-mechanism claim, or paper campaign is authorized.

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_bounded_reader_v24.py launch \
  --root /scratch/djy8hg/workdir/fox-review-v24/run-20261111-1
```
