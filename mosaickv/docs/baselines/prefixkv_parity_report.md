# PrefixKV official parity report

## Status

A controlled one-sample LLaVA-1.5-7B official-versus-reimplementation run
completed on 2026-07-20. The strict comparator returned `status: comparable`
with no control mismatches. This closes the implementation parity gate, but it
is a **non-canonical development diagnostic**, not a paper result: the MosaicKV
worktree was dirty and each latency has only one measured trial.

Measured run: Slurm job `17134561`, exit `0:0`, elapsed `00:00:58`.

Artifacts are outside the repository at:

```text
/scratch/djy8hg/runs/mosaickv/prefixkv/official-parity/
  prefixkv-official-parity-17134561/
    official.json
    reimplementation.json
    comparison.json
    manifest.json
    profile.json
    environment.lock.txt
```

The comparison, official, reimplementation, and manifest file SHA-256 values
are respectively:

```text
e569e4288155e079256ddd405e7c8ce71050f3fc665a9aaae4afa2c1dc8188e6
e1dc2a1b412e3fa80f3537713cbb6c74eab51db4a7fe75a275cd36d8ab69243e
6903ce880a3ec24e1c5e24f88885f273f632ffb2cd64f9cc42dbcf62c3b5d50f
71bc29f28b7cdd3aa2c1cfab945a9ff42242704f38f5a5e906653925971a4053
```

## Controlled inputs

- Official code: unmodified `third_party/PrefixKV` at
  `597f1ab032704951550f93bcc8a23f1454b80aa4`, MIT license.
- Model and tokenizer: legacy `Zuyan/ElasticCache/llava-v1.5-7b` snapshot
  `833edbdc7512240f2a3aa49feeb9468e2297bdbc`, BF16 language model, the
  upstream eager attention patch, and the same loaded model instance for both
  methods.
- Dataset: LLaVA-Description file SHA-256
  `407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a`.
- Calibration: sample `000000539056`; evaluation: sample `000000442786`;
  their recorded intersection is empty. Native profile digest:
  `7ad45343a55f98cd96fee30d488aa6ef485f7a0751095552668adc67e46b7aab`.
- Budget: retained ratio `0.5`, token-sized blocks, first and last positions
  protected, fixed decode distance `-25`.
- Generation: greedy, temperature `0`, one beam, EOS stopping disabled, exactly
  16 output tokens.
- PPL: teacher forcing over the first 16 tokens of the human answer. ROUGE-L:
  the generated answer against the same 16-token FullKV answer.
- Hardware: one NVIDIA A100-SXM4-80GB, driver `595.71.05`, CUDA runtime `12.1`.
- Shared parity environment: Python `3.11.15`, torch `2.1.2+cu121`, Transformers
  `4.31.0`, tokenizers `0.13.3`, accelerate `0.21.0`, NumPy `1.26.4`.
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set and the completed log contains no
  deterministic-algorithm warning.

The complete control hashes are in the raw artifacts. The config hash is
`6693270fd17fb1eba7e64498ae4709a680550b89261fb94b6c9c70627f9fd7f6`.

## Measured comparison

These values are copied from the immutable raw artifacts above; none is
synthetic.

| Metric | Official PrefixKV | `prefixkv_reimpl` | Difference, reimpl - official |
|---|---:|---:|---:|
| Sum of retained positions over 32 layers | 10,015 | 10,016 | +1 |
| Initial active KV bytes | 164,085,760 | 164,102,144 | +16,384 |
| 16-token teacher-forced PPL | 3.161397933959961 | 3.144782066345215 | -0.016615867614746094 |
| ROUGE-L F1 versus FullKV answer | 0.9090909040909091 | 0.9090909040909091 | 0 |
| Fixed-length generated-token agreement | 1.0 | 1.0 | 0 |
| End-to-end latency, seconds | 1.1662340087890626 | 16.21851953125 | +15.052285522460936 |

Both methods generated the identical token IDs:

```text
[450, 9088, 5680, 263, 767, 13407, 297, 263,
 2085, 1017, 538, 29892, 13587, 263, 1472, 413]
```

The official and reimplementation layer sizes were identical except for
zero-based layer 21:

```text
official: [521,462,423,284,279,208,337,353,354,387,407,436,377,348,328,364,
           323,264,244,212,292,168,194,236,192,259,173,170,319,267,386,448]
reimpl:   [521,462,423,284,279,208,337,353,354,387,407,436,377,348,328,364,
           323,264,244,212,292,169,194,236,192,259,173,170,319,267,386,448]
```

The selected position sets match exactly in 31 of 32 layers. At layer 21,
`prefixkv_reimpl` contains every position selected by official PrefixKV plus
physical position 319. Thus the only remaining selection difference is the
single position required to satisfy the exact global budget.

Each source layer has 626 positions, so the exact global target is
`32 * 626 * 0.5 = 10,016`. `prefixkv_reimpl` meets it exactly. Upstream converts
each layer's floating forget ratio independently with `round(...).astype`,
which retains 10,015 positions for this generated profile. The 16,384-byte
delta is exactly one BF16 K/V position across 32 heads of width 128. This is an
intentional documented reimplementation decision, not an undiagnosed mismatch.

The latency row includes `prefixkv_reimpl`'s Python construction of the common
`FullKVState` and its token-sized block descriptors. It is a one-trial
development observation and must not be used as a systems-performance claim.
The official wrapper performs its gather directly. A paper latency comparison
still requires warmups, repeated trials, and equivalent optimized
instrumentation boundaries.

## Environment deviation

The upstream README requests Python 3.8 and tokenizers 0.13.1. MosaicKV itself
requires Python 3.11, and tokenizers 0.13.1 has no Python 3.11 wheel. Therefore
the controlled comparison used Python 3.11 and tokenizers 0.13.3 while keeping
the upstream torch and Transformers versions. Crucially, both sides used this
same environment, checkpoint, inputs, profile, precision, and model instance.

An additional isolated Python 3.8.20 environment at
`/scratch/djy8hg/env/mosaickv_prefixkv_official` passes imports for the pinned
upstream versions, including tokenizers 0.13.1. Its freeze is preserved at
`/scratch/djy8hg/runs/mosaickv/prefixkv/official-env-freeze.txt`. It cannot
import the Python-3.11 MosaicKV package, so it is an upstream compatibility
smoke rather than the controlled comparator environment.

## Reproduction

After materializing the exact checkpoint, data, and two media files outside
the home directory, run:

```bash
sbatch mosaickv/slurm/prefixkv_official_parity.sbatch
```

The job invokes `mosaickv/scripts/run_prefixkv_llava_parity.py`. The runner
imports the official `PrefixKV` class and official eager-attention patch
without modifying the submodule. It adapts the common `prefixkv_reimpl` plan to
the same upstream generation callback so every comparison control is shared.

This measured diagnostic satisfies the official LLaVA parity acceptance gate.
It does not satisfy the clean-worktree or repeated-systems-measurement gates
for an AAAI paper table.
