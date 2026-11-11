# Bounded source-closure follow-up V12

This is a diagnostic follow-up, not a fresh confirmation, paper campaign, or
claim that all review requirements have passed. The validated historical
storage variant and six-mechanism research implementation remain unchanged.
EviCastKV and MosaicKV name the same project, not two competing methods.

## Evidence retained from V11

V11 is pinned to `1aab16c63eee2f063194e6db8c9b3b5af466c3a1`. Its clean CPU suite
passed 5,974 tests. Its two consumed-input K/V factorial controls establish that
the raw-prompt K/raw-prompt V path, using the common lossless history transport,
matches all 128 fixed-prefix reference predictions and logits exactly. Either
reconstructed component alone passes the registered 32-prediction guard on both
inputs; reconstructing both still fails one prediction on the MM-Vet input.
The later-prefix diagnostics are retained, including their failures. This is
causal evidence about a lossy representation, not independent answer quality.

V11 also constructs eight actual caches at 45.5 and 49 MiB initial storage.
Every initial backing-byte target matches exactly. Final storage differs by up
to several KiB because lossless generated-history costs depend on content.
Decode-owned peaks differ by approximately 8 MiB because the source reader
materializes the complete prompt, whereas the Top-K reader materializes only
retained positions. These rows must **not** be called final/peak-byte-matched
or used as evidence of a latency win. A successful initial-byte gate is not a
certificate for every memory boundary or every numerical guard.

The V11 ownership request saved model logits, but job `22796689` reached its
20-minute limit during CPU attribution/archive writing. Its gzip trace is
truncated and has no completion receipt. Preserve it as incomplete evidence.
V12 permits exactly one trace-first observer retry only after verifying that
scheduler TIMEOUT, the completed model-request artifacts, unchanged source and
input bindings, and the missing/EOF-truncated archive. A generic source/hash
error or missing CPU report does not authorize another GPU request. A complete
raw archive must instead be recovered on CPU.

## Finite follow-up lanes

| Lane | Intervention and boundary | What a pass establishes |
| --- | --- | --- |
| Eight-bit source | Encode the original normalized input with eight-bit per-channel source codes; preserve protection, positions, weights and history | Execution and the unchanged numerical guard on consumed inputs only |
| Patched LOOK-M transport | Isolated upstream import cleanup, GenerationMixin, explicit legacy tuple-cache interface | Real-checkpoint execution/parity/isolation if GPU controls pass; not a published numerical reproduction |
| Ownership recovery | Archive actual allocator events before indexed generation attribution | Recoverable measurement evidence; unknown ownership still blocks certification |
| Historical frontier | Continue exact V11 states; avoid iterating empty Parquet projections | More coverage with unchanged collected identities and preserved errors |
| Oversized records | Group registered offsets by source file; hash once before/after the complete transaction | Metadata recovered without weakening source immutability |
| YAML events | Bounded Reader/Scanner/Parser events, no YAML object constructors | Current-file metadata projection only; unbound historical versions remain a separate limitation |
| COCO media availability | ID/image-only cached Arrow projection; encoded and EXIF-canonical RGB hashes | Finite pool availability and observed aliases, not freshness by hash absence |
| Historical media | Verify recorded encoded hashes and canonical RGB for finite known media providers; reuse typed MileBench hashes | Additional alias coverage only; ID-only providers without media mappings remain unresolved |

The optional exact-BF16 normalized-source control is a distinct fidelity
diagnostic. It may only be registered with its own constructor, accounting and
strict original-K/V bitwise checks. It is not the historical three-bit variant.

No source constructor receives benchmark answers, references, or saved future
tokens. A frozen FullKV prefix is available only to the evaluation decode
transport. No model weights are updated. Every source-precision change has a
distinct implementation label; none is relabeled `mosaickv_exact` or a validated
six-component hybrid.

## Measurement and acceptance

- Use the unchanged common environment and pinned submodules. No package
  installation is authorized by the launcher.
- Keep every cache and temporary file on scratch via `cache_env.sh`.
- The GPU workers require one A100 and 12 CPUs each, at most 20 minutes.
  Aggregate A100 occupancy may not exceed six.
- Historical directory traversal is bounded to four progress-making,
  state-preserving CPU segments. YAML projection runs once on the latest
  finished states. No segment restarts the scan or polls its producer logs.
- The ownership worker uses the existing GCC 11.4.0 module if its read-only
  allocator extension needs rebuilding; the Python environment is unchanged.
- Full current-source CPU tests precede canonical diagnostics. Every producer
  receives an `afterany` completeness/error analyzer; lost scheduler
  acknowledgements do not cause duplicate submissions.
- Numerical analysis uses AVX2 CPU dispatch and the registered thread count.
  The original 32-token KL/logit/token thresholds are unchanged. Full-128
  fixed-prefix diagnostics are also stored; they are not free-generation
  sequence agreement.
- Count source codes, floating headers, phase/maps, exact protection, patches,
  history and workspace. Initial, final, decode-owned peak, whole-request
  requested-byte envelopes, and reserved memory are separate quantities.
- An ownership certificate names its exact PyTorch source-KV-plus-workspace
  boundary. It is not automatically KV-only, whole-device HBM, or a reduction
  relative to a separately measured baseline.
- A finite COCO identity certificate concerns canonical image IDs, encoded
  files and EXIF-canonical RGB within its declared, completed historical scope.
  It must exclude observed aliases and pool duplicates. It does not promise
  independence of all possible crops, resized scenes or source families.
- Fresh confirmation, qualified baseline reproduction, strict latency
  superiority, and six-component benefits need their own empirical evidence.
  No positive diagnostic automatically authorizes a paper-scale sweep.

## Canonical entry point

From a clean immutable source, with outputs outside the repository:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_review_v12.py register \
  --root /scratch/djy8hg/workdir/fox-review-v12/run-NEW --execute
```

The registration binds that source SHA, its environment lock and parent
artifacts. An existing output root cannot be reused. Reports remain explicitly
negative or incomplete when a gate fails; no result table may invent a pass.
