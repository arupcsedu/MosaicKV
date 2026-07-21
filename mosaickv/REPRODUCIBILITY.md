# Reproducibility Specification

## Reproduction target

A MosaicKV result is reproducible only when a clean checkout can resolve the same source, configuration, model, dataset, environment, inputs, and measurement procedure and can regenerate the reported artifact within its registered tolerance. A command without immutable revisions is not a reproduction recipe.

This document defines the minimum record. Backend- or benchmark-specific protocols may add fields but may not remove them.

## Immutable run manifest

Every experimental run must emit a machine-readable manifest. Every reported row must point to one or more manifests and record all fields below, either directly or by immutable reference.

```yaml
schema_version: 1
run_id: <globally unique immutable ID>
measurement_type: <controlled value from SCIENTIFIC_INTEGRITY.md>
started_at_utc: <ISO-8601 timestamp>

source:
  git_sha: <40-character commit SHA>
  git_dirty: false
  patch_sha: not_applicable
  config_sha: <SHA-256 of canonical resolved configuration>

model:
  id: <registry or local model ID>
  revision: <immutable commit/revision>
  precision: <fp32|fp16|bf16|other>

dataset:
  id: <dataset ID>
  revision: <immutable revision or checksum>
  split: <split name>

software:
  cuda: <version or not_used>
  driver: <version or not_used>
  pytorch: <version>
  transformers: <version or not_used>
  vllm: <version/commit or not_used>
  sglang: <version/commit or not_used>

environment:
  name: common
  lock_path: mosaickv/env/common/requirements.lock
  lock_sha256: <SHA-256 of exact common lock>
  cache_root: </scratch path outside home>

hardware:
  gpu_type: <exact accelerator model or not_used>
  gpu_count: <integer>

execution:
  backend: <huggingface|vllm|sglang|other>
  attention_implementation: <eager|sdpa|flash_attention_2|other>
  seed: <integer>
  deterministic_algorithms: <true|false>

inputs:
  prompt_set_sha: <SHA-256 or immutable manifest ID>
  media_set_sha: <SHA-256 or immutable manifest ID>
  preprocessing_sha: <SHA-256 of canonical preprocessing specification>
  tokenization_sha: <SHA-256 of token IDs and tokenizer configuration>

generation:
  parameters_sha: <SHA-256 of canonical resolved generation parameters>
  output_length_policy: <canonical specification>

cache:
  budget_value: <number>
  budget_unit: <blocks|retained_slots|bytes|other>
  retention_ratio: <number or not_applicable>
  accounting_spec_sha: <SHA-256 of accounting specification>

artifacts:
  raw_output_sha: <content hash>
  metrics_sha: <content hash or not_applicable>
  log_sha: <content hash>
```

The required version fields are present even when a framework is not used. Store `not_used`; do not omit the key. If a dependency reports both a package version and build or commit identifier, preserve both.

Canonicalize the fully resolved configuration with stable key ordering, normalized scalar representations, and no machine-specific output paths before computing `config_sha`. Preserve the canonical file alongside the run. A run from a dirty worktree is exploratory unless the exact patch is saved and hashed; canonical paper results must use a clean commit.

## Controlled-comparison record

Rows may be grouped as a direct comparison only if the following manifest values match exactly:

- prompt and media set hashes;
- preprocessing and tokenization hashes;
- model ID, model revision, and precision;
- resolved generation-parameter hash;
- output-length policy;
- cache budget value, unit, and accounting-spec hash;
- backend and attention implementation; and
- relevant hardware and execution topology.

For paired task-quality analyses, store an immutable example ID and output for every item so that methods are aligned on the same examples. For stochastic decoding, reuse the same declared seed schedule and sampling implementation. If output length is fixed, enforce the same number of generated tokens; if stopping is semantic, share the same stopping criteria and report realized lengths.

The comparison validator must run before metric aggregation and must fail closed on missing or mismatched values.

## Retention ratio 1.0 equivalence contract

For each supported combination of model revision, model precision, backend, attention implementation, device type, and device count, compare MosaicKV at retention ratio `1.0` with the full-cache reference under identical inputs and execution settings.

The default acceptance contract is:

1. input token IDs, modality positions, attention masks, position IDs, and output lengths are exactly equal;
2. greedy decoding produces exactly equal generated token IDs for every test example;
3. no cache entry is evicted, prototyped, quantized, or approximated on the `1.0` path;
4. shapes and dtypes of compared cache tensors and logits are exactly equal; and
5. finite numeric values satisfy `abs(candidate - reference) <= atol + rtol * abs(reference)` elementwise, using the preregistered tolerance below.

| Precision | `atol` | `rtol` |
|---|---:|---:|
| FP32 | `1e-6` | `1e-5` |
| FP16 | `1e-3` | `1e-2` |
| BF16 | `1e-2` | `1e-2` |

These are maximum default tolerances, not targets. Use exact equality or stricter tolerances when the execution path permits it. Any backend-specific tolerance must be documented before evaluation, justified by a repeatability study of the full-cache reference, and may not exceed the applicable default above. NaNs or infinities fail unless the fixture explicitly expects the same value at the same location and documents why.

Record maximum absolute error, maximum relative error over nonzero reference values, mismatch count, compared element count, and first mismatch location. Run the gate on short deterministic fixtures in continuous testing and on a preregistered evaluation slice before accepting research results.

Failure of token equality or numerical tolerance blocks claims for that configuration. Fix the cause and rerun affected experiments; do not waive the gate or broaden tolerance after inspecting MosaicKV outcomes.

## Dataset, media, and prompt provenance

- Pin hosted datasets to immutable revisions. For local or generated indexes, record content hashes and the script/config that created them.
- Preserve stable example IDs, original split names, and filtering decisions.
- Hash raw media when licensing permits; otherwise hash a manifest of provider IDs and revisions. Record video decoder, frame sampling, resize, crop, normalization, and image ordering.
- Store prompts as versioned data, including system messages, chat templates, role separators, and whitespace-sensitive content.
- Record tokenizer files/revision, special-token mappings, padding/truncation side, maximum lengths, and final token IDs.
- Do not redistribute restricted datasets or media; provide acquisition and verification instructions.

## Environment and backend capture

Record the complete software fields in the manifest plus operating system/container identifier, Python version, compiler versions used for native extensions, relevant kernel/library builds, and backend flags. For GPU runs, retain CUDA runtime and toolkit versions where distinct, driver version, GPU model and count, topology when relevant, and distributed-launch configuration.

Backend configuration includes scheduler and batching settings, KV-cache dtype, tensor/pipeline parallelism, quantization, CUDA graph use, compilation settings, attention kernel, and environment variables that affect execution. Store `pip freeze`, `conda` export, lockfile, or container digest as a supplementary artifact; it does not replace the explicit manifest fields.

## Quality measurement protocol

Before running the final benchmark, version and pin:

- benchmark versions and evaluation scripts;
- task metrics and parsing rules;
- retention ratios and exact budget accounting;
- method, ablation, and baseline configurations;
- seed schedule and sample count;
- exclusion, retry, and failure rules; and
- aggregation, uncertainty, and multiple-comparison procedures where applicable.

Save per-example predictions and metric inputs. Generate summaries by script from immutable raw artifacts. Each aggregate must list constituent run IDs and distinguish measured observations from derived statistics.

## Systems measurement protocol

Version the warmup count, measured repetitions, synchronization points, batch/concurrency schedule, prompt lengths, media sizes or frame counts, generation lengths, and whether preprocessing and transfer time are included. At minimum, define the measurement boundaries for prefill latency, time to first token, decode/inter-token latency, end-to-end latency, throughput, peak device memory, and realized cache bytes when those metrics are reported.

Keep quality and systems configurations paired. Measure on otherwise idle, identified hardware; record power/clock policy when controlled. Report the distribution or uncertainty over repetitions, not only the best observation. Preserve raw timings and profiler traces used for breakdown claims.

## Baseline reproduction

Official external source belongs under `third_party/<baseline>/` at a pinned commit SHA with license and attribution intact. Record upstream URL, paper citation, commit, installation steps, local patch hash, model/backend compatibility, and the exact invoked entry point.

Local paper-faithful implementations must be named `<method>_reimpl`. Their manifests use `baseline_reimpl_measured`, and their documentation must state that they are not official. Preserve a decision log for paper ambiguities and tests used to validate the interpretation.

Offline baseline calibration is separate from evaluation. A PrefixKV profile
must record model/dataset revisions, calibration split, seed, sorted sample
IDs, and a profile digest. Native profile generation and evaluation must abort
on any sample-ID intersection. An upstream profile without calibration IDs may
be used for source inspection or a clearly marked parity diagnostic, but not a
paper result claiming verified disjointness.

The controlled PrefixKV LLaVA parity runner records the native profile and
the raw list consumed by official code, both sample-set digests, the empty
calibration/evaluation intersection, the complete parity-environment freeze,
and synchronized CUDA-event timing. Its Slurm entry point sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. A comparable dirty-worktree diagnostic is
useful implementation evidence, but it remains ineligible for a paper table
until repeated from a clean SHA with the systems-measurement protocol above.

## Artifact lineage and retention

Use content-addressed or otherwise immutable run directories. A recommended lineage is:

```text
resolved config -> run manifest -> raw per-example outputs/timings
                -> validated metrics -> aggregate table/figure
```

Every arrow must be implemented by a versioned command or script. Tables and figures must not be edited by hand. Retain logs for failures as well as successes, mark superseded runs without deleting their lineage, and verify hashes when moving artifacts.

## Minimum reproduction checklist

- [ ] Clean, pinned git SHA and canonical config SHA are available.
- [ ] Model and dataset IDs use immutable revisions.
- [ ] All required CUDA, driver, framework, GPU, backend, attention, seed, and measurement-type fields are present.
- [ ] Prompt, media, preprocessing, tokenization, generation, output-length, and cache-budget identities pass comparison validation.
- [ ] Retention ratio `1.0` passes token and numerical equivalence.
- [ ] Raw artifacts are immutable and content-hashed.
- [ ] Aggregates are generated by a versioned script from listed run IDs.
- [ ] External baselines are pinned and attributed; reimplementations use `*_reimpl` and are not called official.
- [ ] Measured-results tables contain no synthetic numbers.
