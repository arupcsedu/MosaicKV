# Repository and environment audit

Audit date: 2026-07-19 (America/New_York). This is a read-only snapshot. No
package was installed, removed, or upgraded, no model weights were downloaded,
and neither AAFLOW nor its environments were modified.

Evidence was collected with `git status`/`git rev-parse`, direct interpreter
paths plus `importlib.metadata`, isolated imports under
`PYTHONNOUSERSITE=1`, `nvcc --version`, `nvidia-smi`, Slurm `sinfo`, local HF
cache enumeration, and direct inspection of installed `.py` and `.dist-info`
files. Failed imports and absent files are reported as failures/unsupported,
not inferred support.

## Executive finding

The MosaicKV repository is currently a documentation-only skeleton. The most
usable starting point is the functioning Python 3.11 Conda environment
`drc_rag_benchmarks_yml_20260421`, run with `PYTHONNOUSERSITE=1`, together with
the Hugging Face reference harness patterns in AAFLOW. It has PyTorch and
Transformers, but it has no usable FlashAttention, vLLM, SGLang, or lmms-eval.
The separate environment containing vLLM 0.11.2 and SGLang 0.5.10.post1 cannot
start Python. Consequently, none of the five requested checkpoints is currently
verified loadable end to end, and neither serving backend is currently
executable. Architecture support found in source is recorded separately in the
[model matrix](model_capability_matrix.md); it is not load evidence.

## Repository state

MosaicKV is at commit `c4a41fbfb80d9c3e1ea2aa4e4049387535d69e86` on
`main` with remote `https://github.com/arupcsedu/MosaicKV.git`. At audit time,
`AGENTS.md` and `mosaickv/` were untracked governance work; they were preserved.

```text
MosaicKV/
├── AGENTS.md
├── LICENSE
├── README.md
└── mosaickv/
    ├── PLAN.md
    ├── README.md
    ├── REPRODUCIBILITY.md
    ├── SCIENTIFIC_INTEGRITY.md
    └── docs/                 # this audit
```

There is no Python package, test suite, model adapter, backend adapter,
third-party checkout, configuration tree, or experiment runner yet. This audit
does not implement MosaicKV.

## AAFLOW and AAFLOW+ reuse assessment

The adjacent repository `/scratch/djy8hg/workdir/AAFLOW` is on `aaflow-dev` at
`52671a14365b3e767b549664f6e4e345a4a133ff`. It has pre-existing tracked and
untracked changes, so it must remain a read-only source of patterns unless its
owner explicitly authorizes integration. Exact reusable components are:

| Component | Exact source | Reuse decision |
|---|---|---|
| Typed multimodal records and validation | `multimodal_node.py:34` (`Modality`), `:43` (`MultiModalRecord`) | Reuse the schema concepts for prompt/media manifests and provenance. The file was untracked in AAFLOW at audit time, so copy only after provenance is resolved. |
| Arrow serialization | `multimodal_node.py:282` (`create_arrow_schema`), `:310` (`record_to_arrow_table`), `:344` (`arrow_table_to_records`) | Reuse for columnar experiment inputs and results after adding MosaicKV-required SHA/version fields. |
| Multimodal test doubles | `multimodal_embedder.py:25` and `:35`; `multimodal_vectorstore.py:21` and `:27` | Reuse only in unit tests. The embedder explicitly uses deterministic byte transformations, not learned multimodal representations; it must never produce scientific quality results. |
| Benchmark configuration and metrics | `framework_rag_pipeline_benchmark/common.py:18` (`BenchmarkConfig`) and `:58` (`PipelineMetrics`) | Reuse the dataclass/serialization pattern, extended to the mandatory MosaicKV metadata schema. |
| Real Transformers helpers | `framework_rag_pipeline_benchmark/common.py:158` (`TransformersEmbedder`) and `:201` (`TransformersGenerator`) | The batching pattern is reusable. These are text-only helpers and are not a VLM full-cache reference. |
| Local-only HF engine | `framework_rag_pipeline_benchmark/distributed_hf_framework_benchmark.py:228` (`HuggingFaceEngine`) | Reuse local-cache resolution, explicit dtype/attention implementation, inference mode, tokenization, and load-time measurement patterns. Replace `AutoModelForCausalLM` with model-specific VLM loading and add full-cache/correctness controls. |
| AAFLOW overlap runner | `framework_rag_pipeline_benchmark/runners.py:529` (`AAFLOWRunner`) | Reuse bounded queues, independent worker pools, batching, coalescing, and overlap structure for offline stages. Do not reuse timing results. |
| AAFLOW+ Arrow runner | `framework_rag_pipeline_benchmark/runners.py:847` (`AAFLOWPlusRunner`) | Reuse Arrow batch boundaries and streaming embed/upsert organization for artifact plumbing, not for KV-cache algorithms. |
| Agentic benchmark engines | `higress_agentic_benchmark/engines.py:294` (`AAFLOWEngine`) and `:418` (`AAFLOWPlusEngine`) | Reuse only orchestration and record-emission patterns if an agentic workload is added. |

AAFLOW's `environment.benchmarks.yml` specifies Python 3.11 and broad ranges
(`torch>=2.4,<3`, `transformers>=4.45,<5`, `accelerate>=1,<2`,
`datasets>=3,<5`). It is useful as intent but is not an exact MosaicKV lockfile.
The installed environment has already drifted beyond at least its Transformers
upper bound in one prefix.

## Python and environment management

The login shell on `udc-ba38-32c0` has no `python`, `conda`, `mamba`,
`micromamba`, or `uv` command on `PATH`. Environment contents show Conda
(`conda-meta/history`) and `venv` (`pyvenv.cfg`) were used. Direct absolute
interpreter paths are therefore required for reproducible inspection.

### Environment inventory

Versions below come from package metadata with user site packages disabled.
“Broken” means the interpreter fails before `site` initialization, not merely
that an optional import is missing.

| Prefix under `/scratch/djy8hg/env/` | Python | Target packages | State |
|---|---:|---|---|
| environment root (`bin/python`) | 3.6.8 | none of the requested packages | Legacy venv linked to RHEL platform Python; too old for MosaicKV |
| `aaflow_test_env_venv` | 3.11.15 | none of the requested packages | Functional, empty test venv |
| `drc_rag_bench_env` | cannot start (3.11.14 metadata) | torch 2.9.1; transformers 5.3.0; vllm 0.11.2; sglang 0.5.10.post1; datasets 4.8.5; accelerate/flash-attn/lmms-eval absent | **Broken:** missing Python standard-library encodings. Source files remain inspectable. |
| `drc_rag_benchmarks_yml_20260421` | 3.11.15 | torch 2.11.0 (`2.11.0+cu130` at import); transformers 4.57.6; accelerate 1.13.0; datasets 4.8.4; flash-attn/vllm/sglang/lmms-eval absent | **Functional reference candidate** with user site disabled |
| `drc_rag_benchmarks_flashattn` | 3.11.15 | inherits the preceding row; flash-attn 2.8.3.post1 in the venv; vllm/sglang/lmms-eval absent | Python works, but `import flash_attn` is broken |
| `saa_sglang_env` | 3.11.15 | none of the requested packages | Functional, empty venv |
| `saa_vllm_env` | cannot start | torch 2.9.0; transformers 4.57.6; accelerate 1.13.0; vllm 0.11.2 | **Broken:** its venv base is `drc_rag_bench_env` |
| `univid-env` | 3.11.15 | torch 2.12.0; transformers 4.51.3; accelerate 1.5.1; other requested packages absent | Functional but unrelated and not complete |
| `deep_rc_rag_env`, `gcylon_env`, `pytrade` | cannot start | not audited as package sources | Stale interpreter symlinks whose targets do not exist |

The required installed-package answer for the recommended reference prefix is
therefore: torch 2.11.0, transformers 4.57.6, accelerate 1.13.0,
flash-attn 2.8.3.post1 **present but unusable only in the child venv**, vLLM not
installed, SGLang not installed, lmms-eval not installed, and datasets 4.8.4.
The backend-source prefix contains vLLM 0.11.2 and SGLang 0.5.10.post1 but is
not a runnable environment.

Two contamination/failure details are material:

- Without `PYTHONNOUSERSITE=1`, `datasets` resolves to user-site version 4.1.1
  at `/home/djy8hg/.local/lib/python3.11/site-packages`, and importing the VLM
  classes fails through user-site scikit-learn because `threadpoolctl` is
  missing. With user site disabled, imports of
  `LlavaForConditionalGeneration`,
  `Qwen2_5_VLForConditionalGeneration`, and
  `LlavaOnevisionForConditionalGeneration` succeed.
- `flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so` fails to load because
  `/lib64/libstdc++.so.6` does not provide `CXXABI_1.3.15`. Package presence is
  not backend availability.

## CUDA and GPU snapshot

| Item | Observed value |
|---|---|
| Host/kernel | `udc-ba38-32c0`; Linux `4.18.0-553.124.1.el8_10.x86_64` |
| CUDA symlink | `/usr/local/cuda -> /usr/local/cuda-13.2` |
| CUDA compiler/toolkit | `nvcc` release 13.2, build `V13.2.51` |
| PyTorch build runtime | CUDA 13.0 (`torch.version.cuda`) |
| Login-node driver | unavailable: `nvidia-smi` cannot communicate with a driver |
| Login-node GPUs | none visible; `torch.cuda.is_available() == False`, device count 0 |

The CUDA 13.2 toolkit is not evidence of the compute-node driver version. That
must be captured from `nvidia-smi` inside every Slurm job. Likewise, CUDA 13.0
is the PyTorch wheel's build runtime; no CUDA runtime/driver handshake was
exercised on the login node.

Slurm advertised the following GPU types at audit time: A100 40 GB, A100 80
GB, A40, A6000, V100, B200, RTX Pro 6000 (including MIG), RTX 3090, RTX 2080,
and reserved H200 nodes. Allocation is dynamic. The general `gpu` partition
advertised A100 80 GB nodes such as `udc-an34-*`, `udc-an36-*`, and
`udc-an37-*`; these are the safest minimal target because the requested 7B VLMs
and modern attention kernels fit the Ampere feature set. This is a scheduling
recommendation, not a claim that a GPU was allocated during this audit.

## Current model-load readiness

The five repositories are public and their architecture/configuration source
is identifiable, but no complete requested checkpoint is cached locally. The
only matching cache entry is a 1.5 KB reference file for
`Qwen/Qwen2.5-VL-7B-Instruct`; it contains no snapshot or weights. Combined
with the login node having no GPU and the backend environment being broken,
the current load result is:

| Model | HF full weight load now | vLLM load now | SGLang load now | Evidence |
|---|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | **Unsupported in current snapshot** | **Unsupported in current snapshot** | **Unsupported in current snapshot** | No weights; no GPU; serving prefix broken |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **Unsupported in current snapshot** | **Unsupported in current snapshot** | **Unsupported in current snapshot** | No weights; no GPU; serving prefix broken |
| `Qwen/Qwen2.5-VL-7B-Instruct` | **Unsupported in current snapshot** | **Unsupported in current snapshot** | **Unsupported in current snapshot** | Only an HF ref exists, not model data; no GPU; serving prefix broken |
| `OpenGVLab/InternVL2_5-4B` | **Unsupported in current snapshot** | **Unsupported in current snapshot** | **Unsupported in current snapshot** | Remote model source and weights absent; no GPU; serving prefix broken |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | **Unsupported in current snapshot** | **Unsupported in current snapshot** | **Unsupported** | No weights/GPU; installed SGLang has no OneVision architecture entry |

“Unsupported in current snapshot” does not mean the architecture is unknown.
It means an actual `from_pretrained`/engine start was not possible from the
audited files. See [model_capability_matrix.md](model_capability_matrix.md) for
source-level support and exact revisions.

## Immediate, non-installing conclusion

Before implementation, create one clean, locked Python 3.11 environment rather
than repairing the broken prefix in place; disable user site packages; validate
it inside an A100 job; and download each checkpoint at its documented revision
into an explicit shared cache. Those are future actions and were not performed
by this audit. The first executable target should be Hugging Face full-cache on
one A100 80 GB, followed by separate vLLM and SGLang environments whose versions
are locked independently.
