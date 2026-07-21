# vLLM backend

## Current status

The common environment installs `vllm==0.7.2`. Its package import and CUDA
smoke passed on an NVIDIA A100-SXM4-80GB, but no model-serving or HF parity run
has passed with this version. Therefore:

- vLLM FullKV measurement rows are currently unsupported;
- `--enable-mosaickv` remains fail-closed;
- no vLLM row is paper-eligible; and
- importing vLLM is not evidence that a model or measurement wrapper works.

The repository contains a measurement-wrapper prototype written for another
vLLM API version. Its explicit version guard rejects the common-lock version
before engine construction. It is not part of the documented execution path
and must be ported, reviewed, and revalidated before use.

## Installed-source capability

Installed vLLM source registers the requested architectures in
`vllm/model_executor/models/registry.py`: `InternVLChatModel` at line 159,
`LlavaForConditionalGeneration` at line 161,
`LlavaOnevisionForConditionalGeneration` at line 164, and
`Qwen2_5_VLForConditionalGeneration` at line 175. These are source
registrations only; none of the corresponding checkpoints has been loaded
through vLLM in the common environment.

The public `LLM` entry point is
`vllm/entrypoints/llm.py:52`. Its public generation surfaces are `generate`
at line 277, `chat` at line 620, and `encode` at line 756. They do not expose a
post-prefill KV callback, request block table, arbitrary KV replacement, or
decode-time block restoration hook. `CacheEngine` in
`vllm/worker/cache_engine.py:20` and `SelfAttnBlockSpaceManager` in
`vllm/core/block_manager.py:21` are internal ownership boundaries, not public
MosaicKV extension APIs.

Consequently, native whole-block selection, prototype injection, and residual
repair are unsupported in the installed version. Internal tensor reachability
must not be described as a safe native API.

## Supported commands

Create or reconcile only the common environment:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
mosaickv/scripts/create_envs.sh --sync common
```

Verify imports and CUDA without loading model weights:

```bash
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

Run the read-only environment report:

```bash
/scratch/djy8hg/env/mosaickv/bin/mosaickv doctor
```

There is intentionally no documented vLLM model-serving command until a
common-lock wrapper passes deterministic FullKV inference and controlled HF
parity. Cache and model files must remain under
`/scratch/djy8hg/cache/mosaickv`; `HF_TOKEN` is accepted only from the process
environment.

## Re-enablement gates

Before documenting vLLM execution, all of the following must pass from a clean
commit:

1. port the wrapper to vLLM 0.7.2 without bypassing its version guard;
2. run pinned-checkpoint FullKV inference on the common prompt/media path;
3. verify deterministic generated token IDs and controlled HF parity;
4. validate TTFT, inter-token, throughput, memory, and cache telemetry fields;
5. preserve raw per-trial traces and a complete manifest; and
6. keep native MosaicKV disabled unless retention-1 parity and logical-position
   invariants pass through a safe integration boundary.
