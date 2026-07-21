# SGLang backend

## Current status

The common environment installs `sglang==0.4.3.post4`. Its package import,
native-kernel imports, and CUDA smoke passed on an NVIDIA A100-SXM4-80GB, but
no model-serving or HF parity run has passed with this version. Therefore:

- SGLang FullKV measurement rows are currently unsupported;
- native MosaicKV remains fail-closed;
- no optimized server profile is enabled; and
- no SGLang row is paper-eligible.

The repository contains a server-wrapper prototype written for another SGLang
API version. Its flags, telemetry assumptions, and explicit capability guard
are not accepted as a common-lock execution path. It must be ported and
revalidated before use.

## Installed-source capability

The installed source defines `Qwen2_5_VLForConditionalGeneration` in
`sglang/srt/models/qwen2_5_vl.py:474` and exports it through `EntryClass` at
line 759. `sglang/srt/models/llava.py` exports SGLang-specific LLaVA classes at
line 574. These findings establish source registration only. Neither requested
Qwen checkpoint nor a LLaVA checkpoint has been loaded through SGLang in the
common environment.

The public engine is `sglang/srt/entrypoints/engine.py:73`; `Engine.generate`
starts at line 112. It does not expose a post-prefill KV callback or an atomic
operation for request-scoped cache selection and restoration.

Internally, `ReqToTokenPool` at
`sglang/srt/mem_cache/memory_pool.py:46` owns request-to-token mappings,
`TokenToKVPoolAllocator` at line 119 owns token slots, and `RadixCache` at
`sglang/srt/mem_cache/radix_cache.py:79` owns reusable prefix entries. The
`set_kv_buffer` method declared at `memory_pool.py:109` is an attention-backend
write primitive; it does not atomically update all three ownership structures
or multimodal logical positions. It is not a safe public MosaicKV API.

Consequently, whole-block selection, prototype replacement, and residual
repair are unsupported in the installed SGLang version.

## Supported commands

Create or reconcile only the common environment:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
mosaickv/scripts/create_envs.sh --sync common
```

Verify imports and CUDA without launching a server or loading weights:

```bash
sbatch --reservation=bi_fox_dgx mosaickv/slurm/env_smoke.sbatch
```

Run the read-only environment report:

```bash
/scratch/djy8hg/env/mosaickv/bin/mosaickv doctor
```

There is intentionally no documented SGLang server command until the wrapper
passes deterministic FullKV inference and controlled HF parity in the common
environment. Cache and model files must remain under
`/scratch/djy8hg/cache/mosaickv`; `HF_TOKEN` is accepted only from the process
environment.

## Re-enablement gates

Before documenting SGLang execution, all of the following must pass from a
clean commit:

1. port server arguments, request formatting, and telemetry parsing to the
   common-lock version;
2. load each supported pinned checkpoint and preserve exact server arguments;
3. verify deterministic repeats, request isolation, and controlled HF parity;
4. validate TTFT, decode, throughput, memory, and Radix-cache observations;
5. preserve raw per-trial traces and a complete manifest; and
6. keep native MosaicKV disabled unless retention-1 parity, byte accounting,
   and request-ownership invariants pass through a safe integration boundary.
