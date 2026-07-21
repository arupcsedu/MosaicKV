# SGLang native MosaicKV blocker

Status: **unsupported and fail-closed** for the audited installed SGLang
`0.5.10.post1`. No result row produced by this repository may label a
selection performed outside SGLang as native SGLang MosaicKV.

## Audited source

The source inspected is the functional pinned distribution at
`/scratch/djy8hg/env/mosaickv/lib/python3.11/site-packages/sglang`.
The cited files are byte-identical to the original audit copy under
`drc_rag_bench_env`. Runtime support is not inferred from these files; Stage A
is established separately by `scripts/verify_envs.py` and the GPU smokes.

### Public execution boundary

`srt/entrypoints/engine.py` defines `Engine` at line 143. Its public methods
include `generate` (line 271), `async_generate` (361), `flush_cache` (772),
sessions (775 and 805), `get_server_info` (835), and whole-engine memory
release/resume (1040 and 1046). There is no public request-scoped API to
inspect a completed prefill cache, replace its block table, preserve a sparse
logical-position vector, or atomically commit such a mutation before decode.

The HTTP surface has the same limitation. `srt/entrypoints/http_server.py`
defines `/generate` at lines 657-692, `/server_info` at 601, and `/flush_cache`
at 718. These APIs expose generation, aggregate state, and whole-cache flush;
none exposes request-local KV mutation.

### KV memory and request mapping

- `srt/mem_cache/memory_pool.py:126` defines `ReqToTokenPool`. Its
  `req_to_token` tensor is a dense `[request, logical position]` mapping and
  `write` at line 149 overwrites mapping entries. This is an internal pool,
  not a transaction-safe serving extension point.
- `srt/mem_cache/memory_pool.py:741` defines `MHATokenToKVPool`. `_create_buffers`
  at line 845 creates one K and one V tensor per layer with shape
  `[size + page_size, kv_heads, head_dim]`. `set_kv_buffer` at line 995 writes
  tensors for attention execution, and `move_kv_cache` is at line 1034. The
  warning at lines 973-976 states that
  cache getters are for the attention backend, not information access, because
  layerwise loading can require synchronization.
- `srt/mem_cache/allocator.py:117` defines the token allocator, while
  `PagedTokenToKVPoolAllocator` at line 356 owns page-aligned allocation.
  `alloc_extend` and `alloc_decode` at lines 403 and 451 allocate new physical
  locations. An out-of-band deletion would bypass allocator ownership and can
  cause aliasing or double-free.
- `srt/managers/schedule_batch.py` creates `out_cache_loc` through the allocator
  at line 1615, installs it on the batch at line 1743, allocates decode slots at
  2116, and later frees locations selected from `req_to_token` at 2472. There
  is no callback between completed prefill and the first decode step that owns
  all of these mutations atomically.

### RadixAttention entries

`srt/mem_cache/radix_cache.py` defines `TreeNode` at line 121 and `RadixCache`
at 285. `match_prefix` (374), `insert` (446), `cache_finished_req` (463),
`cache_unfinished_req` (510), and `evict` (582) manage prefix-tree ownership.
They represent token-prefix entries and physical indices; they do not support
a request-specific sparse subsequence with holes in original logical
positions. Editing only the request table would leave tree ownership and lock
references inconsistent; editing only the tree would leave the running
request mapping inconsistent.

### Attention and logical positions

`srt/model_executor/forward_batch_info.py:280` defines `ForwardBatch`.
`out_cache_loc`, `positions`, the request and KV pools, and multimodal
`mrope_positions` live at lines 294, 327, 369-370, and 408. Ordinary positions
are derived from sequence lengths at lines 529-580. Qwen2.5-VL mRoPE positions
are built separately at lines 687-810 and replace scalar positions in
`srt/models/qwen2_5_vl.py:731-767`.

`srt/layers/attention/base_attn_backend.py:18` defines the attention backend
interface (`init_forward_metadata`, `forward_extend`, and `forward_decode`).
The correctness backend's `TritonAttnBackend` is defined in
`srt/layers/attention/triton_backend.py:56`; it captures the dense
`req_to_token` tensor at line 86, builds forward metadata at line 236, and
implements extend/decode at lines 806 and 1038. The Triton and FlashAttention
implementations consume dense logical prefixes from `req_to_token`; the
interface accepts no `logical_positions` vector for a sparse cache. Therefore
compacting selected physical entries would renumber RoPE/mRoPE positions
unless multiple private structures and kernels were patched together.

## Missing atomic hook

A safe whole-block implementation needs one scheduler-owned operation after
prefill and before decode that accepts a request ID and:

1. returns immutable per-layer/head KV views plus physical and original
   logical positions;
2. allocates a replacement physical block set through the active allocator;
3. installs an explicit sparse logical-position/mRoPE map consumed by every
   supported attention kernel;
4. updates `ReqToTokenPool`, the request sequence metadata, Radix tree values,
   lock references, and allocator ownership atomically;
5. rolls back all changes if validation fails; and
6. scopes the mutation to one request so no Radix entry or physical slot can
   become visible to another request accidentally.

No such hook exists in `0.5.10.post1`.

## Required patch and correctness risks

A proposed upstream interface is a scheduler command such as
`compress_request_kv(request_id, selected_physical_blocks,
original_logical_positions)` plus an attention-backend capability declaration
for sparse logical positions. The scheduler—not an HTTP wrapper—must validate
page alignment, ownership, mRoPE coordinates, prefix-tree sharing, and decode
slot allocation before committing it.

Without that interface, a local patch risks:

- renumbered 1-D RoPE positions or corrupted Qwen mRoPE coordinates;
- an attention page table whose declared sequence length exceeds its entries;
- freeing a Radix-shared physical page still referenced by another request;
- stale prefix hits returning compressed request-specific state;
- allocator aliasing, double-free, or request-to-request KV leakage; and
- different behavior across Triton, FlashAttention, and CUDA-graph paths.

For these reasons `--enable-mosaickv` raises before server launch and the
Stage B tests are represented by this explicit unsupported verdict. Prototype
replacement and residual repair are not attempted.

## Requested native-test disposition

- **100% retention parity:** not claimed or executed for native MosaicKV;
  the feature gate rejects the request before model loading. Ordinary Stage A
  SGLang FullKV is measured separately.
- **HF `mosaickv_exact` block-ID agreement:** unsupported because the engine
  exposes neither a comparable post-prefill capture nor a safe selection
  commit. No HF selection is replayed and mislabeled native.
- **Native active-byte accounting:** unsupported without a committed sparse
  block table. Stage A logical FullKV bytes are independently validated from
  exact model geometry and sequence lengths.
- **Unsupported model/kernel behavior:** explicit model, SGLang-version,
  attention-backend, tensor-parallel, page-size, and feature checks fail before
  an experimental row can be emitted.
- **Request isolation:** Stage A jobs `17160103` and `17160299` passed an A-B-A
  token-identity probe after an intervening distinct media request. A native
  path has no corresponding result because it does not exist.
