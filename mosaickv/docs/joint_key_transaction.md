# Joint incoming-key transaction: correctness before latency

Frozen native configuration ID31/tile15/splitK1/reduction0/swizzle0/custom0/
stages18/inner4 passed every actual layer K/V at three prefixes on the same two
consumed inputs. This permits this isolated execution experiment, not promotion.

Keep the original model weights, attention, RoPE, cache positions, source codes,
protected evidence, history codec and numerical correction limits unchanged.
Use the original zero-workspace native reader for full 128-row chunks; append
the current raw normalized input to the final incomplete chunk and project with
the frozen compatible configuration. Its incoming raw key is returned through
the original key-projection boundary. V projection remains original: **there
are no placeholders or deferred value commits**. First-update keys remain
original, allowing the existing 64 graphs to capture exactly once.

On warm updates, issue the same K prompt graph before cache.update, under a
request-private layer/epoch authorization. The ordinary materialization consumes
that graph once; it still fills original history, original RoPE-applied new key,
and the unchanged value path. Ordered code replay advances exactly once.
Wrong order, stale layer/epoch, duplicate consumption, active leases, foreign
buffers, unconsumed pending work, nested requests and foreign policies fail.
The shared incoming buffer adds 8 KiB to the existing 2 MiB native workspace;
all owners participate in independent storage inventories and are retired.
Native correction fitting remains original-bit validated and capped at 32 KiB
per layer / 1 MiB per request. No model weights are changed or learned.

CPU state-machine/default-policy/storage tests precede two one-A100 workers.
Each worker executes audited original-native / joint-key / original-native
complete 128-vector requests, requiring own-parent bitwise vocabulary logits,
8128 actual graph replays, 4032 folded keys and 32 original cold keys, exact
original protected K/V, no FullKV fallback and A-B-A storage isolation. The
constructor and source-input copies are not excluded from future request time.
This instrumentation is NOT latency acceptance and not independent quality.
Whole-request GPU peak remains uncertified. Other source-less baseline readers
have no eligible prompt-source projection, but common prepared dispatch and
accounting optimizations remain available to them. No fresh cohort or larger
campaign is authorized by this correctness result.
