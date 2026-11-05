# Joint K-begin / real V-commit: separate correctness gate

The key-only transaction passed full 128-vector bitwise/isolation checks. Its
paired timing screen must complete and miss the 4% target before this extension
is submitted. No threshold is changed. Independent benchmark N=0: reuse only
the two consumed inputs, with no answers or benchmark labels used in construction.

Keep the original pinned full-MHA Llama q_proj, RoPE, eager QK/softmax/PV, o_proj,
source packets and history codec. The default and key-only paths are unchanged.
For explicit `kv` mode, begin the cache transaction using the real key alone.
Reserve but never read the current raw V slot. After QK/softmax, the ordinary
prompt-V graph also computes the real incoming raw V with the already verified
native configuration. Commit it to the original history slot **before** history
reconstruction or PV. A separate v_proj invocation is an error. No fake/zero
raw V is passed to cache.update, and no history can seal while a value is pending.
The common one-buffer attention lease still aliases K then V, as before.

Strict metadata guards enforce per-layer order, one K consumer, one real V
commit, same physical slot, original logical positions, no active overlap,
no borrowed workspace during a pending transaction, and no pending retirement.
The extra incoming workspace remains 8 KiB; no second full cache is introduced.
CPU tests compare split/original history bytes and pinned 4096-wide attention
outputs exactly, including a NaN-poisoned pending V slot and a forbidden
separate v_proj. Weights and parameter versions must not change.

After a full zero-failure/zero-skip CPU suite, run original-native / joint-KV /
original-native on each consumed input, with full tensor audits. Require every
one of 128 parent vocabulary vectors bitwise, 4032 folded warm keys, 32 original
cold keys, 4064 real value commits, 8128 graph replays, packet/position parity,
no FullKV fallback, complete owner retirement and A-B-A storage isolation.
Actual initial/final/decode-owned peak storage is recorded. Whole-request GPU
peak and independent quality remain uncertified. Only a subsequent paired
full-request timing can establish whether the additional execution change helps.
No fresh cohort, larger campaign or paper claim follows automatically.
