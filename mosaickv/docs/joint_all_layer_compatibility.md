# Frozen all-layer joint-row check

The preceding bounded numerical kernel study found configuration
`(ID31,tile15,splitK1,reduction0,swizzle0,custom0,stages18,inner4)` exactly
reproduced incoming raw K/V and its own prompt rows at all three registered
prefixes on both consumed inputs. This is an operator result, not a runtime win.

Freeze that configuration without a timing search. Check all 32 layers, K and V,
at decode updates 1,64,126 (192 comparisons/input) on those same two consumed
inputs. Original model operators, parent logits and source packets remain
unchanged. A single changed incoming/prompt word blocks runtime implementation.
No benchmark labels, new identities, quality/calibration cohorts or performance
campaigns are used. The diagnostic retains decoded tails in an explicitly
charged 33 MiB workspace, not a purported production cache. Whole-request GPU
peak is not certified. API/header/config provenance and automatic failure-aware
analysis follow the preceding numerical study. Two one-A100 workers run only
after a full zero-failure/zero-skip CPU suite. No automatic runtime promotion.

A positive result permits engineering a request-private atomic transaction:
joint K projection before the ordinary cache update, deferred V projection
before V consumption/history sealing, once-only code replay, original RoPE,
bounded raw tails and exact tensor accounting. That implementation would need
separate state-machine tests, full 128-vector parity, storage and paired timing
against every equally optimized baseline. No placeholder may reach attention
or history verification; all pending state must be committed or rejected.
