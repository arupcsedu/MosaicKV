# V104 fixed-capacity joint-reallocation preflight

V103 is a negative runtime preflight. Its two attempted MosaicKV warmups found
byte-feasible route-level candidates, but their assembled caches failed the
unchanged two-token greedy-token/KL/logit gate. The old controller could not
construct another target-feasible cache from its verified capacity levels, so
it returned FullKV. The hard-budget guard correctly rejected that fallback
before packing. The other two preregistered samples were not executed.

V104 addresses only this engineering failure. It does not inspect benchmark
labels, change the 28.5 MB configured cap, relax the 0.02 KL threshold, relax
the 1.0 maximum absolute logit-difference threshold, or permit a greedy-token
change.

## Design

Every layer/KV-head route keeps its ordinary FullKV, midpoint, and target
capacity ladder. During CPU planning, the target boundary additionally exposes
at most two deterministic alternatives. Each alternative restores one omitted
non-protected block near the selection boundary and displaces one retained
non-protected block with the same byte size. Mandatory and atomic layout-group
blocks cannot move.

If the initial final joint probe fails, V104 spends only the remaining bounded
probe allowance on these fixed-capacity alternatives. The first reallocation
round changes one high-risk route per candidate. A second round may change two
routes if both a sequential round and candidate evaluations remain. Every
complete alternative cache is constructed in an isolated state and must pass
the unchanged greedy-token, forward-KL, absolute-logit-difference, and horizon
checks. Failed probe states are discarded.

This is deliberately not a cross-head variable-capacity exchange. Hugging Face
stores the heads of a layer in a dense tensor. Restoring a long route while
shortening another would increase the shared physical sequence dimension and
could hide padding bytes. V104 therefore preserves the active bytes and
physical position count of every route exactly.

## Scientific boundary

All four V103 preflight identities are consumed and excluded. V104 will select
two new TextVQA-training and two new InfoVQA-validation identities from the
reviewed ID-only catalogs using a new SHA-256 salt, without opening labels,
references, prompts, media, scores, or failure outcomes during selection.

The reviewed protocol authorizes only deterministic ID-only ledger
materialization and two one-A100 preflight workers after the implementation,
protocol, materializer, and worker launcher share one clean immutable commit.
The materialized worker specification records that resolved Git SHA. It does
not authorize calibration, candidate freezing, held-out materialization, or
held-out execution. A positive four-sample preflight still requires separate
review before any calibration action.
