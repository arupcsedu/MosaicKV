# PrefixKV provenance-timeout retry

Array `22126353` completed eleven compressed workers and failed worker 8:
MM-Vet, Prompt Top-K, 80 MiB, eight registered inputs. Both FullKV reference
workers had already passed. Worker 8 produced predictions and metric artifacts,
then timed out in `git status --porcelain=v1` during final manifest creation.
The incomplete attempt is retained; without its final manifest it is not
promoted into the complete paired comparison.

The manifest reader now retries the identical read-only Git-status command
once: first 30 seconds, then 120 seconds. No check is removed, no status is
cached, and untracked/submodule changes remain visible. A second timeout raises
`ManifestError` and leaves cleanliness unknown, never canonically eligible.
Other commands keep one 30-second attempt. This changes provenance collection,
not measured model/compression/decode timing or cache selection.

The retry uses a separate clean source worktree and a new immutable output root.
Only worker 8 is submitted. The thirteen successful worker records are imported
by exact file hash, with their original producer Git SHA and original plan SHA;
they are not rewritten as if the corrective source produced them. Every bound
input and completed artifact hash is checked. The only changed package file
must be `manifest.py`; any model/cache/config algorithm change blocks reuse.
The original failure and failed analysis report remain unchanged.

The same checkpoint, inputs, media, precision, eager attention, 1,024 output
tokens, 80 MiB cap, seed, parity gate, five warmups, complete PPL targets and
FullKV ROUGE-L reference are used. The regenerated complete report combines 104
reused successful sample rows with eight retried rows. It reruns no other GPU
arm and changes no scientific thresholds.

Current scheduling reserves five A100 slots for EviCastKV and at most one for
PrefixKV, preserving the six-A100 aggregate limit. Any conditional timing array
also uses a one-worker throttle. The registered timing scope remains two tasks
x two IDs x twenty repetitions x two methods: 80 requests per task worker, 160
overall. This corrects the old plan's ambiguous aggregate count, without adding
identities, repetitions, methods, or a new experiment.

Run `retry_prefixkv_isoquality.py prepare --parent PARENT --root NEW_ROOT`, then
`submit --root NEW_ROOT --plan-sha PRINTED_SHA`, from the committed clean source
with the common scratch environment. A finite `afterany` CPU analysis handles
successes and failures; failed execution never authorizes a larger campaign.
