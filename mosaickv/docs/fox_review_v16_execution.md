# V16 bounded corrections

V15's two model requests passed the unchanged 32/128-token fidelity checks, but
the ownership codec rejected expanded metadata before a trace was published.
Their failed results are immutable. V16 permits exactly those two consumed
requests to repeat after the complete clean-source CPU suite passes. Inputs,
model, precision, guards, history, and cache arithmetic do not change.

The default metadata codec remains bounded at 8 GiB of semantic JSON. The
ownership producer explicitly registers 32 GiB of semantic JSON only; encoded
JSON, compressed size, node, item and depth limits do not increase. Decoding
requires that explicit scoped policy. A complete compressed archive and
hash-bound raw receipt are saved before expensive semantic hashing. An
interrupted full receipt remains a failure, but complete raw evidence can then
be recovered on CPU without repeating inference.

Two CPU lanes run alongside the corrective ownership diagnostic:

1. Reconcile the eight saved V11 constructed-cache rows. Preserve all numerical
   failures. Endpoint growth is not attributed to a component without actual
   per-owner evidence, and decode-owned peaks are not whole-request peaks.
2. Advance only the verified recovered history frontier, preserving original
   producer lineage and scanner semantics. At most four progress-checked
   eight-worker segments may run. Compact checkpoint whitespace does not alter
   canonical state hashes. Missing video journals remain an explicit blocker
   even if the traversal finishes.

The controller is `mosaickv/scripts/evicastkv_review_v16.py register --root
/scratch/djy8hg/workdir/fox-review-v16/NEW_IMMUTABLE_RUN --execute`, invoked from
the clean source after sourcing `mosaickv/scripts/cache_env.sh` and setting
`PYTHONPATH` to the package and scripts directories. It uses only the common
scratch environment. Each corrective GPU worker requests one A100, 12 CPUs and
20 minutes through `bi_fox_dgx`; aggregate project A100 occupancy is limited to
six. CPU jobs use `bii,standard`. Completion uses bounded `afterany` callbacks,
not queue/log polling. No fresh cohort, benchmark confirmation, paper campaign,
baseline superiority, or six-component benefit is authorized by an execution
or metadata pass.
