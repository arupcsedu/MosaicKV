# Frozen complete-owner coding: separate-input preservation check

Freeze the direct additive source encoder and packed-word decoder at `27cf65a`.
The consumed-input engineering run completed all 104 requests, bitwise preserved
128 vocabulary vectors per request, and reduced every declared cache boundary.
It did **not** pass strict equal-or-faster latency. Retain that negative result.
This follow-up tests numerical/general input robustness and memory envelopes,
not latency acceptance, a benchmark quality improvement, or six-mechanism claims.

Select 16 inputs deterministically by SHA-256 of `(20261108, sample_id)` from
the already used 256-image COCO ledger. Exclude both codec-development IDs and
their media hashes. Exclude media aliases. Selection must not read annotations,
answers, quality scores, or previous failure identities. Historical cohort use
is disclosed: these are independent of **codec tuning**, not globally untouched
benchmark confirmation. Freeze both method implementations before selection.
Copy references into a separate analysis artifact; workers receive no references.

The pinned LLaVA-1.5-7B BF16 eager model, original 603-position prompt, processor,
media, quantizer, logical positions, generation controls, and 128-prediction loop
remain unchanged. Compare the storage-only common codec against the frozen direct
additive codec. Apply packed-word replay to **both**. Warm both on the same consumed
development image, then balance measurement order by immutable input index.

Each pair must preserve all 128 **complete vocabulary vectors** bitwise, answers,
mandatory storage, cache isolation, and logical positions. Accepted source packets
must cover all original features, positions and both K/V projections of every
head; weak ownership proofs must show their old source owners retired. Raw
fallback packets remain real, charged owners. No FullKV fallback, prototypes,
CPU residuals, or six-component overlays are allowed. Preserve all failures.

Measure actual initial/final backing bytes and one separate tensor-result-audited
decode-owned peak. Persistent headers, selectors, offsets, source replay, history,
corrections, indices and declared reconstruction tensor workspace are charged.
Also wrap **each complete request** in `WholeRequestEnvelope`: reset before media
loading and suppress inner resets; verify immutable resident model backing before
and after; synchronize after packet retirement/detokenization. Its allocator peak
minus only model backing is a conservative CUDA-KV **upper bound**, including
non-KV allocations. It is not an exact KV measurement and cannot certify matching
or smaller whole-request KV peaks. Do not subtract the arbitrary pre-run baseline.
Opaque vendor memory, CPU objects and Python metadata are not misreported as
PyTorch CUDA storage. Report these exclusions explicitly.

Compute corpus CIDEr with pinned pycocoevalcap/PTB and validate the local reducer
against the official implementation. Recompute the corpus reducer inside each
of 1,000 paired preservation bootstrap replicates. Identical answers imply an
identical score; this is **not** new evidence of improved benchmark quality.

Submit 16 one-A100/12-CPU workers, <=30 minutes each, capped at **six aggregate
A100s** with the tested admission controller. Chain finite CPU analysis with
`afterany` and retain missing/failed rows. No continual log polling or automatic
fresh/paper campaign. Preserve all six research components and the validated
storage-only default; dictionary/Gray GPU work remains stopped.
