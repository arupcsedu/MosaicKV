# PrefixKV common-HF transport limit

## Finding

The shared MosaicKV Hugging Face cache transport has one physical prompt
length and one attention-mask extent for all decoder layers. The pinned
PrefixKV algorithm instead permits a distinct retained prefix length per
layer. These are not interchangeable representations.

In the profile debug artifact
`/scratch/djy8hg/runs/mosaickv/prefixkv/profile-policy-debug-3cb9bec`, the
PrefixKV planner selected `62,799,872` logical K/V bytes from
`314,048,512` source bytes (about 20%). Common-HF packing padded the variable
layer lengths to a common extent, materializing `225,443,840` bytes instead.
The generated outputs also collapsed (corpus teacher-forced PPL `480.6581`).
This is a negative transport/debug result, not a measurement of PrefixKV.

## Enforced policy

`PrefixKVCompressionPlan.has_uniform_layer_capacity` identifies plans that
can be represented by the common transport without padding. The unified HF
runtime now rejects a variable-layer PrefixKV plan before any output row is
written. Equal-per-layer `fixed_global` remains executable only as an
explicitly labeled control, not as paper-faithful PrefixKV.

Paper-native PrefixKV PPL/ROUGE-L measurements must use the isolated pinned
upstream LLaVA runner (`scripts/run_prefixkv_llava_parity.py` and its
dedicated environment), which supports variable per-layer cache lengths. It
must still meet the protocol's dataset-revision, FullKV-reference,
published-operating-point, and clean-worktree gates before it is reported.

The runner now accepts explicit, disjoint comma-separated calibration and
evaluation index sets. It writes per-sample raw observations and sufficient
statistics (`nll_sum`, `evaluated_token_count`), then calculates teacher-
forced corpus PPL as `exp(sum(nll_sum) / sum(token_count))`. It must not
average per-sample PPL values. Its ROUGE-L result is the mean of paired
per-sample FullKV-reference comparisons. For example, a canonical 1,000-item
run is configured through `MOSAICKV_PREFIXKV_CALIBRATION_INDICES` and
`MOSAICKV_PREFIXKV_EVALUATION_INDICES` when submitting
`slurm/prefixkv_official_parity.sbatch`; the source SHA and all index lists are
captured in the output manifest. This is runnable infrastructure, not yet a
new measured reproduction result.

## Consequence

No existing common-HF `prefixkv_reimpl` OCR or LLaVA-Description row with
physical padding may be used as evidence of PrefixKV performance. The rows
remain preserved as negative debugging artifacts with their original
provenance.
