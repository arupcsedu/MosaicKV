# Consumed-input native reconstruction operator study

Preserve both negative policies from `full-prompt-native-consumed-20261103-v1`.
The original reader remains default. This is the frozen source-precision/storage
variant, not the complete six-mechanism algorithm. Baselines are local storage
adaptations, not official code. Independent quality N=0; no new cohort or paper
claim is authorized. Neither own-parent parity nor different-byte cache points
establish quality superiority. Whole-request GPU peak remains uncertified.

Capture the same two consumed inputs at original prefill completion, check all
original packet hashes, and stop before decoding. Use layers 0/16/31 and history
padding 0/64/127. Refit exact numerical repairs only against the frozen original
packet decoder, never against answers. Verify identical K/V bits and untouched
history across the existing 0/1/128 fit checks and the measured padding grid.

Measure source unpack, per-head BMM, RoPE, protected scatter, old transport
patches, new numerical repairs, and workspace allocation/release. Intermediate
CUDA events perturb dispatch and include stream-idle gaps; these are not pure
kernel times. Use separate uninstrumented, randomized five-warmup/twenty-trial
pipeline measurements. Retain every raw observation.

The two bounded alternatives are a single wide GEMM and a fixed 128-token wide
GEMM tile. Both use original weights and explicit layout copies into the same
head-major lease. Full and tiled output buffers are additional charged storage,
not hidden library or alias buffers. The tiled alternative is registered solely
because the full output buffer may consume the existing memory margin; no row,
warp, or quality sweep is authorized. All products are measured as diagnostic
ablations; only a measured BMM bottleneck justifies further implementation.

Require at least 10% lower operator-pipeline CUDA interval than the ORIGINAL
fused reader on every registered geometry and both inputs before considering
full-request integration. Also require explicit initial/final/peak feasibility
against the smallest baseline boundary, including buffers and refitted repairs.
Operator success cannot promote a serving reader: it requires complete-request
bitwise/storage controls, then randomized paired latency verification. Preserve
the previous <=0.99 within-method and <=1.05 Prefix adaptation timing gates.

Clean immutable SHA, common environment, all caches under scratch. CPU full
suite first; two one-A100/12-CPU/48-GiB workers, ten-minute limits, then afterany
CPU analysis. At most six A100s in aggregate; no continuous polling. No automatic
source edits, new identities, or larger campaign in this bounded DAG. The active
agent may analyze completed results and implement a separately versioned narrow
follow-up under the user's current request without weakening numerical gates.
