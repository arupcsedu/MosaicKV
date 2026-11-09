# Common private prefill score/probability storage

The requested KV/workspace peak witness contains a full BF16 score matrix and
separate full probability matrix. The eager attention function owns its matmul
output exclusively. Under this default-off nongrad BF16/FP16 policy, retain the
same native matmul, separate native scalar multiplication, same-dtype mask add,
complete FP32 row softmax, BF16/FP16 cast and native value matmul. Write resulting
probability rows back into that private score owner instead of allocating a
second full probability matrix. No caller/model tensor is modified.

Unsupported mask promotion and autograd fail explicitly. Mask, token, source,
position, projection, dtype and numerical thresholds never change. The same
policy applies to every compatible baseline and storage-only comparator, not
only EviCastKV. These baseline precision variants remain local, not official.

Execution: CPU fixtures -> real CUDA output/probability/caller-input bit tests ->
128-token raw FullKV identity -> 128-token consumed storage-reader identity and
requested-KV/workspace peak -> four compatible baseline before/after controls ->
conditional direct randomized timing with identical common optimizations.
Every producer has finite afterany completeness analysis. No fresh cohorts,
calibration, paper-scale execution, independent quality claim or six-adaptive-
mechanism claim is authorized by these two consumed-input operator tests.
