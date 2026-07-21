# Future-query forecasting

`mosaickv.forecasting` implements MosaicKV's training-free online forecast of
future decoder queries. It consumes query vectors captured during an existing
full-cache prefill; it does not invoke `prefill()` itself.

## Modes and configuration

The strict `[forecasting]` configuration supports:

- `prompt_window`: statistics and centroids from the last `prompt_window`
  logical prompt positions. This mode requires `prompt_window > 0` and
  `draft_steps = 0`.
- `draft_rollout`: `draft_steps` temporary greedy FullKV decode steps. This
  mode requires `prompt_window = 0` and `draft_steps > 0`.
- `hybrid`: prompt-window statistics combined with temporary draft queries.
  Both sizes must be positive.

The remaining controls are `centroid_count`, `covariance` (`diagonal` or
`full`), `low_memory_centroids`, and `centroid_iterations`. Query statistics
use FP32 working tensors and population covariance/variance. A requested
prompt window longer than the prompt is clipped, and the actual positions are
recorded in provenance.

Captured query tensors have shape `[1, query_heads, sequence, head_dim]`.
Grouped-query attention is handled explicitly: contiguous query-head groups
are mapped to their associated KV head, and every member of a group remains a
forecast sample. Construction fails unless the query-head count is divisible
by the observed KV-head count.

## Draft isolation

`forecast_from_prefill(adapter, prefill, config)` accepts a `PrefillOutput`, not
prepared model inputs. Draft rollout begins with the prefill's deterministic
greedy next token and calls only `decode_one_token`. Before drafting it clones
the cache, attention mask, and model-specific decode state. Draft decoding may
mutate that clone; it cannot share K/V tensors with the state reserved for the
final run. After the rollout, the original K/V tensors and attention mask are
compared with their pre-draft copies. Any change raises an error. Draft tokens
and their extended cache are discarded.

## Forecast output

`QueryForecast` contains one `KVHeadQueryForecast` per layer and KV head:

- prompt mean;
- either full prompt covariance or diagonal variance;
- draft query samples;
- L2-normalized forecast centroids;
- normalized empirical cluster weights; and
- query-head, prompt-position, sample-count, mode, and isolation provenance.

The default low-memory centroid method is a deterministic, single-pass
spherical update. Its centroid computation requires O(K·D) auxiliary memory
for `K` centroids of dimension `D`, beyond source/statistics buffers, and never
materializes an N-by-K assignment matrix or a combined hybrid sample tensor.
The non-low-memory option uses deterministic farthest
initialization and fixed-iteration spherical k-means.

`ForecastTiming` keeps cache cloning/integrity checks, draft decoding, query
preparation, prompt statistics, centroid construction, and total forecast
overhead separate. CUDA tensors use synchronized `torch.cuda.Event` intervals;
CPU tensors use `perf_counter`.

## Evaluation-only oracle and diagnostics

True future queries are deliberately absent from the online forecasting
package. Evaluation code must explicitly import:

```python
from mosaickv.evaluation.oracle_queries import (
    collect_evaluation_only_true_future_queries,
)
```

This API clones the original FullKV prefill state and collects the query
vectors from a deterministic reference decode. Its object is named
`EvaluationOnlyOracleQueries`, carries an `evaluation_only` source label, and
is not re-exported from `mosaickv.forecasting` or `mosaickv.evaluation`.

`mosaickv.evaluation.forecast_diagnostics.evaluate_forecast_quality` reports,
per KV head and as macro means:

- maximum-centroid cosine similarity for each true future query;
- Spearman correlation between predicted and true attention rankings;
- recall of oracle-selected blocks under deterministic top-k selection; and
- normalized regret in true attention mass relative to oracle selection.

Attention inputs to diagnostics must be nonnegative, aligned, RoPE-aware
attention values. This is explicit because supported adapters capture pre-RoPE
`q_proj` outputs while their caches store post-RoPE keys; the diagnostic API
does not silently dot incompatible representations.

The common-environment unit suite covers every mode, strict
zero-window/zero-rollout rules, GQA mapping, diagonal and full covariance, both
centroid algorithms, draft-cache mutation isolation, reproducibility, oracle
namespace isolation, and all four diagnostics. Pretrained-checkpoint forecast
quality remains unsupported until a clean pinned-checkpoint gate passes.
