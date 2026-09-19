"""Versioned paper-native metric interfaces.

The legacy :class:`~mosaickv.evaluation.results.EvaluationResult` remains the
canonical per-sample generation record.  This package adds a separate metric
stream so corpus metrics (for example CIDEr and perplexity) are not forced
into the legacy scalar ``task_score`` field.
"""

from mosaickv.evaluation.metrics.comparison import (
    MethodMetricSummary,
    PairedMethodComparison,
    compare_paired_methods,
    summarize_metric,
)
from mosaickv.evaluation.metrics.registry import MetricDefinition, MetricRegistry
from mosaickv.evaluation.metrics.scorers import (
    aggregate_cider,
    aggregate_mean,
    aggregate_perplexity,
    anls_score,
    exact_match_score,
    rouge_l_f1,
)
from mosaickv.evaluation.metrics.statistics import PairedMetricComparison, paired_bootstrap
from mosaickv.evaluation.metrics.storage import (
    JsonlMetricStore,
    load_metric_jsonl,
    write_metric_parquet,
    write_metric_parquet_tables,
)
from mosaickv.evaluation.metrics.types import (
    MetricDirection,
    MetricObservation,
    MetricSample,
    MetricScope,
)

__all__ = [
    "JsonlMetricStore",
    "MethodMetricSummary",
    "MetricDefinition",
    "MetricDirection",
    "MetricObservation",
    "MetricRegistry",
    "MetricSample",
    "MetricScope",
    "PairedMethodComparison",
    "PairedMetricComparison",
    "aggregate_cider",
    "aggregate_mean",
    "aggregate_perplexity",
    "anls_score",
    "compare_paired_methods",
    "exact_match_score",
    "load_metric_jsonl",
    "paired_bootstrap",
    "rouge_l_f1",
    "summarize_metric",
    "write_metric_parquet",
    "write_metric_parquet_tables",
]
