# V42: one-proof correction, no inference retry

The V41 recovery attempt for the already completed V6 derived analysis found
a lexical-path mismatch: original manifest keys use `/scratch/djy8hg`, while
the registered report path uses `/sfs/weka/scratch/djy8hg`. The contents and
hashes had not changed. A failed dictionary lookup blocked the proof.

This correction accepts only that declared prefix pair, rejects conflicting
hashes, and retains the original manifest spelling and hashes. It still checks
the truncated original, derived report, original GPU plan/producer, analyzer
patch, complete 64/64 execution-control fields, and both saved Parquet copies.
No benchmark answer decoding, dataset scan, GPU inference, or metric rerun is
needed. The recovered report remains execution-only evidence, not numerical
quality or freshness certification.

A complete current-source CPU suite qualifies the new helper. The recovery
job depends on both that suite and V41 analysis, and retries only the exact V6
occurrence if it remains unresolved. All other V41 proofs are imported, not
repeated. All scientific admission gates and false freshness/media permissions
remain unchanged.
