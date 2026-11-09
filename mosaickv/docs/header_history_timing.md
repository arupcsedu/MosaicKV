# Clean timing characterization, not promotion of a failed memory gate

`selective-history64-20261108-v1` preserved every greedy token across 128
predictions on both consumed COCO inputs. Full-horizon maximum logit differences
were 0.25 / 0.1875 and maximum KL 0.000438615 / 0.000279550, below the unchanged
1.0 / 0.02 limits. Initial KV was identical to the common-optimized storage
comparator, final KV was approximately 1.03 MiB smaller, and one audited peak
was one byte larger. Keep that strict peak failure; do not round it into a pass.
This is not complete memory superiority or independent quality evidence.

The bounded follow-up characterizes cost only: two consumed-input workers,
each with five warmups per arm and twenty paired timed repetitions, randomized
method order by fixed seed. Both arms get identical complete predicate, early
exact headers and late exact scale replay. Only the candidate gets older INT8
group32 history with 64 exact recent sealed tokens plus the exact mutable tail.
Keep the original model, media, prompts, 128 predictions, precision, backend,
private graph lifecycle and numerical guard. FullKV is not a latency comparator.

Disable allocator/weakref auditing inside timed requests. Native CUDA events and
synchronized boundaries remain in the common runtime; include construction,
transfers, graph lifetime, detokenization and export in end-to-end time. Preserve
all warmup/timed rows, failed rows, initial/final backing bytes, predictions,
numeric checks, model/dataset revisions and starting/ending GPU/process state.
Do not reuse an earlier audited peak as if it were measured in a timed trial.

Bootstrap the twenty paired repetitions separately for each consumed input;
report p50, p95, differences and latency ratios. Use 10,000 replicates and
Bonferroni-adjusted bounds for the two inputs. Equal-or-faster requires an upper
ratio bound <=1.00; <=1.05 remains a trade-off. Fixed-input timing intervals are
not confidence intervals over benchmark-example quality.

Use two one-A100/12-CPU workers, at most one hour each, with an explicit
3,300-second worker budget. Every producer receives `afterany` analysis and
aggregate admission stays at six A100s. Even a timing pass cannot override the
unchanged initial/peak memory result or launch a fresh cohort/paper campaign.
