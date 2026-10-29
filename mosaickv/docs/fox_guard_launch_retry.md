# Launch-only correction for October 29 installed-output guard

The first two Fox workers exited in one second at argument parsing: the generic
campaign CLI did not accept the custom `evidence_canary` stage. No model was
loaded and no benchmark inference happened. Preserve that failed root and its
successful `afterany` completeness report. The PrefixKV source remains ad7c5c6
and must not change while its workers run.

This branch changes only launch/orchestration interfaces and tests. It adds the
two explicitly registered custom stages to the actual worker CLI, restricts
them to worker dispatch, and exercises that dispatch in CPU tests. Runtime,
guard, configuration, K/V packing and numerical thresholds are unchanged.

The original failed Fox terminal callback already transferred six slots to
PrefixKV. Do not start a new Fox array concurrently with its six-worker quality
phase. A finite controller follows PrefixKV's existing after-references and
after-canary decision callbacks, then waits for its final calibration analysis.
It neither polls logs nor modifies the running source, arrays or their inputs.

Only after quality jobs end, permit the launch retry alongside at most two
PrefixKV timing workers. Fox may use four A100s in that phase, so the aggregate
limit stays six. Before submission, inspect the owner's queue and reject any
unknown/quality GPU job. Bind the new clean-source CPU gate. Numerical canary
must pass before the registered 32-input development study; quality thresholds,
data-selection rule and all performance-reporting restrictions stay unchanged.

Independent confirmation remains blocked by its scientific prerequisites.
Calibration results may not be used to alter this already frozen algorithm.
