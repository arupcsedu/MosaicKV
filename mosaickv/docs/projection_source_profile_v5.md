# Bounded profile after measured source-reconstruction overhead

Paired timing array 22186343 and analysis 22186344 completed successfully.
Original selected-cache outputs remained bitwise identical. Source storage
cost EviCastKV 33.5–35.1% more decode time than encoded K/V on the two consumed
inputs. This is not a latency win. The negative union-restoration result is
unchanged and no fresh evaluation follows.

Use those same consumed inputs, frozen positions, BF16 LLaVA-1.5 checkpoint,
eager backend, original logical positions and append-only history. Compare
encoded K/V and hybrid source storage for BOTH selectors. Retain all normal
32-token/retention-one controls. Then warm each profiled format five times,
and profile exactly four own-greedy updates, comparing its five logits bitwise
to ordinary selected-cache decoding. These shorter operator captures do not
replace or relax the 32-token controls.

Scope record-function ranges to source reconstruction and materialization;
restore all wrappers even on failure. Capture CPU/CUDA operators and input
shapes, plus compressed Chrome traces. Full-shape prompt projection GEMMs can
then be distinguished from one-token model GEMMs. Inclusive ranges overlap;
do not sum them or interpret profiler-instrumented durations as normal latency.
Use V4 paired timings for latency and separate workspace-audited runtime
controls for memory. No benchmark annotations are used.

Two one-A100/12-CPU workers, ten-minute limits, maximum six A100s in aggregate,
scratch-only common environment, finite CPU afterany analysis. A profiler pass
is engineering evidence, not a quality pass, new selector, smaller-cache
superiority, independent cohort or paper-campaign authorization.
