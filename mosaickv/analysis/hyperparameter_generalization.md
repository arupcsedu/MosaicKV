# Experiment 5: hyperparameter generalization

> Classification: exploratory dirty-tree, tiny-sample validation. These are not paper-scale estimates.

## Protocol

Calibration used only `Qwen/Qwen2.5-VL-3B-Instruct` on `docvqa` samples ['docvqa:58425', 'docvqa:59264']. The deterministic rule selected `hybrid4`. The frozen artifact SHA is `53e1eeebbfd08b9c90b2100878590c39c4fe17bffb570d765f853d6dfb60c598`. Held-out configurations were materialized only after this artifact existed; calibration and evaluation ledgers are disjoint.

## Algorithmic generalization

The architecture-neutral comparison is `mosaickv_exact` versus FullKV with the same frozen forecast, graph, utility, selector, block size, byte ratio, and generation settings. Prototype availability is not allowed to change these rows.

Completed frozen exact model-task cells: 9. 8 had zero task-local score loss and 1 had a nonzero loss. Active-memory reduction ranged from 0.8030 to 0.8164; selected-block recall ranged from 0.6720 to 0.7941. No task scores are macro-averaged.

## Architecture-specific adaptations

Qwen rows labeled `architecture_adapted_full` add the audited M-RoPE prototype transform and guarded residual repair without retuning the frozen core parameters. They are not pooled with architecture-neutral exact rows. In immutable v1, LLaVA remained exact-only because its phase transform had not yet passed the adapter gate. That limitation was resolved afterward and does not alter these historical rows.

Completed adapted model-task cells: 6; 2 had nonzero task-local loss. These rows triggered repair 25 times and are reported separately.

## Backend implementation support

Pure HF eager/SDPA/FlashAttention-2 FullKV rows are compared within method. Compressed SDPA/FA2 rows use eager prefill plus the named decode backend and are labeled that way. At immutable-v1 run time, vLLM/SGLang had only FullKV parity artifacts, so no compressed serving row was emitted. Later jobs `17869843`/`17871006` validate version-pinned whole-block transport, but not yet common-selector equivalence.

Same-sample Experiment-5 backend rows: 6; separate prior serving artifacts: 3.
All direct rows had exact token-sequence parity: True. FullKV same-prefix, full-vocabulary numerical deviation was measured for 3 backend rows; compressed-path logit deviation remains explicitly unavailable.

## Unsupported combinations

- `llava_onevision_7b_optional/*`: In immutable v1, the pinned 7B checkpoint was not cached and had no exact registered runtime gate. It was resolved after the run; no historical row was simulated. (predeclared_capability_gate).
- `vllm/mosaickv_*`: Excluded by the immutable-v1 predeclared capability gate. A later private, version-pinned transport transaction passed, but the common selector-plan bridge remains ungated. (predeclared_capability_gate).
- `sglang/mosaickv_*`: Excluded by the immutable-v1 predeclared capability gate. A later private, version-pinned transport transaction passed, but the common selector-plan bridge remains ungated. (predeclared_capability_gate).
- `hf_sdpa/mosaickv_proto_or_full`: Prototype risk and residual repair require eager attention capture. (predeclared_capability_gate).
- `hf_flash_attention_2/mosaickv_proto_or_full`: Prototype risk and residual repair require eager attention capture. (predeclared_capability_gate).
- `llava_1_5_7b/mosaickv_proto_or_full`: Cached keys are post-RoPE and no audited LLaVA inverse/anchor transform exists. (predeclared_capability_gate).

## Interpretation boundary

Two examples per model-task and one same-sample backend probe can validate plumbing, freezing, leakage controls, and gross transfer failures. They cannot establish population-level cross-model generalization. Failed and unsupported cells remain visible and no incompatible benchmark scores are macro-averaged.
