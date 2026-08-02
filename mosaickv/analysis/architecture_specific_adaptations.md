# Architecture-specific adaptations

> Exploratory dirty-tree evidence from two held-out examples per model/task. The report distinguishes uniform algorithm transfer from implementation support.

## Frozen architecture-neutral algorithm

The uniform comparison uses exact block selection only. Its frozen core is retention `0.2`, block size `16`, `hybrid` forecasting with `4` draft steps, prompt window `16`, and `24` graph neighbors. No model or task was retuned.

- `Qwen2.5-VL-3B` / `mmstar`: task-local loss `0.0000`, active-memory reduction `0.8051`, latency `8.2203s`, forecast cosine `0.8318`, selected recall `0.7731`, repair events `0`.
- `Qwen2.5-VL-3B` / `docvqa`: task-local loss `0.0000`, active-memory reduction `0.8030`, latency `13.7657s`, forecast cosine `0.8351`, selected recall `0.7941`, repair events `0`.
- `Qwen2.5-VL-3B` / `chartqa`: task-local loss `0.5000`, active-memory reduction `0.8162`, latency `4.2584s`, forecast cosine `0.8332`, selected recall `0.7898`, repair events `0`.
- `Qwen2.5-VL-7B` / `mmstar`: task-local loss `0.0000`, active-memory reduction `0.8051`, latency `11.1336s`, forecast cosine `0.8591`, selected recall `0.7357`, repair events `0`.
- `Qwen2.5-VL-7B` / `docvqa`: task-local loss `0.0000`, active-memory reduction `0.8030`, latency `19.1301s`, forecast cosine `0.8490`, selected recall `0.7536`, repair events `0`.
- `Qwen2.5-VL-7B` / `chartqa`: task-local loss `0.0000`, active-memory reduction `0.8162`, latency `5.7709s`, forecast cosine `0.8408`, selected recall `0.7755`, repair events `0`.
- `LLaVA-1.5-7B` / `mmstar`: task-local loss `0.0000`, active-memory reduction `0.8072`, latency `36.3787s`, forecast cosine `0.8705`, selected recall `0.7882`, repair events `0`.
- `LLaVA-1.5-7B` / `docvqa`: task-local loss `0.0000`, active-memory reduction `0.8109`, latency `34.8575s`, forecast cosine `0.8779`, selected recall `0.7748`, repair events `0`.
- `LLaVA-1.5-7B` / `chartqa`: task-local loss `0.0000`, active-memory reduction `0.8164`, latency `35.7833s`, forecast cosine `0.9000`, selected recall `0.6720`, repair events `0`.

## Model-specific cache semantics

- Qwen2.5-VL uses multimodal rotary positions. Its optional adapted rows use the audited M-RoPE anchor-frame prototype transform and eager-only attention capture for guarded residual repair.
- In immutable v1, LLaVA-1.5 remained exact-selection only because its phase transform had not yet passed; no prototype or repair rows were emitted.
- In immutable v1, the optional LLaVA-OneVision-7B checkpoint was neither cached nor registered, so no row was simulated.
- After v1, exact Llama/Qwen2 phase transforms and uniform-BF16 OneVision-7B loading passed exploratory A100 parity, real-prototype, and repair gates. These new gates do not rewrite the historical experiment.

## Adapted Qwen outcome

- `Qwen2.5-VL-3B` / `mmstar`: task-local loss `0.0000`, latency `50.2367s`, repair events `4`.
- `Qwen2.5-VL-3B` / `docvqa`: task-local loss `0.0000`, latency `103.9248s`, repair events `5`.
- `Qwen2.5-VL-3B` / `chartqa`: task-local loss `0.0000`, latency `18.3749s`, repair events `6`.
- `Qwen2.5-VL-7B` / `mmstar`: task-local loss `0.0000`, latency `78.8819s`, repair events `1`.
- `Qwen2.5-VL-7B` / `docvqa`: task-local loss `0.5000`, latency `164.4281s`, repair events `3`.
- `Qwen2.5-VL-7B` / `chartqa`: task-local loss `0.5000`, latency `28.8079s`, repair events `6`.

On this tiny slice, the adaptation recovered the Qwen-3B ChartQA loss but introduced losses on Qwen-7B DocVQA and ChartQA and incurred substantial diagnostic-enabled latency. It therefore does not establish that prototype/repair hyperparameters generalize; the architecture-neutral exact rows are the valid cross-family result.
LLaVA's zero losses on DocVQA and ChartQA are floor-limited because its matching FullKV rows also scored zero; they are parity checks, not positive quality evidence.

## Backend support versus algorithm transfer

HF eager supports the complete audited Qwen path. HF SDPA and FlashAttention-2 passed same-sample token parity for FullKV and for exact selection with eager prefill plus fused decode. They do not support prototype-risk attention capture or repair in this implementation. At immutable-v1 run time, vLLM and SGLang had FullKV parity only, so no compressed serving row was emitted. Later jobs `17869843`/`17871006` validate version-pinned whole-block transport, but not common-selector equivalence or prototype/repair support.

Direct HF parity rows: `6`. Explicit unsupported combinations: `6`. FullKV same-prefix, full-vocabulary logit deviation was measured for `3` HF rows; compressed serving rows retain token parity only. All direct probes had token-sequence parity `True`.
