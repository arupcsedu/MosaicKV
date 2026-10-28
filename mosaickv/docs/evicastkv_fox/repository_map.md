# Fox-review campaign: implementation map

Public name: **EviCastKV**. Historical identifiers and measured artifacts remain
`mosaickv_exact`; no historical result is renamed or overwritten.

| Component | Actual source / symbol |
|---|---|
| Adapter dispatch | `src/mosaickv/adapters/huggingface/__init__.py::load_hf_adapter` |
| LLaVA / Next | `adapters/huggingface/llava.py`, `llava_next.py` |
| Query / prompt-attention hooks | `adapters/huggingface/base.py::QueryCapture`, `AttentionCapture` |
| Full prefill / one forward | `HuggingFaceMultimodalAdapter.prefill`, `decode_one_token` |
| Prompt-window pooling | `backends/hf_runtime.py::_prompt_token_attention_by_head` |
| Entropy / route counts | `_attention_uncertainty_by_head`, `_uncertain_head_routes` |
| Candidate screening | `_similarity_shortlist` |
| Graph, utility and route-local selection | `build_compression_plan`; `graph/`; `selection/utility.py` |
| Physical-capacity lattice | `_dense_physical_cache_budget`, `_mosaickv_physical_prompt_capacity` |
| Packing and validity | `pack_runtime_cache`, `_pack_exact_runtime_blocks`, `_state_from_packed` |
| Decode loop / traces | `HuggingFaceMosaicKVModel.generate` |
| Unique storage / cache lifecycle | `cache_state.py`; `measurements/cache_lifecycle.py` |
| CUDA phase timing | `measurements/timing.py::CudaEventTimer`, `ModuleCudaTimer` |
| GPU telemetry | `measurements/telemetry.py::capture_gpu_environment` |
| Typed historical manifests | `manifest.py`; `evaluation/worker_support.py::record_hf_run` |
| Materialized media / ledgers | `evaluation/paper_native_lanes.py::load_paper_native_ledger` |
| COCO scorer | `evaluation/coco_shared_bytes.py::TokenizedCider`, `bootstrap_cider_curve` |
| MileBench taxonomy / scoring | `evaluation/milebench.py::route_milebench_dataset`, `score_milebench_sample`, `aggregate_milebench` |
| New contracts / synthetic oracle | `evaluation/fox_campaign.py` |
| New execution | `scripts/evicastkv_campaign.py`, `evicastkv_worker.py`, `evicastkv_analyze.py` |
| Finite scheduler callbacks | `slurm/evicastkv_worker.sbatch`, `evicastkv_completion.sbatch` |

Paths above are relative to `mosaickv/` (package modules under `src/mosaickv/`).

## Recovered policy, not a new search

The authoritative resolved files are the completed COCO October 14 and MileBench
October 18 `configs/validation/mosaickv_exact-40.json` files. Preparation binds
their hashes and the baseline configurations. Their signed utility coefficients
are q=1, v=-0.3, o=-0.3; coverage g=-0.25 and m=-0.25. This follows the existing
objective's signed convention, not a rewrite of its mathematics.

Other controls: window 16, prompt-only forecasting, shortlist 8%, degree 8,
two coverage correction rounds, stable-head prompt routing, graph route fraction
1/256, one-token blocks, unchanged exact K/V, FP16 and HF eager. Entropy ties use
(layer, KV-head) ordering. Route count is max(1, ceil(number of routes*fraction));
the historical implementation cannot express zero through that function. The
new explicitly labeled route-zero control calls the shared Top-K path.

`hf_cli.py::_paper_native_runtime_profile` differs (degree 4, shortlist 3%, one
correction, different capacity/adaptive defaults). It is **not** this campaign's
configuration. No missing coefficient is filled from that helper.

Output scheduling is not early EOS stopping: the runtime performs 64 token
decisions and 63 token-processing forwards, then `decode_quality_output` crops
the scored text at a terminal token. The final emitted token is not yet cached.
Record both quantities; do not call them 64 cache appends.

## Boundaries and unresolved evidence

Batching in the current adapter is batch one (`decode_one_token` requires [1,1]).
Optional batching is NOT_RUN, not an invented throughput result. Requests each
own one allocated A100; multiple workers do not imply model parallelism.

The review's named v1.4 paper and rescue-plan PDFs were not found in the supplied
prompt directory/repository. The complete supplied prompt package is preserved;
configuration facts above were recovered from actual source and run manifests.
No claim of independently checking a missing PDF is made.

Historical inventories are not yet certified exhaustive by reusable media group.
Development inputs are explicitly previously consumed; fresh confirmation remains
blocked until that separate audit and configuration review are complete.
