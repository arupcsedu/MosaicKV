# Prompt 6 — Test graph participation, scoring components and routing controls

Apply 00_MASTER.md. Require valid common runtime and allocation. Use ONLY the 64 development inputs per workload, with budgets COCO [20,40] and MileBench [16,40] MiB. All counts and rules are new protocol choices, not historical facts.

A. Route-fraction screening.
Run the recovered current deterministic routing policy at fractions [0, 1/256, 1/16, 1/4]. Fraction 0 must bypass graph construction and its outputs must be checked against shared-runtime Top-K; do not assume they are identical. Recover/define the count rounding explicitly. Pilot fraction 1 on 8 development inputs and expand only if the prespecified resource cap permits it. No result-based choice of which examples finish. Hold weights, window, shortlist fraction, degree, protection, allocation semantics and generation fixed. Report actual routed pairs, candidate count and bytes. If the original policy cannot rank additional heads, document a simple deterministic extension as a NEW policy rather than pretending it was already implemented.

B. Same-candidate scoring ablations at a development-selected fraction r* (or the preregistered diagnostic default 1/16 if no selection has been authorized).
Use the same prompt observations, routed head IDs, candidate universe, graph descriptors and per-head capacities in a labeled fixed-candidate/fixed-capacity mechanism study:
  complete: recovered coefficients unchanged;
  routed_prompt_control: attention utility only, value/novelty/graph/modality coefficients zero;
  coverage_off: graph and modality coverage coefficients zero, value/novelty unchanged;
  value_novelty_off: value and novelty coefficients zero, coverage unchanged.
Do not retune coefficients in ablated variants. Do not call coverage_off 'no graph': novelty/candidate formation may still need it. Keep individual signal normalization and score domains fixed. If the deployment allocator is treatment-dependent, freeze a documented prompt-derived feasible quota plan for ALL variants and run a complete-policy control under that plan. Label results as that controlled study; do not substitute its timing for native deployment timing.

C. If pursuing entropy routing, use a declared normalized entropy over valid prompt attention: mean(-sum p*log(p)/log(n_valid)), with 0 log 0=0 and entropy 0 when n_valid<=1; aggregate mapped query heads within each KV head. Use no future token observations and no renormalization over shortlists. Compare entropy-selected heads with random heads at the SAME routed count and layer quotas, using seeds [13,29,47]. Measure entropy computation cost. Existing deterministic routing remains a comparator. Do not pool seeds as independent task examples.

D. If making an inter-modal-specific benefit claim, add a bounded cross_modal_edges_off control at r*: remove cross-modal edges while freezing candidate IDs, quotas and all other settings. Report whether the ablation actually changes edges, scores or selections. If no cross-modal edges were active, explicitly mark the claim unsupported; do not fabricate modality metadata.

Save native quality, planning and full-request time, memory, selected-set overlap, raw/marginal score terms, realized graph degrees and active edge-source counts. Profile and quality-only cached diagnostics are not headline timing rows.

Deliver routing_sweep.csv, component_ablation.csv, routing_controls.csv (when run), configuration_activity.json, development_predictions.jsonl and development_decision.md. Include every scheduled condition, failure and timeout. Write a Pareto/trade-off report without combining CIDEr, Accuracy and ROUGE-L into one score. Do not select r* using confirmation data, p-values alone, or best random seed. A human-reviewed selected_config.json and a prespecified claim/analysis family are required before confirmation; otherwise report NO_CONFIG_FROZEN and stop.
