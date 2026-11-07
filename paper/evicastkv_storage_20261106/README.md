# Current EviCastKV storage-variant paper — 6 November 2026

This is a new, complete draft based on the completed 256-image COCO storage
study. It preserves the older manuscript and historical reports elsewhere
under `paper/`. **It does not silently reuse the historical six-component
design as the evaluated algorithm.**

## Read these documents

| File | Purpose |
| --- | --- |
| [build/main.pdf](build/main.pdf) | Anonymous MLSys-format paper: abstract, introduction, motivation, formulation, design, implementation, evaluation, discussion, related work and conclusion |
| [build/appendix.pdf](build/appendix.pdf) | Separate detailed supplement: representation, correctness, phases, accounting, statistics, full configuration, source map and limitations |
| [build/fox_response.pdf](build/fox_response.pdf) | Private, detailed point-by-point response to the available Prof. Fox review-derived requirements |
| [build/complete_reading_copy.pdf](build/complete_reading_copy.pdf) | Combined paper, supplement and named private response; **not** an anonymous submission file |
| [generated/provenance.json](generated/provenance.json) | SHA-256 bindings for every numerical source, worker manifest/raw artifact, generated table and plot dataset |

The response is **paraphrased from the supplied experiment prompts**, not a
verbatim quotation of the original review PDF, which was not available in
the inspected files. Keep the named private response separate from an
anonymous submission. The original instructions requested exact prompt
selection; this storage-variant study is a changed scope, not completion of
that original experiment matrix.

## What the draft reports

- One immutable source: `a6f689fcdfae2b9023045aa412c6d90d8ce79012`.
- Pinned LLaVA-1.5-7B, BF16 weights, HF eager, batch one, A100-SXM4-80GB.
- 256 paired COCO images, common 128-prediction/127-append schedule.
- Six explicitly labeled operating points; all original six mechanisms inactive.
- Measured initial/final storage and audited **decode-owned** peak, corpus
  CIDEr, 256-image request-time distributions, and separate repeated timings.
- 124 completed GPU workers; no missing accepted rows or FullKV fallback.
- The largest baseline adaptation quality caveats, actual replay eligibility,
  caption-limit hits and familywise-adjusted uncertainty are visible.

The source format has smaller observed storage and lower measured request
time than these compressed adaptations. It **does not establish** universal
quality preservation, a 30% advantage over the strongest compressed baseline,
byte-matched superiority, untouched confirmation, whole-request peak
certification, official baseline reproduction or six-component performance.
Lossless generated-history coding is relative to this variant's own forward
passes; the prompt source remains approximate.

The manuscript is a submission-*format* draft, not a certification that the
remaining empirical limitations satisfy a conference's acceptance standard.
The human authors must review all text, attribution, interpretation,
authorship, anonymity and the AI-usage disclosure before submission.

## Reproduce tables and build

No models, packages or GPU jobs are installed or launched by these commands.
All caches remain on scratch. From the repository root:

```bash
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python \
  paper/evicastkv_storage_20261106/scripts/export_results.py
bash paper/evicastkv_storage_20261106/build.sh
/scratch/djy8hg/env/mosaickv/bin/python \
  paper/evicastkv_storage_20261106/scripts/verify_paper.py
```

The exporter checks raw producer hashes, complete method/sample grids,
pairing, statuses, tokens, endpoints and statistics. It does not rewrite run
artifacts. Intervals come from the completed 10,000-draw paired
corpus-CIDEr bootstrap; storage and latency summaries are independently
recalculated from raw rows. Rerunning the export is unnecessary to rebuild
the already included measured tables on another machine.

The build uses the unmodified official style in `../vendor/` and TeX Live
2025 when the cluster module is needed. The [MLSys 2027
instructions](https://mlsys.org/Conferences/2027/CallForResearchPapers)
specify the 2025 style, ten main-text pages excluding references, a separate
appendix, all bibliography authors and an AI-usage disclosure after references.
The generated build-validation record reports the actual main-text page count.

For a standalone TeX editor/Overleaf project, copy this directory together
with the unmodified style/dependency files from `../vendor/`, or configure
`TEXINPUTS`/`BSTINPUTS` as in `build.sh`. Select `main.tex`, `appendix.tex` or
`fox_response.tex`. All diagrams and plots are native TikZ/PGFPlots; the
data are measured discrete operating points, not fabricated curves.

## Source evidence

Canonical experiment root:
`/scratch/djy8hg/runs/evicastkv/coco-256-all-methods-20261106-v3`.

The `plan.json`, `scope.json`, `ledger.json`, `ledger_seal.json`, phase
`analysis/decision.json`, each worker's `manifest.json`, `result.json` and
`raw.jsonl`, and final `summary/summary.json` are bound in the provenance
export. Review-derived instructions are under
`mosaickv/evicastkv_experiment_prompts/` and separately hash-bound.
No dataset images, model weights, tokens or credentials are copied into
the document project.
