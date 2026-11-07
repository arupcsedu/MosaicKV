# MosaicKV MLSys submission draft

> **Current storage-variant draft (6 November 2026):** see
> [evicastkv_storage_20261106/README.md](evicastkv_storage_20261106/README.md).
> It contains the new complete paper, detailed supplement and private
> point-by-point Fox response based on the latest completed COCO study.
> The manuscript described below is preserved historical work; its
> exact-selection results and claim scope must not be transferred to
> the newer shared-source precision/storage variant.

The main manuscript emphasizes measured quality at shared physical KV memory,
request latency, and the limits of lower-memory quality equivalence. Every
reported MosaicKV result uses `mosaickv_exact`. FullKV is a quality reference,
not a latency competitor. Published-method ports are labeled reimplementations.

| File | Purpose |
| --- | --- |
| [build/main.pdf](build/main.pdf) | Anonymous MLSys manuscript; main content within 10 pages, references following |
| [build/supplement.pdf](build/supplement.pdf) | Separate appendix: complete curves, negative tests, algorithms and implementation details |
| [build/thesis.pdf](build/thesis.pdf) | Expanded reading edition, not the conference upload |
| [generated/provenance.json](generated/provenance.json) | SHA-256 bindings for every consumed numerical source |

The official [MLSys 2027 call](https://mlsys.org/Conferences/2027/CallForResearchPapers)
uses the 2025 style, permits 10 main-text pages excluding references, requires
all authors in bibliography entries, and requests a separately uploaded appendix.
The unmodified style and dependencies are under `vendor/`.

## Results included

- COCO Caption: 256 images, 10/20/40/80 MiB, FullKV, Prompt Top-K,
  `vl_cache_reimpl`, and MosaicKV exact; corpus CIDEr and request timing.
- MileBench E6: 192 examples over four subtasks, 16/20/28/40/80 MiB;
  Accuracy and ROUGE-L, matched bytes, latency, and registered cross-budget tests.
- Earlier MileBench: 256 examples, 10/20/40/80 MiB, preserved separately.
- PrefixKV: 1,000 LLaVA-Description and 218 paper-style MM-Vet examples;
  BF16 native-retention and separate FP16 shared-40-MiB lanes;
  corpus PPL, FullKV-reference ROUGE-L, initial/final/peak cache storage.

The main draft highlights fixed-memory gains and explicitly reports close
Prompt Top-K comparisons, negative lower-memory tests, native PrefixKV's
lower final storage, and limitations of baseline ports. It does not claim
universal superiority, production serving speedups, or active prototype/repair
benefits. Mechanistic explanations are distinguished from causal ablations.
Earlier nominal-ratio studies remain historical evidence and are not presented
as equal-byte comparisons.

## Reproduce the document

From the repository root on the cluster:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
/scratch/djy8hg/env/mosaickv/bin/python paper/scripts/export_results.py
source /etc/profile.d/modules.sh
module load texlive/2025
bash paper/scripts/build.sh
bash paper/scripts/build.sh check
```

Export reads completed campaign artifacts, verifies bound hashes, independently
checks numerical completeness and shared bytes, and writes small table/plot
sources under `generated/`. It never performs GPU inference or rewrites raw
results. The numerical exports are already included, so only the TeX build is
needed to rebuild the PDFs elsewhere. Source paths needed for re-export are
recorded in `SOURCE_MAP.md` and the provenance ledger.

All figures are original TikZ/PGFPlots vector sources using measured data;
connecting lines do not count as interpolated evidence. The Makefile sets
`TEXINPUTS` and `BSTINPUTS` for the official vendored style. For Overleaf,
upload the complete directory, move/copy the unmodified files in `vendor/`
into the project root or set the search paths, and select `main.tex`.

Document builds install nothing and keep TeX caches under scratch. No new GPU
campaign or commit is part of this paper edit. Before submission, the authors
must review scientific accuracy, authorship, and anonymity; the PDF is an
anonymous submission-format draft, not a certification of unvalidated claims.
