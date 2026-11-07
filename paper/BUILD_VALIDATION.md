# Manuscript build validation — 2026-10-22

Built with TeX Live 2025 using the unmodified official MLSys 2025 style
prescribed by the MLSys 2027 call. No margins, font sizes, or style internals
were modified to meet the page limit.

| PDF | Pages | Validation |
| --- | ---: | --- |
| `build/main.pdf` | 11 total | Main content ends on page 9; references occupy pages 10–11; below the 10-content-page limit |
| `build/supplement.pdf` | 16 | Separate appendix, complete numerical curves and extended details |
| `build/thesis.pdf` | 28 | Expanded reading edition; not a submission upload |

All three builds pass `bash paper/scripts/build.sh check`: no undefined or
multiply defined citations/references, overfull boxes, bibliography errors,
duplicate PDF destinations, or author-template errors. Main PDF fonts are
embedded Type 1. The main bibliography contains 25 cited primary-source works.
The MLSys 2027 format requirements were checked on 2026-10-22 against the
official call; the unmodified 2025 style is retained.

Checks include source-result hash verification, complete row counts,
per-sample physical-byte equality in the shared lanes, all three TeX builds,
undefined/multiply-defined references, citations, overfull boxes, bibliography
errors, embedded fonts, anonymous PDF metadata, and main-text page count.
The exporter consumed 3,072 E6 rows and 8,526 PrefixKV rows and verified all
bound aggregate files for the other reported cohorts. This document build
adds no inference or benchmark rescoring.

Main figures are vector TikZ/PGFPlots. Architecture, quality curves, memory
boundary plot, latency plot, and representative appendix tables were rendered
and visually inspected. An initial architecture-box collision and repeated
legend labels were corrected before the final checks. Generated numerical
sources and their hash ledger
ship under `generated/`; online data and models are unnecessary for a PDF
rebuild from the exported tables.
