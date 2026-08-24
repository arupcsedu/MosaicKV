# V102 initial candidate-catalog construction: scope-negative record

Date: 2026-08-22

The two initial V102 candidate catalogs were constructed from clean commit
`bdf45126c995972e0d8f334366b2e2dbfc1eaf3e`. Their serialized content passed
the preregistered schema, pinned-contract, cardinality, uniqueness, canonical
ordering, cross-catalog disjointness, SHA-256, and read-only permission checks.

They are nevertheless **ineligible for ledger materialization**. The installed
Hugging Face `datasets` builder prepared both the requested InfographicVQA
validation split and the unrequested 3,288-example test split. This produced
approximately 3.4 GB of Arrow cache. MosaicKV indexed only validation
`questionId` values and did not use prompts, answers, references, scores,
media, or outcomes for selection, but the builder behavior violated the strict
ID-column-only construction boundary.

## Preserved immutable artifacts

- TextVQA train: 34,602 IDs, 1,165,531 bytes, mode `0444`, SHA-256
  `5a6f700a623da65ae62bf4abc7317dcea48f0a9388f4047d3de8cfd643aece8d`.
- InfoVQA validation: 2,801 IDs, 45,008 bytes, mode `0444`, SHA-256
  `38907d03d42a6c52d898b3a543ad2cd737b85569c465ad301ca148aa72e4308c`.

The artifacts are preserved as negative provenance and must not be overwritten,
selected from, or supplied to the deterministic ledger materializer. Replacement
catalogs require pinned Parquet shard discovery and physical projection of only
the audited ID column, without invoking the dataset builder or preparing any
unrequested split. Ledger materialization, disjointness checking, calibration,
freezing, held-out evaluation, and job submission remain unauthorized.
