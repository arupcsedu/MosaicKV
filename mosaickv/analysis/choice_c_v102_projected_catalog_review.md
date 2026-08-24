# V102 projected candidate-catalog construction review

Date: 2026-08-22

The replacement V102 candidate catalogs were constructed locally on CPU from
clean commit `09697c62699a872bb9044858084c05b601437ddc`. The constructor did
not call `datasets.load_dataset`. It listed files at each pinned dataset
revision, selected only the exact preregistered split-specific Parquet glob,
verified the expected shard count, and called `pyarrow.parquet.read_table`
with only the audited ID column, `use_threads=False`, and `pre_buffer=False`.

## Reviewed immutable artifacts

- TextVQA train: 20 `data/train-*.parquet` shards; 34,602 `question_id`
  values; 1,165,531 bytes; mode `0444`; SHA-256
  `5a6f700a623da65ae62bf4abc7317dcea48f0a9388f4047d3de8cfd643aece8d`.
- InfoVQA validation: four `InfographicVQA/validation-*.parquet` shards;
  2,801 `questionId` values; 45,008 bytes; mode `0444`; SHA-256
  `38907d03d42a6c52d898b3a543ad2cd737b85569c465ad301ca148aa72e4308c`.

Both files contain exactly the six allowed schema fields, use the pinned
dataset ID/config/revision/split contracts, contain unique nonempty string IDs
in canonical order, and have zero cross-catalog duplicate IDs. Their bytes are
identical to the earlier content-valid catalogs, demonstrating content parity
through the narrower projected path.

No new Arrow file, `dataset_info.json`, or dataset state file was produced in
the scratch datasets cache during construction. The code path cannot open an
unrequested split because it accepts only the exact reviewed glob and rejects
an unexpected shard count. The projected construction-scope gate therefore
passes.

This review does **not** authorize ledger materialization. Catalog
construction is consumed and reexecution is disabled. Exclusion scanning,
ledger materialization, disjointness checking, calibration, candidate
freezing, held-out evaluation, Slurm submission, GPU execution, and any
paper-level claim remain blocked pending a separate explicit authorization.
