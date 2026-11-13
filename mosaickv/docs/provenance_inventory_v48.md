# V48: inventory remaining provenance, not repeat completed JSON proofs

V47 bulk job 22925911 and analyzer 22925912 passed. All 1,168 remaining JSON
transactions qualified. The final immutable checkpoint resolves 40,625 of
40,760 historical occurrences. Its 135 unresolved occurrences comprise 134
nonempty Parquet gaps and one protected credential fixture. No JSON source
retry is justified, including the original 32 completed proofs.

The remaining Parquet families are not the previously reviewed closed graph
writers. A filename, footer schema, or today's auditor SHA cannot establish
the historical producer. Repeating their existing unsupported proof attempt
would add no evidence.

This continuation therefore reads only four hash-bound V47 metadata artifacts.
It matches each residual occurrence to its original owner, ordinal, error hash,
path, operation, and registration. It inventories existing exact artifact-hash
bindings without inventing producer commits. It does not open historical JSON,
Parquet bodies, credential bodies, benchmark labels, models, or datasets.
An artifact hash alone remains insufficient: subsequent closed-writer proof
work needs independently recovered original producer/artifact linkage and an
audit of that actual historical writer. Missing records remain unresolved.

All limits remain unchanged: 210 seconds per inventory transaction, 2 GiB
input, 8 MiB live projected metadata, 100,000 projected nodes, 512 MiB per
checkpoint, 64 GiB aggregate artifacts, at most 32 GiB CPU RAM and eight worker
processes. Reading the already-qualified checkpoint is not another historical
source parse. Its compact identity encoding and complete receipts are left
untouched; no original checkpoint is rewritten or upgraded to a freshness pass.

Finite CPU chain: complete current-source tests, inventory remaining provenance,
then run afterany analysis even if inventory fails. No follow-up grants
confirmation, materialization, inference, GPU, or paper-campaign permission.
