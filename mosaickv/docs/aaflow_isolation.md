# AAFLOW isolation

MosaicKV does not currently import or vendor AAFLOW or AAFLOW+ source code.
Repository searches must treat the sibling checkout at
`/scratch/djy8hg/workdir/AAFLOW` as an independent project, not an implicit
Python dependency. No MosaicKV launcher may add that checkout to `PYTHONPATH`,
mutate `sys.path` to reach it, or rely on its unpinned worktree.

If a future implementation intentionally reuses AAFLOW code, it must first:

1. identify the exact reusable files and license;
2. pin the source to an immutable commit under `third_party/`, preserving
   attribution and recording local patches, or place an independently written
   compatibility layer inside `src/mosaickv/compat/aaflow/`;
3. expose the functionality through a MosaicKV-owned interface and import it
   with a normal `mosaickv.*` package import;
4. add tests proving that MosaicKV does not depend on the sibling checkout; and
5. record the dependency SHA and license in manifests and artifact docs.

Similar algorithmic structure or documentation references do not by themselves
create a source dependency. Any future reuse must be explicit and auditable.
