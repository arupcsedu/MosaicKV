# Balanced layer allocation: consumed-input algorithm ablation

The completed 400-point-observation representation study did not establish
superiority against LOOK-M. Preserve every point. Eight-bit boundary precision
improved EviCastKV relative to its previous format, but the strongest LOOK-M
point retained higher CIDEr at lower measured storage.

Label-blind capacity audit: EviCastKV/Top-K inherit a PPL-oriented PrefixKV
profile, retaining 505/434/350/411 slots in the four expensive boundary layers
of a representative 603-position input. LOOK-M distributes the same 9,143
total source slots as 285/286 per layer. This is not a source-cap mismatch;
the measured encoded-byte difference partly reflects where those slots reside.

Test one algorithmic change only: deterministically balance EviCastKV's layer
capacities while preserving the exact total source slots/bytes. Give Top-K the
same option. Two formats: unchanged union3, and the already implemented
eight-bit-boundary variant. All selection scores, original K/V, mandatory last
token, logical positions, history, model, prompts and decoding stay fixed.
No text-demand change is combined into this first ablation. Other baselines'
allocations remain their registered algorithms; reuse their complete results,
including every stronger representation point. An empty/infeasible or
heterogeneous geometry fails closed. Balanced allocation has no claim of
optimality or guaranteed quality; it specifically tests profile-transfer cost.

Use only the same 16 consumed COCO images; independent evidence N=0. Add 64
observations (EviCastKV and Top-K, two formats) and reuse all 400 preceding
observations. These are encoded-memory operating points, NOT byte-matched
comparisons. One warmup/one trial; 128 timed predictions and first-EOS scoring.
All common environment/model/processor/reference hashes remain bound. A four
image canary checks both formats, ordinary FullKV parity/reinjection, unchanged
profile-union transport, request isolation, complete storage/timing/provenance.
At most six aggregate A100s, one GPU/12 CPUs/30 minutes per job. Finite afterany
analysis recomputes complete-cohort CIDEr inside 10,000 paired bootstrap draws,
Bonferroni-corrected over 56 comparisons (two candidates versus 28 other
compressed points). No answer labels or sample failure identities enter the
allocator; no independent, larger or paper campaign is automatically enabled.
