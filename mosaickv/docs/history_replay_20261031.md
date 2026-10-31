# Same-input lossless history-storage replay

This is not a new independent quality cohort. Keep N=128, not 256. Preserve
the original cohort and every baseline result. The two consumed-input controls
passed the unchanged 32-token guard and complete 128-token transport comparison.
Raw history reduced complete-request latency on those inputs but increased final
KV storage. Do not claim that it is universally superior or that whole-request
allocated peaks fell substantially.

Replay only the unchanged candidate on the existing 128 COCO inputs. Change no
prompt, selector, profile, bit allocation, checkpoint, processor, seed, source-slot
capacity, backend, precision, or 128-prediction schedule. Use five warmups per
worker and one scored complete request per example, as in the parent cohort.
Only replace lossless source-reconstructed generated history with raw lossless
BF16 generated history. The original query3 prompt representation is unchanged.

Before reusing ANY CIDEr result or bootstrap interval, require every replayed
128-token sequence and EOS-scored answer to equal its saved parent prediction.
Require exact paired identities and no missing/failed requests. If any differs,
stop the reuse path, record a negative correctness result, and do not relabel
the prior scores as measurements of the replay. Workers receive no answers or
saved token prefixes; comparison happens only in the dependent CPU analyzer.
Report initial/final/decode-owned peak and whole-request allocator peaks plus
paired complete-request latency anew. Preserve both storage options and their
trade-off. Ordinary BF16 population-level baseline latency is still unverified;
the consumed native controls cannot establish that population-level claim.

At most six aggregate A100s. First replay indices 0--11 use two workers alongside
at most four parent workers. Replay indices 12--15 wait for both the first replay
array and the complete parent GPU array; they use at most four GPUs alongside
the parent's at-most-two timing workers. One GPU/12 CPUs per worker, 30-minute
limit. The CPU analysis waits for both replay arrays and parent analysis. No
fresh calibration or paper campaign is released by this transport replay.
