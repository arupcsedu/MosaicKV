# PrefixKV evidence-preservation ablations, version 1

Status: implementation and CPU checks only. No new pretrained quality result,
GPU benchmark, submission, or paper-reproduction claim is supplied here.
These are **local algorithm variants**, not corrections claimed to reproduce
the original PrefixKV paper. The pinned official checkout is unchanged.

## Motivation and boundaries

The previous 40-MiB allocation diagnostic found large attention mass on the
first token and poor retention of late prompt tokens. Changing layer capacity
alone did not recover TextVQA quality. That suggests two hypotheses: already
protected attention sinks distort optional layer allocation, and the retained
cache loses the question/instructions or evidence important to those queries.
Those observations do not prove causation or justify tuning on evaluation
answers. DocVQA's zero FullKV score in that small run is not evidence of success.

The original implementation remains available with the two new PrefixKV fields
disabled and the existing mandatory policy unchanged. Old profile JSON omits the
new `evidence_policy` object and retains its existing hash representation.
New profiles bind their policy and label-free captured scores/mapping hashes.
New flags default to false/zero; nothing silently changes old runs.

## Approach A: remove protected mass from calibration

For layer `l`, let `s[l,i]` be the existing PrefixKV importance: average prefill
attention over heads in the incoming dtype, then sum over query rows in FP32.
Let `P` contain already-protected positions (normally first and final prompt
tokens). Optional positions are `U = {0,...,N-1} minus P`.

The new `optional_cdf` search normalizes only `s[l,i]` for `i in U`, sorts those
scores, and searches a shared cumulative-mass threshold. The protected positions
are added **once** to each layer capacity, before exact integer apportionment.
Reserve one optional interior position when available for safe decode eviction.
An infeasible mandatory reservation fails; it never expands the byte cap.
Zero optional attention uses a uniform optional distribution, not division by zero.

Configuration: `prefixkv.exclude_protected_from_calibration: true`.
This changes offline layer allocation, not which original attention rows are used
to rank optional positions online. Increasing protected sink mass alone therefore
cannot change A's calibration capacities. This invariance has a synthetic test.

## Approach B: protect the actual question and instructions

Configuration: `cache.physical_budget_mandatory_policy: question_and_boundaries`.
The shared mandatory set is `P = {first, last} union mapped_user_text_positions`.
No arbitrary suffix length substitutes for identifying the actual user text.

The opt-in LLaVA-1.5 mapper:

1. Renders the unchanged processor chat template.
2. Locates each user-text span by independently rendering a unique marker and
   requiring exact reconstruction of the original prompt.
3. Uses the fast tokenizer's character offsets to identify overlapping tokens.
4. Aligns those IDs against the actual processed input, allowing only expansion
   of image-placeholder IDs; image positions themselves are never user text.
5. Checks that every mapped position is text in the adapter's modality map.

It supports a single user turn, optional system text, and image expansion.
Unsupported families, slow tokenizers, transformed text, multi-turn prompts,
unmapped spans, or any unexplained ID mismatch fail explicitly. Token boundaries
overlapping the user text are conservatively included. The mapping records input,
prompt and mapping SHA-256 hashes and positions; it does not rewrite model inputs.

The same mandatory policy applies to FullKV, Prompt Top-K, MosaicKV exact, and
the PrefixKV variant in this comparison group. New control rows are necessary;
old first/last-only controls cannot be reused as identical-policy controls.
Reservation reduces the optional capacity within the same physical cap.

During PrefixKV decoding, the original mapped question and boundary positions
remain protected. If the usual fixed eviction offset is protected, the variant
evicts the closest eligible offset, with lower index breaking ties. The current
tail is also protected. At most one eviction per layer per decode step remains.
This is explicitly **not upstream fixed-offset eviction**. Original logical
positions, post-RoPE keys, and weights are unchanged.

## Approach C: normalized question-query scoring

Let `q[l,i]` average heads and sum only the actual mapped user-query rows of the
complete prefill attention matrix. Normalize `s` and `q` independently over the
optional positions, then combine:

```text
p_original[i] = s[i] / sum_{j in U} s[j], for i in U
p_question[i] = q[i] / sum_{j in U} q[j], for i in U
score[i] = (1 - alpha) * p_original[i] + alpha * p_question[i]
```

Protected entries receive zero ranking score because they are already reserved.
Each zero-mass source becomes uniform over U. `alpha=0.5` is the fixed ablation
hypothesis; it is not selected from evaluation scores. Alpha must be finite,
numeric, and in [0,1]. The mixture is used in offline profiling and online
optional-token ranking. Complete square prefill attention is required: a cropped
last-window tensor cannot masquerade as question attention.

Configuration: `prefixkv.question_score_weight: 0.5`.
Question scoring alone does not make the question mandatory; this permits C-only
and AC ablations to separate evidence ranking from question preservation.

## Incremental comparison, with separate ablations

The bundle contains eight PrefixKV arms and two sets of three matched controls.

| Arm | A: sink-neutral CDF | B: question protected | C: question mixture |
|---|---|---|---|
| control | off | off | off |
| a_sink | on | off | off |
| b_protect | off | on | off |
| c_rank | off | off | 0.5 |
| ab_sink_protect | on | on | off |
| ac_sink_rank | on | off | 0.5 |
| bc_protect_rank | off | on | 0.5 |
| abc_combined | on | on | 0.5 |

Implement/test/execute in order:

1. A versus control, in the first/last-only group.
2. B and AB, with new question-protected FullKV/Top-K/MosaicKV controls.
3. BC versus B and ABC versus AB; C versus control and AC versus A isolate
   ranking without question protection.

Cross-group differences measure a deliberate policy ablation, not an
identical-mandatory-policy method comparison. Do not merge those controls.
Each effective method includes enabled components, e.g.
`prefixkv_sink_neutral_question_protected_question_scored_reimpl`.
Every new profile/trace records that this is a local, non-paper-native variant.

## Budget and evaluation contract

Use the same immutable model/processor revisions, media, prompts, examples,
greedy generation, output limits, precision, eager attention, one-token blocks,
measurement controls, and initial physical cap within each policy group.
The implemented initial lane is HF eager LLaVA-1.5-7B with ragged layer transport;
vLLM and other models/backends are unsupported for these ablations.

For equal layer byte width `b`, allocate the maximum feasible integer position
sum under the shared dense-quantized target. If a target is attainable exactly,
initial selected K/V storage must equal it. Ragged PrefixKV adds no layer padding.
Record requested cap, realized bytes, capacity shortfall, protected counts, and
each layer's positions. Never add protected bytes on top of the cap.

**An initial cache cap is not a constant decode-time cap.** PrefixKV continues
effective-ratio decode eviction; the existing Top-K/MosaicKV paths append generated
tokens. Report initial, per-step and final cache sizes and the decode policy.
Do not call latency a constant-memory-budget comparison. An append-only evidence
variant can be configured separately, but it requires its own label and protocol.
FullKV remains a quality/reference control, not a compressed-latency competitor.

Calibration captures are label-free and disjoint from evaluation sample IDs.
All eight profiles use the same calibration identities and captures. The bundle
validator rejects mismatched model/revision/cap/settings or calibration evidence.
No automatic winner selection or full-campaign promotion is implemented.

Before any quality claim, evaluate every preregistered arm (including failures)
on the same examples, report official task scores and paired intervals against
the corresponding Top-K control, answer/token agreement, retained question/evidence
mass, exact bytes, and informational latency. Keep all original quality and memory
gates; these CPU tests are not evidence that answer quality recovered.

## Preparing ablations (CPU only)

`prefixkv_calibration_observation` now optionally accepts mapped question positions
and the mapping hash. The existing allocation worker's `collect` helper accepts
`capture_question_evidence=True`; its historical CLI still runs the original
allocation diagnostic, not this new experiment. Old captures lacking mapped
question-query scores cannot be reused for B/C or have those scores invented.

Prepare a JSONL of these label-free observations, a reviewed evaluation-ID list,
and `controls.json` mapping `full_kv`, `prompt_attention_topk`, `mosaickv_exact` to
fully resolved `RunConfig` objects. The CPU bundle command is:

```bash
cd /scratch/djy8hg/workdir/MosaicKV
source mosaickv/scripts/cache_env.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prepare_prefixkv_evidence_ablations.py \
  --attention-jsonl /scratch/djy8hg/ABLATION_INPUT/calibration_scores.jsonl \
  --evaluation-sample-ids /scratch/djy8hg/ABLATION_INPUT/evaluation_ids.json \
  --controls-json /scratch/djy8hg/ABLATION_INPUT/controls.json \
  --bytes-per-layer-position 16384 \
  --calibration-dataset DATASET_ID --calibration-revision DATASET_SHA \
  --calibration-split SPLIT \
  --output /scratch/djy8hg/ABLATION_OUTPUT_NEW
```

The paths/identities above are placeholders, not existing experiment inputs.
16384 is the all-head FP16 K+V byte width per layer position for the audited
LLaVA-1.5-7B geometry; validate actual dtype/geometry rather than reusing it blindly.
The command writes eight hash-bound profiles, fourteen resolved configs, and a
bundle manifest with evaluation IDs and file/config hashes. Existing destinations
are rejected. It does not load weights, fetch datasets, select examples, or submit
jobs. GPU and paper-claim authorizations remain false. These RunConfig files feed
the common HF runtime; they are not complete evaluation-job ledgers.

Single-profile generation also supports the three new flags through
`scripts/generate_prefixkv_profile.py`; they require `--byte-cap`.

## Verification scope

Synthetic tests cover sink-mass invariance, mixture normalization/endpoints,
zero mass, mapping ambiguity, image expansion, all eight exact-byte plans,
determinism, incompatible profiles, leakage checks, strict configuration,
shared-policy controls and write-once configuration generation.
Random tiny LLaVA tests check 64-step protected eviction, isolated source caches,
exact storage, complete traces and retention-1/FullKV token equality.
The cached real processor at revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9` was checked on a synthetic image and
question, without loading weights: 576 image tokens expanded correctly and
mapped user text excluded those image positions.
Pretrained GPU quality, latency and logit-parity gates remain unexecuted for
these new variants. No benchmark score is inferred from the CPU checks.
