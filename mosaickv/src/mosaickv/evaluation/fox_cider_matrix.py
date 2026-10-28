"""CPU-only sparse evaluation of the pinned COCO CIDEr formula.

The official PTB tokenizer and ``precook`` remain the input boundary. Only
ngram counts are cached: document frequency and every TF-IDF norm are rebuilt
for *each* bootstrap draw, including duplicate occurrences. This is not an
average of per-image scores computed with a fixed reference corpus.

Reference: installed pycocoevalcap/cider/cider_scorer.py, CiderScorer
counts2vec/sim/compute_cider; lmms_eval.tasks.coco_cap.utils.Cider, multiplied
by 100 as in the existing corpus reducer. FP64 sparse sums can differ in the
last few ulps from the official Python loop; regression tolerance is 1e-10
absolute CIDEr points, not a change to any model/logit gate.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider


class FrozenCiderMatrix:
    """Score a fixed, aligned set of predictions on arbitrary paired draws."""

    def __init__(self, points, *, tokenizer=None):
        from pycocoevalcap.cider.cider_scorer import precook

        if not points or not points[0] or any(len(p) != len(points[0]) for p in points):
            raise ValueError("CIDEr requires nonempty aligned corpus points")
        self.size = len(points[0])
        self.point_count = len(points)
        ids = tuple(s.sample_id for s in points[0])
        refs = tuple(s.references for s in points[0])
        if len(set(ids)) != self.size or any(
            tuple(s.sample_id for s in p) != ids or tuple(s.references for s in p) != refs
            for p in points
        ):
            raise ValueError("CIDEr point identities or reference annotations differ")
        tokenizer = tokenizer or TokenizedCider([s for p in points for s in p])
        tokens = tokenizer._tokens
        hypotheses = [
            precook(tokens[(s.prediction, s.references)][1][0]) for p in points for s in p
        ]
        reference_counts, owners = [], []
        for i, sample in enumerate(points[0]):
            reference_tokens = tokens[(sample.prediction, sample.references)][0]
            if not reference_tokens:
                raise ValueError("CIDEr requires at least one reference per image")
            reference_counts.extend(precook(text) for text in reference_tokens)
            owners.extend([i] * len(reference_tokens))
        self.owners = np.asarray(owners, dtype=np.int64)
        self.reference_count = len(owners)
        self.references_per_image = np.bincount(self.owners, minlength=self.size)
        vocabulary = {
            gram: index
            for index, gram in enumerate(
                sorted({g for counts in (*hypotheses, *reference_counts) for g in counts})
            )
        }
        self.vocabulary_size = len(vocabulary)

        def sparse(rows, columns, values, count):
            return csr_matrix(
                (values, (rows, columns)), shape=(count, self.vocabulary_size), dtype=np.float64
            )

        def norms(counts_list):
            rows, columns, values = [], [], []
            for index, counts in enumerate(counts_list):
                for gram, frequency in counts.items():
                    rows.append(index * 4 + len(gram) - 1)
                    columns.append(vocabulary[gram])
                    values.append(float(frequency**2))
            return sparse(rows, columns, values, len(counts_list) * 4)

        self.hypothesis_norms = norms(hypotheses)
        self.reference_norms = norms(reference_counts)
        presence_rows, presence_columns = [], []
        for i in range(self.size):
            grams = {
                gram for r, owner in enumerate(owners) if owner == i for gram in reference_counts[r]
            }
            presence_rows.extend([i] * len(grams))
            presence_columns.extend(vocabulary[g] for g in grams)
        self.presence = sparse(
            presence_rows, presence_columns, np.ones(len(presence_rows)), self.size
        )
        rows, columns, values = [], [], []
        gaussian = []
        for p in range(self.point_count):
            for r, owner in enumerate(owners):
                hyp, ref = hypotheses[p * self.size + owner], reference_counts[r]
                # The pinned evaluator's length is BIGRAM count, including zero
                # for empty/single-word captions, not the unigram token count.
                length_hyp = sum(v for g, v in hyp.items() if len(g) == 2)
                length_ref = sum(v for g, v in ref.items() if len(g) == 2)
                gaussian.append(np.e ** (-(float(length_hyp - length_ref) ** 2) / 72.0))
                for gram, frequency in hyp.items():
                    reference_frequency = ref.get(gram, 0)
                    if reference_frequency:
                        rows.append((p * self.reference_count + r) * 4 + len(gram) - 1)
                        columns.append(vocabulary[gram])
                        values.append(
                            float(min(frequency, reference_frequency) * reference_frequency)
                        )
        self.numerators = sparse(rows, columns, values, self.point_count * self.reference_count * 4)
        self.gaussian = np.asarray(gaussian).reshape(self.point_count, self.reference_count)

    def score_draws(self, draws, *, batch_size=32):
        """Return [draw, point] scores; repeated IDs retain their multiplicity."""
        if batch_size < 1:
            raise ValueError("CIDEr batch size must be positive")
        result = []
        for start in range(0, len(draws), batch_size):
            batch = draws[start : start + batch_size]
            if any(not draw or any(i < 0 or i >= self.size for i in draw) for draw in batch):
                raise ValueError("empty or out-of-range CIDEr resample")
            counts = np.asarray(
                [np.bincount(draw, minlength=self.size) for draw in batch], dtype=float
            )
            total = counts.sum(axis=1)
            df = np.asarray(self.presence.T @ counts.T)
            idf_squared = (np.log(total)[None, :] - np.log(np.maximum(1.0, df))) ** 2
            width = len(batch)
            hyp = np.sqrt(self.hypothesis_norms @ idf_squared).reshape(
                self.point_count, self.size, 4, width
            )
            ref = np.sqrt(self.reference_norms @ idf_squared).reshape(
                self.reference_count, 4, width
            )
            numerator = (self.numerators @ idf_squared).reshape(
                self.point_count, self.reference_count, 4, width
            )
            denominator = hyp[:, self.owners] * ref[None]
            similarity = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
            )
            pair_scores = similarity.mean(axis=2) * self.gaussian[:, :, None]
            weights = counts[:, self.owners].T / self.references_per_image[self.owners, None]
            scores = (pair_scores * weights[None]).sum(axis=1) * 1000.0 / total[None, :]
            result.append(scores.T)
        output = np.concatenate(result, axis=0) if result else np.empty((0, self.point_count))
        if not np.isfinite(output).all():
            raise ValueError("nonfinite sparse CIDEr result")
        return output
