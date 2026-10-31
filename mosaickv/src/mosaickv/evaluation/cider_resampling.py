"""Exact CPU corpus CIDEr resampling with sparse, draw-specific TF-IDF.

This is an analysis-only acceleration of pycocoevalcap==1.2's CiderScorer.
PTB tokens and n-gram counts are cached, NEVER per-image CIDEr scores. Each
resample rebuilds reference document frequencies, IDF weights, norms, clipped
dot products, and the corpus mean. Repeated identities count as repeated
documents, matching the official evaluator with fresh occurrence IDs.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
from mosaickv.evaluation.metrics.types import MetricSample

# Analysis-local annotation, also compatible with the older frozen inference
# checkout where no Point alias exists. It never affects metric arithmetic.
Point: TypeAlias = tuple[str, float | None]


def paired_draw_counts(size: int, resamples: int, seed: int) -> NDArray[Any]:
    """Same sorted-ID, Python Random draws as bootstrap_cider_curve."""
    if size < 2 or resamples < 1 or seed < 0:
        raise ValueError("need at least two samples, positive resamples, nonnegative seed")
    rng = random.Random(seed)
    return np.asarray(
        [
            np.bincount([rng.randrange(size) for _ in range(size)], minlength=size)
            for _ in range(resamples)
        ],
        dtype=np.int64,
    )


class ResampledCider:
    """Evaluate complete paired corpora, not a mean of frozen contributions.

    Official formula: TF * (log(N) - log(max(1, DF))); clipped hypothesis-
    reference cosine, bigram-count Gaussian length penalty (sigma=6), mean
    over four orders and all references, times 1000 for repository score points.
    IDF is nonnegative, so clipped products have coefficient min(h, r)*r.
    """

    def __init__(self, groups: Mapping[Point, Sequence[MetricSample]]) -> None:
        from pycocoevalcap.cider.cider_scorer import precook
        from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

        if not groups:
            raise ValueError("empty CIDEr groups")
        self.points = tuple(sorted(groups, key=str))
        indexed = {point: {s.sample_id: s for s in samples} for point, samples in groups.items()}
        self.sample_ids = tuple(sorted(indexed[self.points[0]]))
        if len(self.sample_ids) < 2:
            raise ValueError("CIDEr resampling needs at least two paired samples")
        for point in self.points:
            if len(indexed[point]) != len(groups[point]):
                raise ValueError("duplicate CIDEr sample ID")
            if tuple(sorted(indexed[point])) != self.sample_ids:
                raise ValueError("CIDEr points require identical sample IDs")
        self.ordered = tuple(tuple(indexed[p][sid] for sid in self.sample_ids) for p in self.points)
        for column in zip(*self.ordered, strict=True):
            if len({s.references for s in column}) != 1 or not column[0].references:
                raise ValueError("paired references differ or are empty")
        self.official = TokenizedCider([s for point in self.ordered for s in point])
        # Reuse only the frozen official PTB tokens, not official metric scores.
        token_pairs = self.official._tokens
        ngrams: dict[tuple[str, ...], int] = {}
        norm_rows: list[int] = []
        norm_cols: list[int] = []
        norm_data: list[float] = []
        count_vectors: list[dict[Any, int]] = []
        lengths: list[int] = []
        document_rows: list[int] = []
        document_cols: list[int] = []

        def register(counts: dict[Any, int]) -> int:
            index = len(count_vectors)
            count_vectors.append(counts)
            lengths.append(sum(tf for gram, tf in counts.items() if len(gram) == 2))
            for gram, tf in counts.items():
                gid = ngrams.setdefault(gram, len(ngrams))
                norm_rows.append(index * 4 + len(gram) - 1)
                norm_cols.append(gid)
                norm_data.append(float(tf * tf))
            return index

        references = []
        for i, sample in enumerate(self.ordered[0]):
            ref_tokens = token_pairs[sample.prediction, sample.references][0]
            refs = [precook(text) for text in ref_tokens]
            references.append([register(counts) for counts in refs])
            for gram in {gram for counts in refs for gram in counts}:
                document_rows.append(ngrams[gram])
                document_cols.append(i)
        dot_rows: list[int] = []
        dot_cols: list[int] = []
        dot_data: list[float] = []
        hypothesis_indices: list[int] = []
        reference_indices: list[int] = []
        owner_rows: list[int] = []
        owner_cols: list[int] = []
        owner_data: list[float] = []
        length_penalties: list[float] = []
        for p, samples in enumerate(self.ordered):
            for i, sample in enumerate(samples):
                hyp_tokens = token_pairs[sample.prediction, sample.references][1]
                if len(hyp_tokens) != 1:
                    raise ValueError("CIDEr expects one hypothesis per image")
                hid = register(precook(hyp_tokens[0]))
                for rid in references[i]:
                    pair = len(hypothesis_indices)
                    hypothesis_indices.append(hid)
                    reference_indices.append(rid)
                    delta = lengths[hid] - lengths[rid]
                    length_penalties.append(float(np.exp(-(delta * delta) / 72.0)))
                    owner_rows.append(p * len(self.sample_ids) + i)
                    owner_cols.append(pair)
                    owner_data.append(1000.0 / (4 * len(references[i])))
                    for gram, htf in count_vectors[hid].items():
                        rtf = count_vectors[rid].get(gram, 0)
                        if rtf:
                            dot_rows.append(pair * 4 + len(gram) - 1)
                            dot_cols.append(ngrams[gram])
                            dot_data.append(float(min(htf, rtf) * rtf))
        self.reference_presence = csr_matrix(
            (np.ones(len(document_rows)), (document_rows, document_cols)),
            shape=(len(ngrams), len(self.sample_ids)),
        )
        self.norm_coefficients = csr_matrix(
            (norm_data, (norm_rows, norm_cols)), shape=(len(count_vectors) * 4, len(ngrams))
        )
        self.dot_coefficients = csr_matrix(
            (dot_data, (dot_rows, dot_cols)), shape=(len(hypothesis_indices) * 4, len(ngrams))
        )
        self.pair_owners = csr_matrix(
            (owner_data, (owner_rows, owner_cols)),
            shape=(len(self.points) * len(self.sample_ids), len(hypothesis_indices)),
        )
        self.hypothesis_indices = np.asarray(hypothesis_indices)
        self.reference_indices = np.asarray(reference_indices)
        self.penalties = np.asarray(length_penalties)

    def score(self, counts: NDArray[Any], *, batch_size: int = 32) -> NDArray[Any]:
        """Return draw-by-point corpus scores with all TF-IDF rebuilt per draw."""
        counts = np.asarray(counts)
        if (
            counts.ndim != 2
            or counts.shape[1] != len(self.sample_ids)
            or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any()
            or (counts.sum(axis=1) != len(self.sample_ids)).any()
            or batch_size < 1
        ):
            raise ValueError("draw counts must be nonnegative integers totaling cohort size")
        if not len(counts):
            return np.empty((0, len(self.points)))
        output = []
        for start in range(0, len(counts), batch_size):
            weights = counts[start : start + batch_size].T
            # Crucially DF and all resulting similarities depend on this draw.
            df = self.reference_presence @ weights
            idf_squared = (np.log(len(self.sample_ids)) - np.log(np.maximum(1.0, df))) ** 2
            norm = np.sqrt(self.norm_coefficients @ idf_squared).reshape(-1, 4, weights.shape[1])
            dot = (self.dot_coefficients @ idf_squared).reshape(-1, 4, weights.shape[1])
            denominator = norm[self.hypothesis_indices] * norm[self.reference_indices]
            similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
            pair_scores = similarity.sum(axis=1) * self.penalties[:, None]
            image_scores = (self.pair_owners @ pair_scores).reshape(
                len(self.points), len(self.sample_ids), weights.shape[1]
            )
            output.append((image_scores * weights[None, :, :]).sum(axis=1).T / len(self.sample_ids))
        return np.concatenate(output)

    def validate_official(self, counts: NDArray[Any], scores: NDArray[Any]) -> float:
        """Compare entire resampled corpora against the pinned official reducer."""
        maximum = 0.0
        for draw, observed in zip(counts, scores, strict=True):
            indices = np.repeat(np.arange(len(self.sample_ids)), draw)
            for point, value in zip(self.ordered, observed, strict=True):
                expected = self.official([point[int(i)] for i in indices])
                error = abs(expected - value)
                maximum = max(maximum, error)
                if error > 1e-9:
                    raise ValueError(f"corpus CIDEr does not match official reducer: {error}")
        return maximum
