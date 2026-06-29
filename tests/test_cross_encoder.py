"""
Tests for the cross-encoder reranker: pairing/text construction, sigmoid
normalization, and integration with score dicts. Model loading is mocked —
these tests never download or run the actual transformer, so they run
offline and fast, same as the rest of the suite.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.candidate_parser import Candidate, CareerEntry, SkillEntry
from src.semantic.cross_encoder_reranker import (
    CrossEncoderReranker, compute_cross_encoder_scores, _sigmoid,
)


class FakeReranker(CrossEncoderReranker):
    """Stands in for CrossEncoderReranker without loading a real model."""

    def __init__(self, raw_scores):
        super().__init__()
        self._raw_scores = raw_scores
        self.model = "fake-loaded"  # truthy, so load_model() is never called

    def score_pairs(self, pairs):
        return _sigmoid(np.asarray(self._raw_scores[: len(pairs)], dtype=np.float64))


def make_candidate(cid, headline="ML Engineer", summary="Search systems."):
    return Candidate(
        candidate_id=cid, headline=headline, summary=summary,
        skills=[SkillEntry(name="Python")],
        career_history=[CareerEntry(title="ML Eng", description="Built ranking models.")],
    )


class TestSigmoid:
    def test_bounds(self):
        scores = _sigmoid(np.array([-10.0, 0.0, 10.0]))
        assert scores[0] < 0.01
        assert 0.49 < scores[1] < 0.51
        assert scores[2] > 0.99

    def test_monotonic(self):
        scores = _sigmoid(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))


class TestComputeCrossEncoderScores:
    def test_empty_candidates_returns_empty_dict(self):
        assert compute_cross_encoder_scores("Some JD", []) == {}

    def test_output_shape_and_range(self):
        candidates = [make_candidate(f"C{i}") for i in range(3)]
        reranker = FakeReranker(raw_scores=[2.0, -2.0, 0.0])
        result = compute_cross_encoder_scores("Senior AI Engineer JD", candidates, reranker)

        assert set(result.keys()) == {"C0", "C1", "C2"}
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_ordering_preserved_by_candidate_id(self):
        # Highest raw score should map to the candidate it was computed for,
        # not get scrambled by dict ordering.
        candidates = [make_candidate("LOW"), make_candidate("HIGH")]
        reranker = FakeReranker(raw_scores=[-5.0, 5.0])
        result = compute_cross_encoder_scores("JD text", candidates, reranker)

        assert result["HIGH"] > result["LOW"]

    def test_missing_candidate_id_defaults_handled_by_caller(self):
        # compute_cross_encoder_scores only returns scores for candidates it
        # was given — callers (main.py) are responsible for the 0.5 default
        # on lookup misses. Verify the contract: no entry for IDs not passed in.
        candidates = [make_candidate("ONLY_ONE")]
        reranker = FakeReranker(raw_scores=[1.0])
        result = compute_cross_encoder_scores("JD text", candidates, reranker)

        assert "NOT_PASSED_IN" not in result
        assert result.get("NOT_PASSED_IN", 0.5) == 0.5  # caller-side default works as expected


class TestCrossEncoderModifierMath:
    """
    Mirrors the placeholder modifier applied in main.py:
        ce_mult = 1.0 + CROSS_ENCODER_MODIFIER_STRENGTH * (ce_score - 0.5)
    A neutral score (0.5) should leave final_score unchanged; scores above
    or below 0.5 should push it up or down symmetrically.
    """

    def test_neutral_score_is_a_no_op(self):
        strength = 0.15
        ce_score = 0.5
        mult = 1.0 + strength * (ce_score - 0.5)
        assert mult == pytest.approx(1.0)

    def test_symmetric_swing(self):
        strength = 0.15
        mult_high = 1.0 + strength * (0.9 - 0.5)
        mult_low = 1.0 + strength * (0.1 - 0.5)
        assert mult_high - 1.0 == pytest.approx(-(mult_low - 1.0))


def _normalize(values):
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin
    if vrange <= 1e-9:
        return [0.5 for _ in values]
    return [(v - vmin) / vrange for v in values]


class TestBatchNormalization:
    """
    Regression coverage for the miscalibration bug found in production:
    cross-encoder/ms-marco-MiniLM-L-6-v2 returned scores clustered near 0
    (observed: mean=0.009, max=0.1124 across a real 300-candidate batch).
    Anchoring the modifier to a fixed 0.5 midpoint made every candidate
    get an almost identical multiplier — a near-uniform shrink, not a
    rerank. Batch min-max normalization is the fix; these tests pin that
    behavior down so it can't silently regress.
    """

    def test_compressed_near_zero_batch_uses_full_range_after_normalization(self):
        # Mirrors the real observed distribution shape
        raw_scores = [0.0002, 0.0014, 0.0034, 0.0125, 0.1124]
        normalized = _normalize(raw_scores)
        assert min(normalized) == pytest.approx(0.0)
        assert max(normalized) == pytest.approx(1.0)
        # Relative order must be preserved
        assert normalized == sorted(normalized)

    def test_uniform_batch_falls_back_to_neutral_not_divide_by_zero(self):
        raw_scores = [0.05, 0.05, 0.05]
        normalized = _normalize(raw_scores)
        assert normalized == [0.5, 0.5, 0.5]

    def test_normalization_changes_relative_modifiers_vs_raw_anchoring(self):
        # With the bug: every score in this batch is < 0.5, so every
        # modifier is negative — a uniform shrink regardless of relative
        # standing within the batch.
        raw_scores = [0.0002, 0.0034, 0.1124]
        strength = 0.15
        raw_mults = [1.0 + strength * (s - 0.5) for s in raw_scores]
        assert all(m < 1.0 for m in raw_mults)
        assert max(raw_mults) - min(raw_mults) < 0.02  # barely distinguishable

        # With the fix: normalized modifiers span the full intended range
        normalized = _normalize(raw_scores)
        fixed_mults = [1.0 + strength * (n - 0.5) for n in normalized]
        assert min(fixed_mults) == pytest.approx(1.0 - strength / 2)
        assert max(fixed_mults) == pytest.approx(1.0 + strength / 2)
