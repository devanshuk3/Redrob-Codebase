"""
Tests for the semantic module: text building and similarity computation.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.candidate_parser import Candidate, CareerEntry, SkillEntry
from src.semantic.text_builder import build_candidate_text, build_jd_text
from src.semantic.semantic_ranker import compute_semantic_scores


class TestTextBuilder:
    def test_candidate_text_not_empty(self):
        c = Candidate(candidate_id="T1", headline="ML Engineer",
                       summary="Search systems.", skills=[SkillEntry(name="Python")],
                       career_history=[CareerEntry(title="ML Eng", description="Retrieval.")])
        text = build_candidate_text(c)
        assert len(text) > 0 and "Python" in text

    def test_truncation(self):
        c = Candidate(candidate_id="T2", summary="x" * 3000)
        assert len(build_candidate_text(c, max_chars=500)) <= 500

    def test_jd_strips_markdown(self):
        text = build_jd_text("# Title\n- Python\n* ML")
        assert "#" not in text and "Python" in text


class TestSemanticRanker:
    def test_cosine_basic(self):
        jd = np.array([1.0, 0.0, 0.0])
        cands = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scores = compute_semantic_scores(jd, cands, ["C1", "C2"])
        assert scores["C1"] == 1.0 and scores["C2"] == 0.0

    def test_output_range(self):
        rng = np.random.RandomState(42)
        scores = compute_semantic_scores(rng.randn(384), rng.randn(10, 384),
                                         [f"C{i}" for i in range(10)])
        for s in scores.values():
            assert 0.0 <= s <= 1.0
