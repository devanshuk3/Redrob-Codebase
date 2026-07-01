"""Tests for ranking module: score combiner, ranker, reasoning, submission."""
import os, sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.candidate_parser import Candidate, SkillEntry, CareerEntry, RedrobSignals
from src.features.jd_feature_mapper import JDFeatures
from src.ranking.score_combiner import combine_scores
from src.ranking.ranker import rank_candidates
from src.ranking.reasoning_generator import generate_reasoning
from src.ranking.submission_builder import validate_submission


def _make_features(n=200):
    feats, sem, qual = {}, {}, {}
    for i in range(n):
        cid = f"CAND_{i:04d}"
        feats[cid] = {"structured_score": (n - i) / n, "behavioral_score": 0.5,
                       "skill_score": 0.6, "retrieval_score": 0.4, "ranking_score": 0.3,
                       "production_score": 0.5, "evaluation_score": 0.2, "experience_score": 0.7}
        sem[cid] = (n - i) / n
        qual[cid] = 0.8
    return feats, sem, qual


class TestScoreCombiner:
    def test_combine_produces_final_score(self):
        feats, sem, qual = _make_features(10)
        results = combine_scores(feats, sem, qual)
        assert len(results) == 10
        assert all("final_score" in r for r in results)

    def test_scores_in_range(self):
        feats, sem, qual = _make_features(10)
        for r in combine_scores(feats, sem, qual):
            assert 0.0 <= r["final_score"] <= 1.0


class TestRanker:
    def test_top_100_selection(self):
        feats, sem, qual = _make_features(200)
        scored = combine_scores(feats, sem, qual)
        ranked = rank_candidates(scored, top_n=100)
        assert len(ranked) == 100
        assert ranked[0]["rank"] == 1
        assert ranked[-1]["rank"] == 100

    def test_unique_ids_and_ranks(self):
        feats, sem, qual = _make_features(200)
        ranked = rank_candidates(combine_scores(feats, sem, qual), top_n=100)
        ids = [r["candidate_id"] for r in ranked]
        ranks = [r["rank"] for r in ranked]
        assert len(set(ids)) == 100
        assert len(set(ranks)) == 100

    def test_scores_non_increasing(self):
        feats, sem, qual = _make_features(200)
        ranked = rank_candidates(combine_scores(feats, sem, qual), top_n=100)
        scores = [r["final_score"] for r in ranked]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-9


class TestReasoningGenerator:
    def test_reasoning_uses_real_data(self):
        jd = JDFeatures(must_have_skills=["python", "machine learning"],
                        domain_keywords=["search", "retrieval"])
        cand = Candidate(candidate_id="CAND_0001", years_of_experience=6.0,
                         current_title="ML Engineer", current_company="BigCo",
                         skills=[SkillEntry(name="Python", proficiency="expert")],
                         redrob_signals=RedrobSignals(recruiter_response_rate=0.85))
        ranked = [{"candidate_id": "CAND_0001", "final_score": 0.9, "rank": 1,
                   "retrieval_score": 0.7, "ranking_score": 0.5, "production_score": 0.6,
                   "evaluation_score": 0.3, "behavioral_score": 0.8}]
        result = generate_reasoning(ranked, {"CAND_0001": cand}, jd)
        assert "reasoning" in result[0]
        assert len(result[0]["reasoning"]) > 10
        assert "6" in result[0]["reasoning"]  # years of experience


class TestSubmissionValidator:
    def test_valid_submission(self):
        rows = [{"candidate_id": f"C_{i}", "rank": i, "score": 1.0 - i * 0.005,
                 "reasoning": f"Reason {i}"} for i in range(1, 101)]
        df = pd.DataFrame(rows)
        errors = validate_submission(df)
        assert len(errors) == 0

    def test_wrong_row_count(self):
        rows = [{"candidate_id": f"C_{i}", "rank": i, "score": 1.0,
                 "reasoning": "x"} for i in range(1, 51)]
        assert len(validate_submission(pd.DataFrame(rows))) > 0

    def test_dynamic_row_count(self):
        from src.utils.config import Config
        original_top_n = Config.TOP_N
        try:
            Config.TOP_N = 10
            rows = [{"candidate_id": f"C_{i}", "rank": i, "score": 1.0 - i * 0.01,
                     "reasoning": f"Reason {i}"} for i in range(1, 11)]
            df = pd.DataFrame(rows)
            errors = validate_submission(df)
            assert len(errors) == 0
        finally:
            Config.TOP_N = original_top_n

    def test_empty_row_count(self):
        from src.utils.config import Config
        original_top_n = Config.TOP_N
        try:
            Config.TOP_N = 0
            df = pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])
            errors = validate_submission(df)
            assert len(errors) == 0
        finally:
            Config.TOP_N = original_top_n

    def test_duplicate_ids(self):
        rows = [{"candidate_id": "C_1", "rank": i, "score": 1.0,
                 "reasoning": "x"} for i in range(1, 101)]
        assert len(validate_submission(pd.DataFrame(rows))) > 0
