"""
Tests for the features module: all scorers and feature builder.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.candidate_parser import (
    Candidate, CareerEntry, EducationEntry, SkillEntry, RedrobSignals, parse_candidate,
)
from src.features.jd_feature_mapper import parse_job_description, JDFeatures
from src.features.skill_extractor import score_skills, get_matched_skills
from src.features.experience_scorer import score_experience
from src.features.education_scorer import score_education
from src.features.certification_scorer import score_certifications
from src.features.career_analyzer import score_career
from src.features.behavioral_scorer import score_behavioral
from src.features.retrieval_scorer import score_retrieval
from src.features.ranking_scorer import score_ranking
from src.features.evaluation_scorer import score_evaluation
from src.features.production_scorer import score_production
from src.features.feature_builder import build_structured_features


# ── Helpers ──────────────────────────────────────────────────────────

def _make_jd_features(**kwargs):
    defaults = {
        "must_have_skills": ["python", "machine learning", "pytorch", "retrieval systems"],
        "preferred_skills": ["docker", "kubernetes", "mlops"],
        "negative_signals": [],
        "experience_range": [5.0, 9.0],
        "domain_keywords": ["search", "retrieval", "ranking", "embedding", "production"],
        "raw_text": "Senior ML Engineer focused on search and retrieval systems.",
    }
    defaults.update(kwargs)
    return JDFeatures(**defaults)


def _make_candidate(**overrides):
    defaults = {
        "candidate_id": "CAND_TEST",
        "headline": "Senior ML Engineer",
        "summary": "Building retrieval and ranking systems at scale.",
        "years_of_experience": 6.0,
        "current_title": "ML Engineer",
        "current_company": "BigTech",
        "career_history": [
            CareerEntry(
                company="BigTech", title="ML Engineer",
                duration_months=36, is_current=True,
                description="Built semantic search and ranking systems serving millions of users. "
                            "Deployed models to production with monitoring and A/B testing.",
            ),
        ],
        "education": [
            EducationEntry(
                degree="M.S.", field_of_study="Computer Science",
                institution="Stanford", end_year=2018, tier="tier_1",
            ),
        ],
        "skills": [
            SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=72),
            SkillEntry(name="PyTorch", proficiency="advanced", endorsements=30, duration_months=48),
            SkillEntry(name="Machine Learning", proficiency="expert", endorsements=40, duration_months=60),
            SkillEntry(name="Elasticsearch", proficiency="intermediate", endorsements=10, duration_months=24),
        ],
        "redrob_signals": RedrobSignals(
            profile_completeness_score=90.0,
            recruiter_response_rate=0.85,
            github_activity_score=80.0,
            open_to_work_flag=True,
            interview_completion_rate=0.9,
            notice_period_days=30,
            saved_by_recruiters_30d=10,
            last_active_date="2026-06-01",
        ),
    }
    defaults.update(overrides)
    return Candidate(**defaults)


# ── JD Feature Mapper Tests ─────────────────────────────────────────

class TestJDFeatureMapper:
    def test_parse_from_text(self):
        jd_text = """
# Senior ML Engineer

## Requirements
- Python expertise
- Machine Learning experience
- 5-9 years of experience
- Retrieval systems
- Search ranking

## Preferred
- Docker
- Kubernetes
- MLOps
"""
        features = parse_job_description(jd_text=jd_text)
        assert len(features.must_have_skills) > 0
        assert features.experience_range == [5.0, 9.0]
        assert "python" in features.must_have_skills

    def test_empty_jd(self):
        features = parse_job_description(jd_text="No specific requirements listed.")
        # Should still produce defaults
        assert isinstance(features, JDFeatures)
        assert features.experience_range  # should have defaults


# ── Skill Extractor Tests ────────────────────────────────────────────

class TestSkillExtractor:
    def test_exact_match(self):
        jd = _make_jd_features()
        candidate = _make_candidate()
        score = score_skills(candidate, jd)
        assert score > 0.4, f"Should match several skills, got {score}"

    def test_no_skills(self):
        jd = _make_jd_features()
        candidate = _make_candidate(skills=[])
        score = score_skills(candidate, jd)
        assert score == 0.0

    def test_no_jd_skills(self):
        jd = _make_jd_features(must_have_skills=[], preferred_skills=[])
        candidate = _make_candidate()
        score = score_skills(candidate, jd)
        assert score == 0.0

    def test_matched_skills_list(self):
        jd = _make_jd_features()
        candidate = _make_candidate()
        matched = get_matched_skills(candidate, jd)
        assert "Python" in matched


# ── Experience Scorer Tests ──────────────────────────────────────────

class TestExperienceScorer:
    def test_ideal_range(self):
        jd = _make_jd_features(experience_range=[5.0, 9.0])
        candidate = _make_candidate(years_of_experience=7.0)
        score = score_experience(candidate, jd)
        assert score > 0.5

    def test_under_experienced(self):
        jd = _make_jd_features(experience_range=[5.0, 9.0])
        candidate = _make_candidate(years_of_experience=1.0)
        score = score_experience(candidate, jd)
        assert score < 0.5

    def test_zero_experience(self):
        jd = _make_jd_features()
        candidate = _make_candidate(years_of_experience=0.0)
        score = score_experience(candidate, jd)
        assert score >= 0.0


# ── Education Scorer Tests ───────────────────────────────────────────

class TestEducationScorer:
    def test_cs_masters(self):
        candidate = _make_candidate()
        score = score_education(candidate)
        assert score > 0.5

    def test_no_education(self):
        candidate = _make_candidate(education=[])
        score = score_education(candidate)
        assert score == 0.1  # small default

    def test_unrelated_education(self):
        candidate = _make_candidate(education=[
            EducationEntry(degree="B.A.", field_of_study="History", tier="tier_4"),
        ])
        score = score_education(candidate)
        assert score < 0.5


# ── Retrieval Scorer Tests ───────────────────────────────────────────

class TestRetrievalScorer:
    def test_strong_retrieval(self):
        candidate = _make_candidate(
            career_history=[CareerEntry(
                description="Built semantic search and retrieval systems. "
                            "Implemented RAG pipeline with vector search using FAISS. "
                            "Designed hybrid search combining dense and sparse retrieval.",
            )],
        )
        score = score_retrieval(candidate)
        assert score > 0.5, f"Strong retrieval background should score high, got {score}"

    def test_no_retrieval(self):
        candidate = _make_candidate(
            headline="Accountant",
            summary="Managing financial records.",
            career_history=[CareerEntry(description="Prepared tax returns and audits.")],
            skills=[SkillEntry(name="Excel")],
        )
        score = score_retrieval(candidate)
        assert score < 0.3


# ── Ranking Scorer Tests ────────────────────────────────────────────

class TestRankingScorer:
    def test_strong_ranking(self):
        candidate = _make_candidate(
            career_history=[CareerEntry(
                description="Built recommendation systems and ranking models. "
                            "Implemented learning to rank with LambdaMART. "
                            "Designed personalization engine for content recommendation.",
            )],
        )
        score = score_ranking(candidate)
        assert score > 0.5

    def test_no_ranking(self):
        candidate = _make_candidate(
            headline="Graphic Designer",
            summary="Creating visual designs.",
            career_history=[CareerEntry(description="Designed logos and branding.")],
            skills=[SkillEntry(name="Photoshop")],
        )
        score = score_ranking(candidate)
        assert score < 0.3


# ── Production Scorer Tests ──────────────────────────────────────────

class TestProductionScorer:
    def test_strong_production(self):
        candidate = _make_candidate(
            career_history=[CareerEntry(
                description="Deployed ML models to production. Monitored latency and "
                            "serving performance. Built CI/CD pipelines with Docker and Kubernetes. "
                            "Managed distributed systems at scale with high availability.",
            )],
        )
        score = score_production(candidate)
        assert score > 0.5

    def test_research_only(self):
        candidate = _make_candidate(
            career_history=[CareerEntry(
                description="Published papers on theoretical machine learning. "
                            "Conducted experiments in Jupyter notebooks.",
            )],
            skills=[SkillEntry(name="Python")],
        )
        score = score_production(candidate)
        assert score < 0.5


# ── Behavioral Scorer Tests ──────────────────────────────────────────

class TestBehavioralScorer:
    def test_active_candidate(self):
        candidate = _make_candidate()
        score = score_behavioral(candidate)
        assert score > 0.5

    def test_inactive_candidate(self):
        candidate = _make_candidate(
            redrob_signals=RedrobSignals(
                recruiter_response_rate=0.1,
                github_activity_score=-1,
                open_to_work_flag=False,
                interview_completion_rate=0.2,
                notice_period_days=150,
                profile_completeness_score=20.0,
                last_active_date="2024-01-01",
            ),
        )
        score = score_behavioral(candidate)
        assert score < 0.5


# ── Feature Builder Tests ────────────────────────────────────────────

class TestFeatureBuilder:
    def test_builds_all_scores(self):
        jd = _make_jd_features()
        candidate = _make_candidate()
        features = build_structured_features(candidate, jd)
        assert "candidate_id" in features
        assert "skill_score" in features
        assert "retrieval_score" in features
        assert "ranking_score" in features
        assert "evaluation_score" in features
        assert "production_score" in features
        assert "experience_score" in features
        assert "behavioral_score" in features
        assert "structured_score" in features
        # All scores should be 0-1
        for key, val in features.items():
            if key not in ("candidate_id", "scale_boost", "behavioral_boost", "assessment_modifier", "experience_distance"):
                assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"
