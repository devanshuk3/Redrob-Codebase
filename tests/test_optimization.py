"""
Tests for TASK 1: Verify skills_score is non-zero and meaningful.
Proves that retrieval/search-domain skills produce measurable score increases.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.candidate_parser import (
    Candidate, CareerEntry, EducationEntry, SkillEntry, RedrobSignals,
)
from src.features.jd_feature_mapper import JDFeatures
from src.features.skill_extractor import (
    score_skills, get_matched_skills, compute_skill_strength,
    _proficiency_weight, _duration_weight, _endorsement_bonus, _assessment_modifier,
)
from src.features.feature_builder import build_structured_features, _compute_llm_hype_penalty


def _make_jd(**kwargs):
    defaults = {
        "must_have_skills": [
            "python", "machine learning", "embeddings", "retrieval systems",
            "vector databases", "search systems", "ranking systems",
            "pinecone", "milvus", "faiss", "weaviate", "elasticsearch",
            "ndcg", "mrr", "map", "rag", "llms", "distributed systems",
        ],
        "preferred_skills": [
            "docker", "kubernetes", "mlops", "qdrant", "opensearch",
            "information retrieval", "vector search", "bm25",
        ],
        "negative_signals": [],
        "experience_range": [5.0, 9.0],
        "domain_keywords": ["search", "retrieval", "ranking", "embedding", "production"],
        "raw_text": "Senior ML Engineer focused on search and retrieval systems.",
    }
    defaults.update(kwargs)
    return JDFeatures(**defaults)


def _make_candidate(**overrides):
    defaults = {
        "candidate_id": "CAND_SKILL_TEST",
        "headline": "ML Engineer",
        "summary": "Building search systems.",
        "years_of_experience": 6.0,
        "current_title": "ML Engineer",
        "current_company": "TestCo",
        "career_history": [
            CareerEntry(
                company="TestCo", title="ML Engineer",
                duration_months=36, is_current=True,
                description="Building retrieval and search systems.",
            ),
        ],
        "education": [EducationEntry(degree="M.S.", field_of_study="Computer Science")],
        "skills": [],
        "redrob_signals": RedrobSignals(),
    }
    defaults.update(overrides)
    return Candidate(**defaults)


# TASK 1: Skills Score is Non-Zero

class TestSkillsScoreNonZero:
    """Prove that skills_score is actually computed and non-zero."""

    def test_milvus_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Milvus", proficiency="advanced", endorsements=10, duration_months=24),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Milvus should produce non-zero skill score, got {score}"

    def test_faiss_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="FAISS", proficiency="advanced", endorsements=15, duration_months=36),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"FAISS should produce non-zero skill score, got {score}"

    def test_embeddings_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Embeddings", proficiency="expert", endorsements=20, duration_months=48),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Embeddings should produce non-zero skill score, got {score}"

    def test_pinecone_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Pinecone", proficiency="intermediate", endorsements=5, duration_months=12),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Pinecone should produce non-zero skill score, got {score}"

    def test_qdrant_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Qdrant", proficiency="advanced", endorsements=8, duration_months=18),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Qdrant should produce non-zero skill score, got {score}"

    def test_vector_search_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Vector Search", proficiency="expert", endorsements=20, duration_months=36),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Vector Search should produce non-zero skill score, got {score}"

    def test_retrieval_produces_score(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Information Retrieval", proficiency="expert", endorsements=25, duration_months=60),
        ])
        score = score_skills(cand, jd)
        assert score > 0, f"Information Retrieval should produce non-zero skill score, got {score}"

    def test_multiple_matching_skills_high_score(self):
        """A candidate matching many JD skills should score significantly higher."""
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=72),
            SkillEntry(name="Machine Learning", proficiency="expert", endorsements=40, duration_months=60),
            SkillEntry(name="Embeddings", proficiency="advanced", endorsements=20, duration_months=36),
            SkillEntry(name="FAISS", proficiency="advanced", endorsements=10, duration_months=24),
            SkillEntry(name="Milvus", proficiency="intermediate", endorsements=5, duration_months=12),
            SkillEntry(name="Docker", proficiency="advanced", endorsements=15, duration_months=48),
        ])
        score = score_skills(cand, jd)
        assert score > 0.15, f"Multiple matching skills should produce measurable score, got {score}"


class TestSkillsContributeToFinal:
    """Prove skills_score is included in structured_score and final_score."""

    def test_skill_score_in_structured(self):
        jd = _make_jd()
        cand = _make_candidate(skills=[
            SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=72),
            SkillEntry(name="Machine Learning", proficiency="advanced", endorsements=30, duration_months=48),
        ])
        features = build_structured_features(cand, jd)
        assert features["skill_score"] > 0, \
            f"skill_score should be non-zero in features, got {features['skill_score']}"
        assert features["structured_score"] > 0, \
            f"structured_score should include skill_score, got {features['structured_score']}"

    def test_no_skills_lower_structured(self):
        jd = _make_jd()
        cand_with = _make_candidate(skills=[
            SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=72),
            SkillEntry(name="Machine Learning", proficiency="advanced", endorsements=30, duration_months=48),
        ])
        cand_without = _make_candidate(skills=[])
        feat_with = build_structured_features(cand_with, jd)
        feat_without = build_structured_features(cand_without, jd)
        assert feat_with["structured_score"] > feat_without["structured_score"], \
            "Candidate with matching skills should have higher structured_score"


# TASK 2: Skill Metadata Exploitation

class TestSkillMetadata:
    """Verify proficiency, duration, endorsements affect scoring."""

    def test_advanced_beats_beginner(self):
        assert _proficiency_weight("advanced") > _proficiency_weight("beginner")

    def test_expert_at_top(self):
        assert _proficiency_weight("expert") >= _proficiency_weight("advanced")

    def test_long_duration_beats_short(self):
        assert _duration_weight(60) > _duration_weight(6)

    def test_zero_duration_neutral(self):
        assert _duration_weight(0) == 0.5

    def test_endorsement_bonus_scales(self):
        assert _endorsement_bonus(50) > _endorsement_bonus(5)
        assert _endorsement_bonus(0) == 0.0

    def test_advanced_long_beats_beginner_short(self):
        """Core requirement: long-term advanced > short-term beginner."""
        jd = _make_jd()
        advanced_long = _make_candidate(skills=[
            SkillEntry(name="Python", proficiency="advanced", endorsements=30, duration_months=60),
        ])
        beginner_short = _make_candidate(skills=[
            SkillEntry(name="Python", proficiency="beginner", endorsements=0, duration_months=3),
        ])
        score_adv = score_skills(advanced_long, jd)
        score_beg = score_skills(beginner_short, jd)
        assert score_adv > score_beg, \
            f"Advanced long-term ({score_adv}) should beat beginner short-term ({score_beg})"

    def test_skill_strength_uses_all_metadata(self):
        skill = SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=72)
        strength = compute_skill_strength(skill)
        assert strength > 0.5, f"Expert+long+endorsed should have high strength, got {strength}"


# TASK 3: LLM Hype Penalty

class TestLLMHypePenalty:
    """Verify LLM-hype candidates are penalized."""

    def test_no_penalty_for_balanced(self):
        cand = _make_candidate(skills=[
            SkillEntry(name="LangChain"),
            SkillEntry(name="Retrieval Systems"),
            SkillEntry(name="Machine Learning"),
            SkillEntry(name="Search"),
        ])
        penalty = _compute_llm_hype_penalty(cand, retrieval=0.8, ranking=0.5, evaluation=0.6, production=0.6)
        assert penalty == 0.0, f"Balanced candidate should have no hype penalty, got {penalty}"

    def test_penalty_for_hype_heavy(self):
        cand = _make_candidate(
            headline="ChatGPT Expert | LangChain Developer",
            summary="Building chatbots with OpenAI and LangChain. Prompt engineering specialist.",
            skills=[
                SkillEntry(name="LangChain"),
                SkillEntry(name="OpenAI"),
                SkillEntry(name="ChatGPT"),
                SkillEntry(name="Prompt Engineering"),
                SkillEntry(name="Claude"),
            ],
            career_history=[CareerEntry(
                description="Built chatbots using LangChain and OpenAI APIs. Prompt engineering for GPT-4.",
            )],
        )
        penalty = _compute_llm_hype_penalty(cand, retrieval=0.0, ranking=0.0, evaluation=0.0, production=0.0)
        assert penalty > 0, f"Hype-heavy candidate should have non-zero penalty, got {penalty}"



# TASK 8: Assessment Scores

class TestAssessmentScores:
    """Verify skill_assessment_scores modify confidence."""

    def test_high_assessment_boosts(self):
        mod = _assessment_modifier("NLP", {"NLP": 88})
        assert mod > 1.0, f"High assessment should boost, got {mod}"

    def test_low_assessment_reduces(self):
        mod = _assessment_modifier("NLP", {"NLP": 38})
        assert mod < 1.0, f"Low assessment should reduce, got {mod}"

    def test_no_assessment_neutral(self):
        mod = _assessment_modifier("NLP", {})
        assert mod == 1.0, f"No assessment should be neutral, got {mod}"

    def test_assessment_integrated_in_skill_score(self):
        jd = _make_jd()
        cand_high = _make_candidate(
            skills=[SkillEntry(name="Machine Learning", proficiency="advanced", duration_months=48)],
            redrob_signals=RedrobSignals(skill_assessment_scores={"Machine Learning": 92}),
        )
        cand_low = _make_candidate(
            skills=[SkillEntry(name="Machine Learning", proficiency="advanced", duration_months=48)],
            redrob_signals=RedrobSignals(skill_assessment_scores={"Machine Learning": 30}),
        )
        score_high = score_skills(cand_high, jd)
        score_low = score_skills(cand_low, jd)
        assert score_high > score_low, \
            f"High assessment ({score_high}) should beat low assessment ({score_low})"


# TASK 1 & 2 & 3: Experience Fit & Consistency & Trap Probability

class TestPipelineOptimizationFeatures:
    """Verify experience fit scoring, consistency checks, and trap probability detection."""

    def test_experience_fit_scoring(self):
        from src.features.experience_scorer import compute_experience_fit_score
        assert compute_experience_fit_score(7.0) == 1.0     # Ideal range (5-9)
        assert compute_experience_fit_score(4.0) == 0.9     # 1 year outside (5-9) -> 0.9
        assert compute_experience_fit_score(12.0) == 0.75   # 3 years outside (5-9) -> 0.75
        assert compute_experience_fit_score(15.0) == 0.5    # 6 years outside (5-9) -> 0.5
        assert compute_experience_fit_score(2.0) == 0.75    # 3 years outside (5-9) -> 0.75

    def test_education_timeline_chronology(self):
        from src.ingestion.honeypot_filter import check_education_timeline
        from src.ingestion.candidate_parser import EducationEntry
        # 1. Valid chronology
        valid_cand = _make_candidate(
            education=[
                EducationEntry(degree="B.Tech", start_year=2012, end_year=2016),
                EducationEntry(degree="M.Tech", start_year=2017, end_year=2019),
                EducationEntry(degree="PhD", start_year=2020, end_year=2024),
            ]
        )
        score, issues = check_education_timeline(valid_cand)
        assert score == 1.0, f"Expected 1.0, got {score}. Issues: {issues}"

        # 2. Invalid chronology (M.Tech before B.Tech)
        invalid_cand = _make_candidate(
            education=[
                EducationEntry(degree="M.Tech", start_year=2003, end_year=2006),
                EducationEntry(degree="B.Tech", start_year=2012, end_year=2017),
            ]
        )
        score, issues = check_education_timeline(invalid_cand)
        assert score < 1.0, "Expected penalty for invalid chronology"
        assert any("Chronology error" in iss for iss in issues)

        # 3. Suspicious overlap of multiple Bachelors
        overlap_cand = _make_candidate(
            education=[
                EducationEntry(degree="B.Tech", start_year=2006, end_year=2010),
                EducationEntry(degree="B.E", start_year=2007, end_year=2012),
            ]
        )
        score, issues = check_education_timeline(overlap_cand)
        assert score < 1.0, "Expected penalty for suspicious Bachelors overlap"

    def test_profile_consistency_score(self):
        from src.ingestion.honeypot_filter import _compute_consistency_score
        # 1. Consistent candidate
        consistent_cand = _make_candidate(
            years_of_experience=6.0,
            current_title="ML Engineer",
            education=[EducationEntry(degree="B.Tech", field_of_study="CS", start_year=2016, end_year=2020)],
            career_history=[CareerEntry(company="BigCo", title="ML Engineer", duration_months=72, description="retrieval systems development")]
        )
        score_consistent = _compute_consistency_score(consistent_cand)
        assert score_consistent == 1.0, f"Expected 1.0, got {score_consistent}"

        # 2. Inconsistent candidate (Senior title with < 3 YOE)
        inconsistent_cand = _make_candidate(
            years_of_experience=1.5,
            current_title="Senior Principal Architect",
            career_history=[CareerEntry(company="A", title="CTO", duration_months=18)]
        )
        score_inconsistent = _compute_consistency_score(inconsistent_cand)
        assert score_inconsistent < 1.0, f"Expected penalty, got {score_inconsistent}"

    def test_trap_probability_detection(self):
        from src.ingestion.honeypot_filter import compute_trap_probability
        # 1. Normal candidate should have low trap probability
        normal_cand = _make_candidate(
            years_of_experience=6.0,
            current_title="ML Engineer",
            career_history=[CareerEntry(company="BigCo", title="ML Engineer", duration_months=72, description="retrieval systems development")]
        )
        prob_normal = compute_trap_probability(normal_cand)
        assert prob_normal < 0.2, f"Normal profile should have low trap probability, got {prob_normal}"

        # 2. Trap candidate (many contradictions, low YOE but senior title, skill stuffing)
        trap_cand = _make_candidate(
            years_of_experience=1.0,
            current_title="CTO",
            skills=[SkillEntry(name="Photoshop"), SkillEntry(name="Sales"), SkillEntry(name="Accounting"), SkillEntry(name="FAISS"), SkillEntry(name="Milvus")],
            education=[EducationEntry(degree="B.Tech", field_of_study="CS", start_year=2020, end_year=2018)], # Impossible timeline
            career_history=[CareerEntry(company="A", title="CTO", duration_months=12)]
        )
        prob_trap = compute_trap_probability(trap_cand)
        assert prob_trap >= 0.4, f"Suspicious profile should have high trap probability, got {prob_trap}"

