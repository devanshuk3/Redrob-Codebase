"""
Feature builder — orchestrates all sub-scorers and produces
the combined structured score (CHANGE 8 — expanded features).

Structured sub-scores:
  skill_score, retrieval_score, ranking_score, evaluation_score,
  production_score, experience_score, education_bonus, certification_bonus
"""

from typing import Dict, Any

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.features.skill_extractor import score_skills
from src.features.experience_scorer import score_experience
from src.features.education_scorer import score_education
from src.features.certification_scorer import score_certifications
from src.features.career_analyzer import score_career
from src.features.behavioral_scorer import score_behavioral
from src.features.retrieval_scorer import score_retrieval
from src.features.ranking_scorer import score_ranking
from src.features.evaluation_scorer import score_evaluation
from src.features.production_scorer import score_production
from src.utils.config import Config


def build_structured_features(
    candidate: Candidate,
    jd_features: JDFeatures,
) -> Dict[str, Any]:
    """
    Compute all feature scores for a single candidate.

    Returns:
        Dictionary containing:
        - candidate_id
        - All individual sub-scores
        - behavioral_score (separate from structured)
        - structured_score (weighted combination of sub-scores)
    """
    # ── Compute individual sub-scores ────────────────────────────────
    skill = score_skills(candidate, jd_features)
    retrieval = score_retrieval(candidate)
    ranking = score_ranking(candidate)
    evaluation = score_evaluation(candidate)
    production = score_production(candidate)
    experience = score_experience(candidate, jd_features)
    education = score_education(candidate)
    certification = score_certifications(candidate)
    career = score_career(candidate, jd_features)
    behavioral = score_behavioral(candidate)

    # ── Compute structured score (weighted combination) ──────────────
    # These weights correspond to Config.STRUCT_WEIGHT_* values
    structured_score = (
        Config.STRUCT_WEIGHT_SKILL * skill
        + Config.STRUCT_WEIGHT_RETRIEVAL * retrieval
        + Config.STRUCT_WEIGHT_RANKING * ranking
        + Config.STRUCT_WEIGHT_EVALUATION * evaluation
        + Config.STRUCT_WEIGHT_PRODUCTION * production
        + Config.STRUCT_WEIGHT_EXPERIENCE * experience
        + Config.STRUCT_WEIGHT_EDUCATION * education
        + Config.STRUCT_WEIGHT_CERTIFICATION * certification
    )

    # Normalize — the weights should sum to 1.0, but let's ensure it
    total_weight = (
        Config.STRUCT_WEIGHT_SKILL
        + Config.STRUCT_WEIGHT_RETRIEVAL
        + Config.STRUCT_WEIGHT_RANKING
        + Config.STRUCT_WEIGHT_EVALUATION
        + Config.STRUCT_WEIGHT_PRODUCTION
        + Config.STRUCT_WEIGHT_EXPERIENCE
        + Config.STRUCT_WEIGHT_EDUCATION
        + Config.STRUCT_WEIGHT_CERTIFICATION
    )
    if total_weight > 0:
        structured_score = structured_score / total_weight

    return {
        "candidate_id": candidate.candidate_id,
        "skill_score": round(skill, 4),
        "retrieval_score": round(retrieval, 4),
        "ranking_score": round(ranking, 4),
        "evaluation_score": round(evaluation, 4),
        "production_score": round(production, 4),
        "experience_score": round(experience, 4),
        "career_score": round(career, 4),
        "education_score": round(education, 4),
        "certification_score": round(certification, 4),
        "behavioral_score": round(behavioral, 4),
        "structured_score": round(structured_score, 4),
    }
