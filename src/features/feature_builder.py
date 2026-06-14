"""
Feature builder — orchestrates all sub-scorers and produces
the combined structured score (TASK 3 — LLM hype penalty, TASK 11 — rebalanced).

Structured sub-scores:
  skill_score, retrieval_score, ranking_score, evaluation_score,
  production_score, experience_score, education_bonus, certification_bonus

Additional modifiers:
  llm_hype_penalty — penalizes LLM-only profiles without real ML depth
"""

import math
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
from src.utils.constants import LLM_HYPE_KEYWORDS, REAL_ML_DEPTH_KEYWORDS
from src.utils.text_utils import count_keyword_matches


def _compute_llm_hype_penalty(candidate: Candidate) -> float:
    """
    Compute LLM-hype penalty (TASK 3).

    The JD explicitly warns: candidates whose AI experience consists
    mostly of recent LangChain/OpenAI projects should be ranked lower.

    Trigger: heavy presence of LLM hype keywords but weak presence
    of retrieval/ranking/evaluation/production keywords.

    Returns:
        Penalty value [0.0, Config.LLM_HYPE_PENALTY_MAX].
    """
    # Build combined text
    parts = []
    for career in candidate.career_history:
        if career.description:
            parts.append(career.description.lower())
        if career.title:
            parts.append(career.title.lower())
    if candidate.headline:
        parts.append(candidate.headline.lower())
    if candidate.summary:
        parts.append(candidate.summary.lower())

    skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
    parts.append(skill_text)

    combined = " ".join(parts)
    if not combined.strip():
        return 0.0

    hype_hits = count_keyword_matches(combined, LLM_HYPE_KEYWORDS)
    depth_hits = count_keyword_matches(combined, REAL_ML_DEPTH_KEYWORDS)

    # Only penalize when hype significantly outweighs depth
    if hype_hits <= 2:
        return 0.0  # minimal hype mention is fine

    # Ratio: if hype mentions >> depth mentions, apply penalty
    if depth_hits >= hype_hits:
        return 0.0  # they have real depth — no penalty

    # Scale penalty by how much hype outweighs depth
    hype_ratio = hype_hits / max(1, depth_hits)
    if hype_ratio < 2.0:
        return 0.0  # acceptable balance

    # Graduated penalty: ratio 2.0→4.0+ maps to 0.03→max
    raw_penalty = min(1.0, (hype_ratio - 2.0) / 3.0)
    penalty = raw_penalty * Config.LLM_HYPE_PENALTY_MAX

    return round(penalty, 4)


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
        - llm_hype_penalty (for debugging and transparency)
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

    # ── Compute LLM hype penalty (TASK 3) ────────────────────────────
    hype_penalty = _compute_llm_hype_penalty(candidate)

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

    # Apply LLM hype penalty — subtracts from structured score
    structured_score = max(0.0, structured_score - hype_penalty)

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
        "llm_hype_penalty": round(hype_penalty, 4),
    }
