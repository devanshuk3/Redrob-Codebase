"""
Feature builder — orchestrates all sub-scorers and produces
the combined structured score (TASK 4 — LLM hype penalty audit, TASK 7 — assessment scores).

Structured sub-scores:
  skill_score, retrieval_score, ranking_score, evaluation_score,
  production_score, experience_score, education_bonus, certification_bonus

Additional modifiers:
  llm_hype_penalty — penalizes LLM-only profiles without real ML depth
  experience_fit_score — penalizes candidates outside target range
  consistency_score — consistency of profile
  trap_probability — probability of synthetic/trap profile
"""

import math
from typing import Dict, Any, Optional

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.features.skill_extractor import score_skills, score_career_domain_keywords
from src.features.experience_scorer import score_experience, compute_experience_fit_score
from src.features.education_scorer import score_education
from src.features.certification_scorer import score_certifications
from src.features.career_analyzer import score_career
from src.features.behavioral_scorer import score_behavioral
from src.features.retrieval_scorer import score_retrieval
from src.features.ranking_scorer import score_ranking
from src.features.evaluation_scorer import score_evaluation
from src.features.production_scorer import score_production
from src.ingestion.honeypot_filter import compute_trap_probability, _compute_consistency_score as compute_consistency_score
from src.utils.config import Config
from src.utils.constants import LLM_HYPE_KEYWORDS
from src.utils.text_utils import count_keyword_matches

# Pre-compute total structured weight once at module load (sum of 8 Config constants)
_TOTAL_WEIGHT = (
    Config.STRUCT_WEIGHT_SKILL
    + Config.STRUCT_WEIGHT_RETRIEVAL
    + Config.STRUCT_WEIGHT_RANKING
    + Config.STRUCT_WEIGHT_EVALUATION
    + Config.STRUCT_WEIGHT_PRODUCTION
    + Config.STRUCT_WEIGHT_EXPERIENCE
    + Config.STRUCT_WEIGHT_EDUCATION
    + Config.STRUCT_WEIGHT_CERTIFICATION
)


def _compute_llm_hype_penalty(
    candidate: Candidate,
    retrieval: float,
    ranking: float,
    evaluation: float,
    production: float,
    combined_text: Optional[str] = None,
) -> float:
    """
    Compute LLM-hype penalty (TASK 4).

    The JD explicitly warns: candidates whose AI experience consists
    mostly of recent LangChain/OpenAI projects should be ranked lower.

    Trigger: heavy presence of LLM hype keywords but weak presence
    of retrieval/ranking/evaluation/production keywords.

    Returns:
        Penalty value [0.0, Config.LLM_HYPE_PENALTY_MAX].
    """
    # Build combined text
    if combined_text is None:
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

        combined_text = " ".join(parts)

    if not combined_text.strip():
        return 0.0

    hype_hits = count_keyword_matches(combined_text, LLM_HYPE_KEYWORDS)
    if hype_hits <= 2:
        return 0.0  # minimal hype mention is fine

    # Weak depth defined by low domain sub-scores
    domain_depth = retrieval + ranking + evaluation + production
    if domain_depth >= 1.2:
        return 0.0  # they have real depth — no penalty

    # Graduated penalty: less domain depth -> higher penalty
    # If domain_depth = 0.0 -> penalty is max (0.12)
    # If domain_depth = 1.2 -> penalty is 0.0
    scale = (1.2 - domain_depth) / 1.2
    penalty = scale * Config.LLM_HYPE_PENALTY_MAX

    # Scale with number of hype hits (max reached at 6+ hits)
    hype_factor = min(1.0, hype_hits / 6.0)
    penalty *= hype_factor

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
        - experience_fit_score
        - consistency_score
        - trap_probability
        - assessment_score
        - assessment_modifier
    """
    # --- Pre-compute shared text blobs once per candidate ---
    # career_text: lowercased career titles + descriptions
    career_parts = []
    for career_entry in candidate.career_history:
        if career_entry.description:
            career_parts.append(career_entry.description.lower())
        if career_entry.title:
            career_parts.append(career_entry.title.lower())
    career_text = " ".join(career_parts)

    # other_text: headline + summary + skill names (lowercased)
    other_parts = []
    if candidate.headline:
        other_parts.append(candidate.headline.lower())
    if candidate.summary:
        other_parts.append(candidate.summary.lower())
    skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
    other_parts.append(skill_text)
    other_text = " ".join(other_parts)

    # combined_text: career + headline + summary + skills (used by production scorer & hype penalty)
    combined_text = (career_text + " " + other_text) if career_text else other_text

    # Compute individual sub-scores (passing pre-computed text)
    retrieval = score_retrieval(candidate, career_text=career_text, other_text=other_text)
    ranking = score_ranking(candidate, career_text=career_text, other_text=other_text)
    evaluation = score_evaluation(candidate, career_text=career_text, other_text=other_text)
    production = score_production(candidate, combined_text=combined_text)
    experience = score_experience(candidate, jd_features)
    education = score_education(candidate)
    certification = score_certifications(candidate)
    career, career_detail = score_career(candidate, jd_features)
    behavioral = score_behavioral(candidate)

    # Career domain keyword scoring (TASK B)
    career_keyword_scores = score_career_domain_keywords(candidate, career_text=career_text)

    # Blend domain scores: 70% structured scorer + 30% career keyword scorer
    retrieval = 0.70 * retrieval + 0.30 * career_keyword_scores.get("retrieval", 0.0)
    ranking   = 0.70 * ranking   + 0.30 * career_keyword_scores.get("ranking", 0.0)
    evaluation = 0.70 * evaluation + 0.30 * career_keyword_scores.get("evaluation", 0.0)
    production = 0.70 * production + 0.30 * career_keyword_scores.get("production", 0.0)

    # Skill Assessment Modifier & Score (TASK 7)
    scores = candidate.redrob_signals.skill_assessment_scores or {}
    if not scores:
        assessment_score = 0.0
        assessment_modifier = 1.0
    else:
        # Find scores matching must-have or preferred skills from JD
        from src.utils.text_utils import canonicalize_skill
        jd_skills = {canonicalize_skill(s) for s in jd_features.all_skills()}
        relevant_scores = []
        for skill_name, val in scores.items():
            if canonicalize_skill(skill_name) in jd_skills:
                try:
                    relevant_scores.append(float(val))
                except (ValueError, TypeError):
                    continue
        if not relevant_scores:
            assessment_score = 0.0
            assessment_modifier = 1.0
        else:
            assessment_score = max(relevant_scores)
            if assessment_score >= 90:
                assessment_modifier = 1.05
            elif assessment_score >= 80:
                assessment_modifier = 1.03
            elif assessment_score >= 70:
                assessment_modifier = 1.01
            elif assessment_score >= 60:
                assessment_modifier = 1.00
            elif assessment_score >= 50:
                assessment_modifier = 0.99
            elif assessment_score >= 40:
                assessment_modifier = 0.97
            else:
                assessment_modifier = 0.95

    # Retrieve skill score and apply assessment_modifier
    skill = score_skills(candidate, jd_features)
    skill = skill * assessment_modifier

    # Experience Fit Score (TASK 1)
    experience_fit, experience_dist = compute_experience_fit_score(
        candidate.years_of_experience,
        jd_features.min_experience,
        jd_features.max_experience,
        return_distance=True
    )

    # Consistency & Trap Probability (TASK 2 & 3)
    consistency = compute_consistency_score(candidate)
    trap_prob = compute_trap_probability(candidate)

    # Education Consistency Score (TASK 2)
    from src.ingestion.honeypot_filter import get_candidate_issues
    issues = get_candidate_issues(candidate)
    edu_consistency = issues.get("EDUCATION_CHRONOLOGY", {}).get("score", 1.0)

    # Technical and Behavioral Rewards (TASK 6 Boosts)
    # A. Scale Boost: if career history shows scale / production / latency metrics
    has_scale_terms = any(w in career_text for w in ["scale", "million", "billion", "qps", "latency", "throughput", "production", "shipped", "deployed"])
    scale_boost = 1.03 if has_scale_terms else 1.0

    # B. Behavioral Boost: strong response rate and active GitHub
    sig = candidate.redrob_signals
    behavioral_boost = 1.0
    if sig.recruiter_response_rate >= 0.8 and sig.github_activity_score > 70:
        behavioral_boost = 1.03

    # Compute LLM hype penalty
    hype_penalty = _compute_llm_hype_penalty(candidate, retrieval, ranking, evaluation, production, combined_text=combined_text)

    # Compute structured score (weighted combination)
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

    # Normalize — the weights sum to 1.0, using pre-computed module-level constant
    if _TOTAL_WEIGHT > 0:
        structured_score = structured_score / _TOTAL_WEIGHT

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
        "consulting_ratio": round(career_detail.get("consulting_ratio", 0.0), 4),
        "education_score": round(education, 4),
        "certification_score": round(certification, 4),
        "behavioral_score": round(behavioral, 4),
        "structured_score": round(structured_score, 4),
        "llm_hype_penalty": round(hype_penalty, 4),
        "experience_fit_score": round(experience_fit, 4),
        "experience_distance": round(experience_dist, 2),
        "consistency_score": round(consistency, 4),
        "education_consistency_score": round(edu_consistency, 4),
        "trap_probability": round(trap_prob, 4),
        "assessment_score": round(assessment_score, 2),
        "assessment_modifier": round(assessment_modifier, 2),
        "scale_boost": round(scale_boost, 2),
        "behavioral_boost": round(behavioral_boost, 2),
        "career_keyword_scores": career_keyword_scores,
    }

