"""
Experience scoring — total years, role relevance, ideal range matching.
"""

from rapidfuzz import fuzz

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.utils.config import Config


def score_experience(candidate: Candidate, jd_features: JDFeatures) -> float:
    """
    Score candidate's experience against JD expectations.

    Components:
    1. Years-in-range score (40%): how well total experience matches ideal range
    2. Role relevance score (40%): current title similarity to JD domain keywords
    3. Recency score (20%): bonus for recent relevant work

    Returns:
        Normalized score [0, 1].
    """
    # ── 1. Years in range ────────────────────────────────────────────
    yoe = candidate.years_of_experience
    exp_range = jd_features.experience_range or [Config.EXPERIENCE_IDEAL_MIN, Config.EXPERIENCE_IDEAL_MAX]
    ideal_min, ideal_max = exp_range[0], exp_range[1] if len(exp_range) > 1 else exp_range[0] + 4

    if ideal_min <= yoe <= ideal_max:
        years_score = 1.0
    elif yoe < ideal_min:
        years_score = max(0.0, yoe / ideal_min)
    else:
        # Over-experienced: gentle decay
        over = yoe - ideal_max
        years_score = max(0.2, 1.0 - (over * 0.08))

    # ── 2. Role relevance ────────────────────────────────────────────
    title = candidate.current_title.lower() if candidate.current_title else ""
    domain_keywords = jd_features.domain_keywords + jd_features.must_have_skills[:5]

    role_score = 0.0
    if title and domain_keywords:
        best_match = max(
            fuzz.partial_ratio(title, kw) for kw in domain_keywords
        )
        role_score = min(1.0, best_match / 100.0)

    # ── 3. Recency score ─────────────────────────────────────────────
    recency_score = 0.0
    if candidate.career_history:
        current_roles = [c for c in candidate.career_history if c.is_current]
        if current_roles:
            current = current_roles[0]
            desc_lower = current.description.lower() if current.description else ""
            if domain_keywords:
                keyword_hits = sum(1 for kw in domain_keywords if kw in desc_lower)
                recency_score = min(1.0, keyword_hits / max(3, len(domain_keywords) * 0.3))

    # ── Combine ──────────────────────────────────────────────────────
    score = 0.40 * years_score + 0.40 * role_score + 0.20 * recency_score
    return round(min(1.0, score), 4)


def compute_experience_fit_score(
    yoe: float,
    min_exp: float = 5.0,
    max_exp: float = 9.0,
    return_distance: bool = False
):
    """
    Compute experience fit score.
    Uses a smooth distance-based curve.
    
    Inside range -> 1.0
    1 year outside -> ~0.9
    3 years outside -> ~0.75
    6+ years outside -> ~0.5
    
    Do not use hard thresholds.
    """
    if yoe < min_exp:
        distance = min_exp - yoe
    elif yoe > max_exp:
        distance = yoe - max_exp
    else:
        distance = 0.0

    if distance == 0.0:
        score = 1.0
    elif distance <= 1.0:
        # Interpolate between (0, 1.0) and (1.0, 0.9)
        score = 1.0 - 0.1 * distance
    elif distance <= 3.0:
        # Interpolate between (1.0, 0.9) and (3.0, 0.75)
        score = 0.9 - 0.075 * (distance - 1.0)
    elif distance <= 6.0:
        # Interpolate between (3.0, 0.75) and (6.0, 0.5)
        score = 0.75 - (0.25 / 3.0) * (distance - 3.0)
    else:
        # For > 6.0, decay slowly to a floor of 0.3
        score = max(0.3, 0.5 - 0.02 * (distance - 6.0))

    if return_distance:
        return round(score, 4), round(distance, 2)
    return round(score, 4)


