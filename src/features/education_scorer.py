"""
Education scoring — lightweight bonus only (CHANGE 4 — de-prioritized).

Education collectively contributes <= 5% of the structured score.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import DEGREE_WEIGHTS, RELEVANT_EDUCATION_FIELDS, TIER_WEIGHTS
from src.utils.text_utils import normalize_text


def score_education(candidate: Candidate) -> float:
    """
    Lightweight education bonus score [0, 1].

    Components:
    - Degree level (50%)
    - Field relevance (35%)
    - Tier bonus (15%)

    Returns:
        Normalized score [0, 1].
    """
    if not candidate.education:
        return 0.1  # Small default — don't penalize heavily for missing education

    best_score = 0.0

    for edu in candidate.education:
        degree_lower = normalize_text(edu.degree)
        field_lower = normalize_text(edu.field_of_study)
        tier = normalize_text(edu.tier)

        # Degree level score
        degree_score = 0.3  # default for unknown degrees
        for key, weight in DEGREE_WEIGHTS.items():
            if key in degree_lower:
                degree_score = weight
                break

        # Field relevance
        field_score = 0.2  # default for unrelated fields
        for relevant_field in RELEVANT_EDUCATION_FIELDS:
            if relevant_field in field_lower or field_lower in relevant_field:
                field_score = 1.0
                break

        # Tier bonus
        tier_score = TIER_WEIGHTS.get(tier, 0.3)

        # Combined for this education entry
        entry_score = 0.50 * degree_score + 0.35 * field_score + 0.15 * tier_score
        best_score = max(best_score, entry_score)

    return round(min(1.0, best_score), 4)
