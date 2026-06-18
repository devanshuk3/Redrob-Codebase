"""
Retrieval system experience scorer (TASK 4 — strengthened, TASK 6 — career history hardened).

Detects evidence of search/retrieval/RAG/vector-search experience
in career descriptions, titles, and skills.

Retrieval/search/vector-DB skills carry MORE weight than generic AI keywords.
"""

from typing import Optional

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import RETRIEVAL_KEYWORDS, RETRIEVAL_SKILL_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


def score_retrieval(
    candidate: Candidate,
    career_text: Optional[str] = None,
    other_text: Optional[str] = None,
) -> float:
    """
    Score retrieval/search system experience.

    Scans career descriptions, titles, skills, headline, and summary
    for retrieval-domain keywords. Uses graduated scoring with
    stronger ceiling and skill-name bonuses.

    Args:
        candidate: Candidate object.
        career_text: Pre-computed lowercased career text (titles+descriptions).
        other_text: Pre-computed lowercased other text (headline+summary+skills).

    Returns:
        Normalized score [0, 1].
    """
    # 1. Career history text
    if career_text is None:
        career_parts = []
        for career in candidate.career_history:
            if career.description:
                career_parts.append(career.description.lower())
            if career.title:
                career_parts.append(career.title.lower())
        career_text = " ".join(career_parts)

    # 2. Other profile text (headline, summary, skills)
    if other_text is None:
        other_parts = []
        if candidate.headline:
            other_parts.append(candidate.headline.lower())
        if candidate.summary:
            other_parts.append(candidate.summary.lower())
        
        skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
        other_parts.append(skill_text)
        other_text = " ".join(other_parts)

    # Count matches
    career_matches = count_keyword_matches(career_text, RETRIEVAL_KEYWORDS)
    other_matches = count_keyword_matches(other_text, RETRIEVAL_KEYWORDS)
    total_matches = career_matches + other_matches

    if total_matches == 0:
        return 0.0

    # Base score on total matches
    if total_matches >= 10:
        score = 1.0
    elif total_matches >= 8:
        score = 0.92
    elif total_matches >= 6:
        score = 0.82
    elif total_matches >= 5:
        score = 0.72
    elif total_matches >= 4:
        score = 0.62
    elif total_matches >= 3:
        score = 0.50
    elif total_matches >= 2:
        score = 0.38
    else:
        score = 0.22

    # Bonus for retrieval-related skill names (stronger bonus)
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in RETRIEVAL_SKILL_KEYWORDS):
            skill_bonus += 0.08

    score = min(1.0, score + min(0.25, skill_bonus))

    # TASK 6 Rule: career evidence matters more.
    # If 0 matches in career history, penalize heavily (only skills/summary mention it)
    if career_matches == 0:
        score *= 0.4
    else:
        # Increase confidence when retrieval appears in career history
        boost = min(0.15, career_matches * 0.03)
        score = min(1.0, score + boost)

    return round(score, 4)
