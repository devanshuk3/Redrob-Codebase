"""
Ranking/recommendation system experience scorer (TASK 4 — strengthened).

Detects evidence of ranking, recommendation, matching, and
learning-to-rank experience. These carry more weight than generic AI.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import RANKING_KEYWORDS, RANKING_SKILL_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


def score_ranking(candidate: Candidate) -> float:
    """
    Score ranking/recommendation system experience.

    Scans career descriptions, titles, skills, headline, and summary
    for ranking-domain keywords.

    Returns:
        Normalized score [0, 1].
    """
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

    matches = count_keyword_matches(combined, RANKING_KEYWORDS)

    # Graduated scoring with better granularity
    if matches >= 9:
        score = 1.0
    elif matches >= 7:
        score = 0.90
    elif matches >= 5:
        score = 0.78
    elif matches >= 4:
        score = 0.65
    elif matches >= 3:
        score = 0.52
    elif matches >= 2:
        score = 0.38
    elif matches >= 1:
        score = 0.22
    else:
        score = 0.0

    # Bonus for ranking-related skill names
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in RANKING_SKILL_KEYWORDS):
            skill_bonus += 0.08

    score = min(1.0, score + min(0.25, skill_bonus))
    return round(score, 4)
