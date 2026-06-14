"""
Ranking/recommendation system experience scorer (TASK 4 — strengthened, TASK 6 — career history hardened).

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
    # 1. Career history text
    career_parts = []
    for career in candidate.career_history:
        if career.description:
            career_parts.append(career.description.lower())
        if career.title:
            career_parts.append(career.title.lower())
    career_text = " ".join(career_parts)

    # 2. Other profile text (headline, summary, skills)
    other_parts = []
    if candidate.headline:
        other_parts.append(candidate.headline.lower())
    if candidate.summary:
        other_parts.append(candidate.summary.lower())
    
    skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
    other_parts.append(skill_text)
    other_text = " ".join(other_parts)

    # Count matches
    career_matches = count_keyword_matches(career_text, RANKING_KEYWORDS)
    other_matches = count_keyword_matches(other_text, RANKING_KEYWORDS)
    total_matches = career_matches + other_matches

    if total_matches == 0:
        return 0.0

    # Base score on total matches
    if total_matches >= 9:
        score = 1.0
    elif total_matches >= 7:
        score = 0.90
    elif total_matches >= 5:
        score = 0.78
    elif total_matches >= 4:
        score = 0.65
    elif total_matches >= 3:
        score = 0.52
    elif total_matches >= 2:
        score = 0.38
    else:
        score = 0.22

    # Bonus for ranking-related skill names
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in RANKING_SKILL_KEYWORDS):
            skill_bonus += 0.08

    score = min(1.0, score + min(0.25, skill_bonus))

    # TASK 6 Rule: career evidence matters more.
    # If 0 matches in career history, penalize heavily
    if career_matches == 0:
        score *= 0.4
    else:
        # Increase confidence when ranking appears in career history
        boost = min(0.15, career_matches * 0.03)
        score = min(1.0, score + boost)

    return round(score, 4)
