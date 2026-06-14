"""
Retrieval system experience scorer (TASK 4 — strengthened).

Detects evidence of search/retrieval/RAG/vector-search experience
in career descriptions, titles, and skills.

Retrieval/search/vector-DB skills carry MORE weight than generic AI keywords.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import RETRIEVAL_KEYWORDS, RETRIEVAL_SKILL_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


def score_retrieval(candidate: Candidate) -> float:
    """
    Score retrieval/search system experience.

    Scans career descriptions, titles, skills, headline, and summary
    for retrieval-domain keywords. Uses graduated scoring with
    stronger ceiling and skill-name bonuses.

    Returns:
        Normalized score [0, 1].
    """
    # Build combined text from all relevant fields
    parts = []

    # Career descriptions and titles
    for career in candidate.career_history:
        if career.description:
            parts.append(career.description.lower())
        if career.title:
            parts.append(career.title.lower())

    # Headline and summary
    if candidate.headline:
        parts.append(candidate.headline.lower())
    if candidate.summary:
        parts.append(candidate.summary.lower())

    # Skill names
    skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
    parts.append(skill_text)

    combined = " ".join(parts)

    if not combined.strip():
        return 0.0

    matches = count_keyword_matches(combined, RETRIEVAL_KEYWORDS)

    # Graduated scoring — more granular for better discrimination
    if matches >= 10:
        score = 1.0
    elif matches >= 8:
        score = 0.92
    elif matches >= 6:
        score = 0.82
    elif matches >= 5:
        score = 0.72
    elif matches >= 4:
        score = 0.62
    elif matches >= 3:
        score = 0.50
    elif matches >= 2:
        score = 0.38
    elif matches >= 1:
        score = 0.22
    else:
        score = 0.0

    # Bonus for retrieval-related skill names (stronger bonus)
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in RETRIEVAL_SKILL_KEYWORDS):
            skill_bonus += 0.08

    score = min(1.0, score + min(0.25, skill_bonus))
    return round(score, 4)
