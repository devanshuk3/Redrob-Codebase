"""
Evaluation framework experience scorer (CHANGE 2).

Detects evidence of evaluation metrics, A/B testing, and
experiment framework experience.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import EVALUATION_KEYWORDS
from src.utils.text_utils import count_keyword_matches


def score_evaluation(candidate: Candidate) -> float:
    """
    Score evaluation framework experience.

    Scans career descriptions, titles, skills, headline, and summary
    for evaluation-domain keywords (NDCG, MAP, MRR, A/B testing, etc.).

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

    matches = count_keyword_matches(combined, EVALUATION_KEYWORDS)

    if matches >= 6:
        score = 1.0
    elif matches >= 4:
        score = 0.85
    elif matches >= 3:
        score = 0.65
    elif matches >= 2:
        score = 0.45
    elif matches >= 1:
        score = 0.25
    else:
        score = 0.0

    return round(score, 4)
