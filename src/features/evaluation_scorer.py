"""
Evaluation framework experience scorer (TASK 5 — strengthened).

Detects evidence of evaluation metrics, A/B testing, and
experiment framework experience. These signals carry increased
weight as the JD explicitly requires ranking evaluation expertise.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import EVALUATION_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


# Skill names that indicate evaluation expertise
_EVAL_SKILL_KEYWORDS = {
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    "metrics", "experiment", "benchmark", "statistical",
    "precision", "recall", "f1", "roc", "auc",
}


def score_evaluation(candidate: Candidate) -> float:
    """
    Score evaluation framework experience.

    Scans career descriptions, titles, skills, headline, and summary
    for evaluation-domain keywords (NDCG, MAP, MRR, A/B testing, etc.).
    Increased scoring thresholds to reward genuine evaluation expertise.

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

    # Graduated scoring — more responsive to evaluation signals
    if matches >= 8:
        score = 1.0
    elif matches >= 6:
        score = 0.90
    elif matches >= 4:
        score = 0.78
    elif matches >= 3:
        score = 0.62
    elif matches >= 2:
        score = 0.45
    elif matches >= 1:
        score = 0.25
    else:
        score = 0.0

    # Bonus for evaluation-related skill names
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in _EVAL_SKILL_KEYWORDS):
            skill_bonus += 0.08

    score = min(1.0, score + min(0.20, skill_bonus))
    return round(score, 4)
