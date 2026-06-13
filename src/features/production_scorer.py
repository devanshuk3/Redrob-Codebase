"""
Production experience scorer (CHANGE 3).

Detects evidence of production-grade engineering: deployment,
serving, monitoring, scale, distributed systems, etc.

Production deployment experience ranks higher than research-only.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import PRODUCTION_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


def score_production(candidate: Candidate) -> float:
    """
    Score production engineering experience.

    Scans career descriptions, titles, skills, headline, and summary
    for production/deployment/scale keywords.

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

    matches = count_keyword_matches(combined, PRODUCTION_KEYWORDS)

    if matches >= 10:
        score = 1.0
    elif matches >= 7:
        score = 0.85
    elif matches >= 5:
        score = 0.7
    elif matches >= 3:
        score = 0.5
    elif matches >= 2:
        score = 0.35
    elif matches >= 1:
        score = 0.2
    else:
        score = 0.0

    # Bonus for production-related skills
    prod_skill_keywords = {"docker", "kubernetes", "aws", "azure", "gcp",
                           "mlops", "airflow", "kafka", "redis", "cicd",
                           "terraform", "ansible", "jenkins", "mlflow"}
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in prod_skill_keywords):
            skill_bonus += 0.05

    score = min(1.0, score + min(0.2, skill_bonus))
    return round(score, 4)
