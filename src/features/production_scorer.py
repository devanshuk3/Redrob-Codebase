"""
Production experience scorer (TASK 6 — strengthened).

Detects evidence of production-grade engineering: deployment,
serving, monitoring, scale, distributed systems, etc.

Production deployment experience ranks higher than research/hobby projects.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import PRODUCTION_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


# Skills that indicate real production engineering
_PROD_SKILL_KEYWORDS = {
    "docker", "kubernetes", "aws", "azure", "gcp",
    "mlops", "airflow", "kafka", "redis", "cicd",
    "terraform", "ansible", "jenkins", "mlflow",
    "kubeflow", "grpc", "rest api", "microservices",
    "distributed systems", "model serving", "production",
    "monitoring", "observability", "prometheus", "grafana",
    "datadog", "elasticsearch", "system design",
}


def score_production(candidate: Candidate) -> float:
    """
    Score production engineering experience.

    Scans career descriptions, titles, skills, headline, and summary
    for production/deployment/scale keywords. Strongly rewards
    evidence of real production deployments.

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

    # Graduated scoring — more granular for production signals
    if matches >= 12:
        score = 1.0
    elif matches >= 10:
        score = 0.92
    elif matches >= 8:
        score = 0.82
    elif matches >= 6:
        score = 0.72
    elif matches >= 4:
        score = 0.58
    elif matches >= 3:
        score = 0.45
    elif matches >= 2:
        score = 0.32
    elif matches >= 1:
        score = 0.18
    else:
        score = 0.0

    # Bonus for production-related skills (slightly stronger)
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in _PROD_SKILL_KEYWORDS):
            skill_bonus += 0.06

    score = min(1.0, score + min(0.25, skill_bonus))
    return round(score, 4)
