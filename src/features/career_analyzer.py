"""
Career analyzer — searches career descriptions for domain-relevant keywords.
Produces a general career relevance score.
"""

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.utils.text_utils import count_keyword_matches


def score_career(candidate: Candidate, jd_features: JDFeatures) -> float:
    """
    Analyze career descriptions for domain-relevant keywords from the JD.

    Scans all career descriptions for JD domain keywords and must-have skills.

    Returns:
        Normalized score [0, 1].
    """
    if not candidate.career_history:
        return 0.0

    # Combine all career descriptions into one text
    all_descriptions = " ".join(
        c.description.lower() for c in candidate.career_history
        if c.description
    )
    all_titles = " ".join(
        c.title.lower() for c in candidate.career_history
        if c.title
    )
    combined = all_descriptions + " " + all_titles

    if not combined.strip():
        return 0.0

    # Use JD domain keywords + must-have skills as search terms
    keywords = jd_features.domain_keywords + jd_features.must_have_skills
    if not keywords:
        return 0.0

    matches = count_keyword_matches(combined, keywords)
    # Normalize: hitting ~40% of keywords is a strong signal
    score = min(1.0, matches / (len(keywords) * 0.4))

    return round(score, 4)


def get_career_highlights(candidate: Candidate, jd_features: JDFeatures) -> list:
    """Return career keywords found (for reasoning)."""
    highlights = []
    keywords = jd_features.domain_keywords + jd_features.must_have_skills[:10]
    for career in candidate.career_history:
        desc = (career.description or "").lower()
        title = (career.title or "").lower()
        text = desc + " " + title
        for kw in keywords:
            if kw in text and kw not in highlights:
                highlights.append(kw)
    return highlights[:10]
